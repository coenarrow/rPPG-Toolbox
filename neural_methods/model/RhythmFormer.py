"""RhythmFormer: hierarchical temporal periodic transformer for rPPG.

Zou et al., "RhythmFormer: Extracting Patterned rPPG Signals based on
Hierarchical Temporal Periodic Transformer",
https://arxiv.org/abs/2402.12788 — the official implementation
(https://github.com/zizheng-guo/RhythmFormer), with its bi-level routing
attention helpers vendored from BiFormer
(https://github.com/rayleizhu/BiFormer).

The architecture is unchanged from the original: the same two-path fusion
stem over the frame and its four temporal differences, the same 1x4x4 patch
embedding, the same three TPT stages (temporal periodic transformer blocks at
2-, 4- and 8-frame patches with top-40 region routing), the same single-plane
readout. One thing differs from the published network: the stem takes
``in_channels`` inputs (the interface's channel count) instead of 3, and its
difference path takes ``4 * in_channels`` — the hard-coded ``12`` upstream is
four frame differences of three channels. The readout (``ConvBlockLast``)
stays a single plane, a bare ``Conv1d`` with no activation, so an
absolute-class signal like ABP can be predicted directly in mmHg; a
multi-signal run is one complete copy of this network per trace
(``MultiTraceModel``), never a widened readout on a shared trunk. The loss is
the trainer's: RhythmFormer's published criterion adds frequency-domain
cross-entropy and label-distribution terms to negative Pearson, and those
belong in the interface's ``LOSS`` block, not here.

On ``configs/interfaces/rhythmformer_interface.yaml`` (128x128 frames,
160-frame windows) the forward pass is the published one, bit for bit:
loaded from the same weights, this module and the pre-migration one return
identical tensors on a paper-shaped clip.

Any other frame size from 16x16 and any window length are accepted through
three adaptive stages, each an exact no-op at the paper's shape:

* the original read the token grid as ``view(N, D, 64, H//4, W//4)``, true
  only where the stem's arithmetic happens to land on ``H//4``. Here the grid
  is read off the stem output's actual shape with einops;
* the TPT stages patch time by 2, 4 and 8 frames (stride-2 time convolutions
  undone by 2x upsamples), so the window has to divide by 8. The patch
  embedding's output is average-pooled in time to the nearest multiple of 8
  and the prediction is linearly interpolated back to the window length;
  at 160 frames neither runs;
* the bi-level routing attention cuts the token grid into ``REGION_SPLIT``
  regions per spatial axis, which only tiles the grid when the region size
  divides it. The grid is replicate-padded up to a whole number of regions
  and the attention output cropped back, and the routing ``topk`` is clamped
  to the number of regions there actually are. At the paper's 8x8 grid the
  padding is empty and 40 of 320 regions are routed, so neither fires.

No parameter is added, renamed or resized, so the upstream checkpoint loads
strictly. A clip backbone on the multi-signal contract: ``(B, C_in, T, H, W)``
in, ``(B, 1, T)`` out, preprocessing done by the dataset, the loss owned by
the trainer. All reshaping is einops.
"""

import math

import torch
from einops import rearrange, reduce, repeat
from timm.layers import DropPath, trunc_normal_
from torch import Tensor, nn
from torch.nn import functional as F

from neural_methods.model.shared import (
    nearest_multiple, require_min_frame, sum_spatial,
)

#: ``Fusion_Stem``'s stride-2 convolution and stride-2 max pool.
STEM_SPATIAL_STRIDE = 4

#: The patch embedding's ``(1, 4, 4)`` kernel and stride.
PATCH_SPATIAL = 4

#: The smallest frame the stem and the patch embedding leave a 1x1 token grid
#: of; anything smaller tokenizes to nothing.
MIN_FRAME = STEM_SPATIAL_STRIDE * PATCH_SPATIAL

#: Regions per spatial axis in the bi-level routing attention: the region size
#: is the token grid divided by this, i.e. 2x2 on the paper's 8x8 grid.
REGION_SPLIT = 4


# ---------------------------------------------------------------------------
# Bi-level routing attention, adapted from https://github.com/rayleizhu/BiFormer
# ---------------------------------------------------------------------------
def _grid2seq(x: Tensor, region_size: tuple, num_heads: int):
    """``(B, C, T, H, W)`` -> ``(B, nhead, nregion, region_volume, head_dim)``.

    Also returns the number of regions per t / row / column.
    """
    region_t, region_h, region_w = (n // r for n, r in zip(x.shape[2:], region_size))
    x = rearrange(x, "b (m d) (rt o) (rh p) (rw q) -> b m (rt rh rw) (o p q) d",
                  m=num_heads, o=region_size[0], p=region_size[1], q=region_size[2])
    return x, region_t, region_h, region_w


def _seq2grid(x: Tensor, region_t: int, region_h: int, region_w: int,
              region_size: tuple) -> Tensor:
    """``(B, nhead, nregion, region_volume, head_dim)`` -> ``(B, C, T, H, W)``."""
    return rearrange(x, "b m (rt rh rw) (o p q) d -> b (m d) (rt o) (rh p) (rw q)",
                     rt=region_t, rh=region_h, rw=region_w,
                     o=region_size[0], p=region_size[1], q=region_size[2])


def video_regional_routing_attention_torch(
        query: Tensor, key: Tensor, value: Tensor, scale: float,
        region_graph: Tensor, region_size: tuple,
        kv_region_size: tuple = None):
    """Token-to-token attention inside the routed regions.

    Args:
      query, key, value: ``(B, C, T, H, W)`` tensors.
      scale: the scale/temperature of the dot product.
      region_graph: ``(B, nhead, q_nregion, topk)``, the routed key regions.
      region_size: ``(rt, rh, rw)``, the region size for the queries.
      kv_region_size: the same for keys and values; defaults to ``region_size``.

    Returns the attended ``(B, C, T, H, W)`` tensor and the attention matrix.
    """
    kv_region_size = kv_region_size or region_size
    num_heads, q_nregion = region_graph.shape[1:3]

    # To sequence format, i.e. (bs, nhead, nregion, region_volume, head_dim).
    query, region_t, region_h, region_w = _grid2seq(query, region_size, num_heads)
    key, _, _, _ = _grid2seq(key, kv_region_size, num_heads)
    value, _, _, _ = _grid2seq(value, kv_region_size, num_heads)

    # Gather the routed keys and values. torch.gather does not broadcast, so
    # the graph is expanded to the shape it indexes into.
    kv_volume, head_dim = key.shape[3:]
    graph = repeat(region_graph, "b m s k -> b m s k r d", r=kv_volume, d=head_dim)
    key_g = torch.gather(repeat(key, "b m u r d -> b m s u r d", s=q_nregion),
                         dim=3, index=graph)
    value_g = torch.gather(repeat(value, "b m u r d -> b m s u r d", s=q_nregion),
                           dim=3, index=graph)

    # (bs, nhead, q_nregion, reg_volume, head_dim) against the topk*kv_volume
    # tokens each query region routed to.
    attn = (query * scale) @ rearrange(key_g, "b m s k r d -> b m s d (k r)")
    attn = torch.softmax(attn, dim=-1)
    output = attn @ rearrange(value_g, "b m s k r d -> b m s (k r) d")

    return _seq2grid(output, region_t, region_h, region_w, region_size), attn


class CDC_T(nn.Module):
    """Temporal center-difference 3-D convolution.

    From https://github.com/ZitongYu/PhysFormer/model/transformer_layer.py.
    ``theta`` mixes the plain convolution with the central-difference one;
    ``theta = 0`` is an ordinary ``Conv3d``.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal

        # Only the central difference over a temporal kernel > 1 is meaningful.
        if self.conv.weight.shape[2] > 1:
            # Summed over kh and then over kw, the order upstream's
            # ``.sum(2).sum(2)`` accumulates in: a single joint reduction over
            # both axes is the same number in exact arithmetic but not in
            # float32, and this kernel goes on to weight a convolution.
            kernel_diff = (sum_spatial(self.conv.weight[:, :, 0])
                           + sum_spatial(self.conv.weight[:, :, 2]))
            kernel_diff = rearrange(kernel_diff, "cout cin -> cout cin 1 1 1")
            out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias,
                                stride=self.conv.stride, padding=0,
                                dilation=self.conv.dilation, groups=self.conv.groups)
            return out_normal - self.theta * out_diff

        return out_normal


class video_BRA(nn.Module):
    """Bi-level routing attention over a spatio-temporal token grid.

    The grid is cut into regions, each query region attends to the ``topk``
    key regions its pooled descriptor scores highest, and attention then runs
    token to token inside that selection. The region size is a forward-time
    quantity read off the grid, so no parameter here depends on the frame
    size or the window length.
    """

    def __init__(self, dim, num_heads=8, t_patch=8, qk_scale=None, topk=4,
                 side_dwconv=3):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim must be divisible by num_heads!'
        self.head_dim = self.dim // self.num_heads
        self.scale = qk_scale or self.dim ** -0.5
        self.topk = topk
        self.t_patch = t_patch  # frame of patch
        # side_dwconv, i.e. LCE in the Shunted Transformer.
        self.lepe = nn.Conv3d(dim, dim, kernel_size=side_dwconv, stride=1,
                              padding=side_dwconv // 2, groups=dim) if side_dwconv > 0 else \
            (lambda x: torch.zeros_like(x))
        # Unused by the published forward, which projects q, k and v
        # separately below; kept so the upstream checkpoint loads strictly.
        self.qkv_linear = nn.Conv3d(self.dim, 3 * self.dim, kernel_size=1)
        self.output_linear = nn.Conv3d(self.dim, self.dim, kernel_size=1)
        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=0.2),
            nn.BatchNorm3d(dim),
        )
        self.proj_k = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=0.2),
            nn.BatchNorm3d(dim),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
        )

    def forward(self, x: Tensor):
        frames, height, width = x.shape[2:]
        t_region = max(4 // self.t_patch, 1)
        region_size = (t_region, max(height // REGION_SPLIT, 1),
                       max(width // REGION_SPLIT, 1))

        # STEP 1: linear projection
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)

        # Any frame size: the regions have to tile the grid. Padding it up to
        # a whole number of them and cropping the attention output back is
        # empty at the paper's 8x8 token grid, which 2x2 regions tile exactly.
        # Replicate rather than zeros, so a padded region's pooled descriptor
        # stays a descriptor of real tokens.
        pad = tuple(-n % r for n, r in zip((frames, height, width), region_size))
        if any(pad):
            padding = (0, pad[2], 0, pad[1], 0, pad[0])
            q = F.pad(q, padding, mode="replicate")
            k = F.pad(k, padding, mode="replicate")
            v_pad = F.pad(v, padding, mode="replicate")
        else:
            v_pad = v

        # STEP 2: pre attention
        q_r = F.avg_pool3d(q.detach(), kernel_size=region_size, ceil_mode=True,
                           count_include_pad=False)
        k_r = F.avg_pool3d(k.detach(), kernel_size=region_size, ceil_mode=True,
                           count_include_pad=False)
        a_r = (rearrange(q_r, "n c t h w -> n (t h w) c")
               @ rearrange(k_r, "n c t h w -> n c (t h w)"))       # n(thw)(thw)
        # A short window on a small frame can leave fewer regions than the
        # paper routes; its deepest stage offers 320 and routes 40, so at the
        # paper's shape this clamp is not reached.
        idx_r = torch.topk(a_r, k=min(self.topk, a_r.shape[-1]), dim=-1).indices
        idx_r = repeat(idx_r, "n s k -> n m s k", m=self.num_heads)

        # STEP 3: refined attention
        output, _ = video_regional_routing_attention_torch(
            query=q, key=k, value=v_pad, scale=self.scale,
            region_graph=idx_r, region_size=region_size)
        if any(pad):
            output = output[:, :, :frames, :height, :width]

        output = output + self.lepe(v)      # nctHW
        return self.output_linear(output)   # nctHW


class video_BiFormerBlock(nn.Module):
    """One routing-attention block and its convolutional MLP."""

    def __init__(self, dim, drop_path=0., num_heads=4, t_patch=1, qk_scale=None,
                 topk=4, mlp_ratio=2, side_dwconv=5):
        super().__init__()
        self.t_patch = t_patch
        self.norm1 = nn.BatchNorm3d(dim)
        self.attn = video_BRA(dim=dim, num_heads=num_heads, t_patch=t_patch,
                              qk_scale=qk_scale, topk=topk, side_dwconv=side_dwconv)
        self.norm2 = nn.BatchNorm3d(dim)
        self.mlp = nn.Sequential(nn.Conv3d(dim, int(mlp_ratio * dim), kernel_size=1),
                                 nn.BatchNorm3d(int(mlp_ratio * dim)),
                                 nn.GELU(),
                                 nn.Conv3d(int(mlp_ratio * dim), int(mlp_ratio * dim),
                                           3, stride=1, padding=1),
                                 nn.BatchNorm3d(int(mlp_ratio * dim)),
                                 nn.GELU(),
                                 nn.Conv3d(int(mlp_ratio * dim), dim, kernel_size=1),
                                 nn.BatchNorm3d(dim),
                                 )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Fusion_Stem(nn.Module):
    """Two-path stem: the frame itself and its four temporal differences.

    ``in_channels`` is the only width that differs from the published network;
    at ``3`` the difference path takes the original's 12 planes, four frame
    differences of three channels.
    """

    def __init__(self, in_channels: int = 3, dim: int = 64, apha=0.5, belta=0.5):
        super().__init__()

        self.stem11 = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )

        self.stem12 = nn.Sequential(
            nn.Conv2d(4 * in_channels, dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )

        self.stem21 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        self.stem22 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        self.apha = apha
        self.belta = belta

    def forward(self, x):
        """``(N, D, C, H, W)`` -> ``(N*D, dim, H/4, W/4)``."""
        D = x.shape[1]
        x1 = torch.cat([x[:, :1], x[:, :1], x[:, :D - 2]], 1)
        x2 = torch.cat([x[:, :1], x[:, :D - 1]], 1)
        x3 = x
        x4 = torch.cat([x[:, 1:], x[:, D - 1:]], 1)
        x5 = torch.cat([x[:, 2:], x[:, D - 1:], x[:, D - 1:]], 1)
        diff = torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], 2)
        x_diff = self.stem12(rearrange(diff, "n d c h w -> (n d) c h w"))
        x = self.stem11(rearrange(x3, "n d c h w -> (n d) c h w"))

        # fusion layer 1
        x_path1 = self.apha * x + self.belta * x_diff
        x_path1 = self.stem21(x_path1)
        # fusion layer 2
        x_path2 = self.stem22(x_diff)
        return self.apha * x_path1 + self.belta * x_path2


class TPT_Block(nn.Module):
    """One temporal periodic transformer stage.

    Time is halved ``log2(t_patch)`` times by stride-2 convolutions, the
    routing-attention blocks run on the patched sequence, and the same number
    of 2x upsamples put the window length back.
    """

    def __init__(self, dim, depth, num_heads, t_patch, topk,
                 mlp_ratio=4., drop_path=0., side_dwconv=5):
        super().__init__()
        self.dim = dim
        self.depth = depth
        # --------- downsample layers & upsample layers ---------
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.layer_n = int(math.log(t_patch, 2))
        for i in range(self.layer_n):
            downsample_layer = nn.Sequential(
                nn.BatchNorm3d(dim),
                nn.Conv3d(dim, dim, kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            )
            self.downsample_layers.append(downsample_layer)
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=(2, 1, 1)),
                nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1, 0, 0)),
                nn.BatchNorm3d(dim),
                nn.ELU(),
            )
            self.upsample_layers.append(upsample_layer)
        # -------------------------------------------------------
        self.blocks = nn.ModuleList([
            video_BiFormerBlock(
                dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                num_heads=num_heads,
                t_patch=t_patch,
                topk=topk,
                mlp_ratio=mlp_ratio,
                side_dwconv=side_dwconv,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor):
        """``(N, C, D, H, W)`` -> ``(N, C, D, H, W)``."""
        for i in range(self.layer_n):
            x = self.downsample_layers[i](x)
        for blk in self.blocks:
            x = blk(x)
        for i in range(self.layer_n):
            x = self.upsample_layers[i](x)

        return x


class RhythmFormer(nn.Module):
    """RhythmFormer as a clip backbone: ``(B, in_channels, T, H, W)`` -> ``(B, 1, T)``.

    ``in_channels`` is the number of input channels (default 3); every other
    argument is a published hyperparameter at its published value.
    """

    def __init__(
        self,
        in_channels: int = 3,
        stem_dim: int = 64,
        head_dim: int = 16,
        embed_dim=(64, 64, 64),
        mlp_ratios=(1.5, 1.5, 1.5),
        depth=(2, 2, 2),
        t_patchs=(2, 4, 8),
        topks=(40, 40, 40),
        side_dwconv: int = 3,
        drop_path_rate: float = 0.,
    ):
        super().__init__()

        self.in_channels = in_channels
        #: The deepest stage patches time by 8, so the window must divide by it.
        self.temporal_stride = max(t_patchs)

        self.Fusion_Stem = Fusion_Stem(in_channels=in_channels, dim=stem_dim)
        self.patch_embedding = nn.Conv3d(
            stem_dim, embed_dim[0], kernel_size=(1, PATCH_SPATIAL, PATCH_SPATIAL),
            stride=(1, PATCH_SPATIAL, PATCH_SPATIAL))
        # The readout: one plane, bias kept (the trainer seeds it with the
        # trace's physiological prior) and deliberately no activation, so an
        # absolute-class signal is expressible in its own units.
        self.ConvBlockLast = nn.Conv1d(embed_dim[-1], 1, kernel_size=1, stride=1, padding=0)

        # -------------------------------------------------------------------
        self.stages = nn.ModuleList()
        nheads = [dim // head_dim for dim in embed_dim]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        for i in range(len(embed_dim)):
            stage = TPT_Block(dim=embed_dim[i],
                              depth=depth[i],
                              num_heads=nheads[i],
                              mlp_ratio=mlp_ratios[i],
                              drop_path=dp_rates[sum(depth[:i]):sum(depth[:i + 1])],
                              t_patch=t_patchs[i], topk=topks[i], side_dwconv=side_dwconv
                              )
            self.stages.append(stage)
        # -------------------------------------------------------------------

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def output_layers(self):
        """The activation-free readout: the final 1x1 conv."""
        return (self.ConvBlockLast,)

    def forward(self, x):
        """``(B, in_channels, T, H, W)`` -> ``(B, 1, T)``."""
        frames, height, width = x.shape[2:]
        require_min_frame("RhythmFormer", MIN_FRAME, height, width)

        x = self.Fusion_Stem(rearrange(x, "b c t h w -> b t c h w"))
        # The token grid is read off the stem's output, not asserted to be H/4.
        x = rearrange(x, "(b t) c h w -> b c t h w", t=frames)
        x = self.patch_embedding(x)            # [B, dim, T, H/16, W/16]

        # Any window length: the stages patch time by up to 8. At the paper's
        # 160-frame windows the target equals T and this is skipped.
        patched = nearest_multiple(frames, self.temporal_stride)
        if patched != frames:
            x = F.adaptive_avg_pool3d(x, (patched, x.shape[3], x.shape[4]))

        for stage in self.stages:
            x = stage(x)                       # [B, dim, T, gh, gw]

        # Pool the token grid away, row by row then across, the order
        # upstream's two ``torch.mean(., 3)`` calls average in; time survives.
        features_last = reduce(reduce(x, "b c t gh gw -> b c t gw", "mean"),
                               "b c t gw -> b c t", "mean")
        rPPG = self.ConvBlockLast(features_last)               # [B, 1, T']

        # Back to the window length if the temporal patches did not divide it.
        if rPPG.shape[-1] != frames:
            rPPG = F.interpolate(rPPG, size=frames, mode="linear", align_corners=False)
        return rPPG
