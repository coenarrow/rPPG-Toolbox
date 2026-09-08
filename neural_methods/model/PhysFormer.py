"""PhysFormer: temporal-difference transformer for physiological measurement.

Yu et al., https://arxiv.org/abs/2111.12082 — a combination of ``Physformer.py``
and ``transformer_layer.py`` from the official implementation
(https://github.com/ZitongYu/PhysFormer).

The architecture is unchanged from the original. One thing differs from the
published network: the first layer (``Stem0``'s conv) takes ``in_channels``
inputs (the interface's channel count) instead of 3. The readout
(``ConvBlockLast``) stays a single plane, a bare ``Conv1d`` with no
activation, so an absolute-class signal like ABP can be predicted directly in
mmHg; a multi-signal run is one complete copy of this network per trace
(``MultiTraceModel``), never a widened readout on a shared trunk. The losses
are the trainer's, not the original's: PhysFormer's published DLDL
frequency/KL term is the ``SPECTRAL`` component of the per-signal loss, not a
parallel criterion.

Everything between those two layers — the 3-D stem, the (4,4,4) tube
tokenization, the three temporal-difference transformer stages, the temporal
upsampling head — is the published network, and on
``configs/interfaces/physformer_interface.yaml`` (128x128 frames, 160-frame
windows) the forward pass is numerically the published one.

Any other frame size or window length is accepted through two adaptive
stages around that network, both exact no-ops at the paper's shape:

* the original wrote the token grid as ``view(B, C, P//16, 4, 4)``, true only
  for 128x128 frames. Here the stem's output is average-pooled to the nearest
  multiple of the 4-pixel patch and the grid is read off the result: 128x128
  frames still tokenize to the paper's 4x4 (a pool from 16x16 to 16x16 is the
  identity, bin for bin), 72x72 frames to 2x2, and so on down to 8x8 frames,
  the smallest the stem's three 2x pools leave anything of;
* the 4-frame tubes and the head's two 2x temporal upsamples need a window
  that divides by 4. The stem's output is average-pooled in time to the
  nearest multiple of 4 and the prediction is linearly interpolated back to
  the window length; when the window already divides by 4 neither runs.

No parameter is added, renamed or resized, so the upstream checkpoint loads
strictly. A clip backbone on the multi-signal contract: ``(B, C_in, T, H, W)``
in, ``(B, 1, T)`` out, preprocessing done by the dataset, the loss owned by
the trainer. All reshaping is einops.
"""

import math

import torch
from einops import einsum, rearrange, reduce
from torch import nn
from torch.nn import functional as F

#: The stem's three ``MaxPool3d((1, 2, 2))`` stages, i.e. the spatial factor
#: the tube patch embedding sees on top of its own patch size.
STEM_SPATIAL_STRIDE = 8

#: The smallest frame the stem leaves a 1x1 map of; anything smaller pools
#: to nothing.
MIN_FRAME = STEM_SPATIAL_STRIDE

#: The head's two ``Upsample(scale_factor=(2, 1, 1))`` stages. The temporal
#: patch size has to match it for the output to come back at the input length.
HEAD_TEMPORAL_UPSAMPLE = 4


def _nearest_multiple(n: int, k: int) -> int:
    """The positive multiple of ``k`` nearest to ``n``."""
    return max(round(n / k), 1) * k


class CDC_T(nn.Module):
    """Temporal center-difference 3-D convolution.

    ``theta`` controls the mix of the plain convolution and the central
    difference one; ``theta = 0`` is an ordinary ``Conv3d``.
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
            kernel_diff = (reduce(self.conv.weight[:, :, 0], "co ci kh kw -> co ci", "sum")
                           + reduce(self.conv.weight[:, :, 2], "co ci kh kw -> co ci", "sum"))
            kernel_diff = rearrange(kernel_diff, "cout cin -> cout cin 1 1 1")
            out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias,
                                stride=self.conv.stride, padding=0,
                                dilation=self.conv.dilation, groups=self.conv.groups)
            return out_normal - self.theta * out_diff

        return out_normal


class MultiHeadedSelfAttention_TDC_gra_sharp(nn.Module):
    """Multi-headed dot-product attention whose Q/K projections are 3-D CDCs.

    The token sequence is folded back into its ``(gt, gh, gw)`` tube grid so the
    projections can be depth-wise 3-D convolutions over space *and* time —
    which is what makes the attention temporal-difference aware. The grid is
    a forward-time argument: the projections are convolutions, so nothing in
    the parameters depends on it.
    """

    def __init__(self, dim, num_heads, dropout, theta):
        super().__init__()
        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
        )
        self.proj_k = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
        )

        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None  # for visualization

    def forward(self, x, gra_sharp, grid):
        """``(B, gt*gh*gw, dim)`` in, the same shape out (plus the score map)."""
        gh, gw = grid
        x = rearrange(x, "b (gt gh gw) c -> b c gt gh gw", gh=gh, gw=gw)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (
            rearrange(t, "b (nh dh) gt gh gw -> b nh (gt gh gw) dh", nh=self.n_heads)
            for t in (q, k, v)
        )

        # ``gra_sharp`` replaces the usual sqrt(d_k): a tunable softmax
        # temperature, which is the "gradient sharpness" of the paper's title.
        scores = einsum(q, k, "b nh s dh, b nh u dh -> b nh s u") / gra_sharp
        scores = self.drop(F.softmax(scores, dim=-1))

        h = einsum(scores, v, "b nh s u, b nh u dh -> b nh s dh")
        h = rearrange(h, "b nh s dh -> b s (nh dh)")
        self.scores = scores
        return h, scores


class PositionWiseFeedForward_ST(nn.Module):
    """Feed-forward with a depth-wise spatio-temporal conv between the two 1x1s."""

    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Conv3d(dim, ff_dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        self.STConv = nn.Sequential(
            nn.Conv3d(ff_dim, ff_dim, 3, stride=1, padding=1, groups=ff_dim, bias=False),
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv3d(ff_dim, dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(dim),
        )

    def forward(self, x, grid):
        gh, gw = grid
        x = rearrange(x, "b (gt gh gw) c -> b c gt gh gw", gh=gh, gw=gw)
        x = self.fc1(x)
        x = self.STConv(x)
        x = self.fc2(x)
        return rearrange(x, "b c gt gh gw -> b (gt gh gw) c")


class Block_ST_TDC_gra_sharp(nn.Module):
    """One pre-norm transformer block."""

    def __init__(self, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.attn = MultiHeadedSelfAttention_TDC_gra_sharp(dim, num_heads, dropout, theta)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward_ST(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gra_sharp, grid):
        Atten, Score = self.attn(self.norm1(x), gra_sharp, grid)
        h = self.drop(self.proj(Atten))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x), grid))
        x = x + h
        return x, Score


class Transformer_ST_TDC_gra_sharp(nn.Module):
    """One of the three transformer stages."""

    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block_ST_TDC_gra_sharp(dim, num_heads, ff_dim, dropout, theta)
            for _ in range(num_layers)
        ])

    def forward(self, x, gra_sharp, grid):
        for block in self.blocks:
            x, Score = block(x, gra_sharp, grid)
        return x, Score


class ViT_ST_ST_Compact3_TDC_gra_sharp(nn.Module):
    """stem_3DCNN + ST-ViT with local depth-wise spatio-temporal MLP.

    ``in_channels`` is the only width that differs from the published network;
    at ``3`` this is exactly the original. Any frame size from 8x8 and any
    window length are accepted (see the module docstring for how, and why the
    paper's shape is untouched).
    """

    def __init__(
        self,
        patches: int = 4,
        dim: int = 96,
        ff_dim: int = 144,
        num_heads: int = 4,
        num_layers: int = 12,
        dropout_rate: float = 0.1,
        in_channels: int = 3,
        theta: float = 0.7,
    ):
        super().__init__()

        self.dim = dim
        self.in_channels = in_channels
        self.patch = patches                       # (4, 4, 4) tubes in the paper

        if patches != HEAD_TEMPORAL_UPSAMPLE:
            raise ValueError(
                f"PhysFormer's head upsamples time by {HEAD_TEMPORAL_UPSAMPLE}x "
                f"(two Upsample(scale_factor=(2,1,1)) stages), so the temporal "
                f"patch size must be {HEAD_TEMPORAL_UPSAMPLE} for the prediction "
                f"to come back at the window length; got {patches}."
            )

        # Patch embedding: one [ft x fh x fw] tube -> one token.
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(patches,) * 3,
                                         stride=(patches,) * 3)

        stage_layers = num_layers // 3
        self.transformer1 = Transformer_ST_TDC_gra_sharp(
            num_layers=stage_layers, dim=dim, num_heads=num_heads, ff_dim=ff_dim,
            dropout=dropout_rate, theta=theta)
        self.transformer2 = Transformer_ST_TDC_gra_sharp(
            num_layers=stage_layers, dim=dim, num_heads=num_heads, ff_dim=ff_dim,
            dropout=dropout_rate, theta=theta)
        self.transformer3 = Transformer_ST_TDC_gra_sharp(
            num_layers=stage_layers, dim=dim, num_heads=num_heads, ff_dim=ff_dim,
            dropout=dropout_rate, theta=theta)

        # --- The one deviation: the first layer takes ``in_channels``, not 3. ---
        self.Stem0 = nn.Sequential(
            nn.Conv3d(in_channels, dim // 4, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(dim // 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim // 2, [3, 1, 1], stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim // 2),
            nn.ELU(),
        )

        # The readout: one plane, bias kept (the trainer seeds it with the
        # trace's physiological prior) and deliberately no activation, so an
        # absolute-class signal is expressible in its own units.
        self.ConvBlockLast = nn.Conv1d(dim // 2, 1, 1, stride=1, padding=0)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)

    def forward(self, x, gra_sharp):
        """``(B, C_in, T, H, W)`` -> ``((B, 1, T), Score1, Score2, Score3)``."""
        frames, height, width = x.shape[2:]
        if min(height, width) < MIN_FRAME:
            raise ValueError(
                f"PhysFormer's stem pools frames {STEM_SPATIAL_STRIDE}x, so they "
                f"must be at least {MIN_FRAME}x{MIN_FRAME}; got {height}x{width}.")

        x = self.Stem0(x)
        x = self.Stem1(x)
        x = self.Stem2(x)                                  # [B, dim, T, H/8, W/8]

        # Any size: bring the stem's output to whole tubes. At the paper's
        # 128x128 frames and 160-frame windows the target equals the input
        # and this is skipped, so that path stays the published network.
        target = tuple(_nearest_multiple(n, self.patch) for n in x.shape[2:])
        if tuple(x.shape[2:]) != target:
            x = F.adaptive_avg_pool3d(x, target)

        x = self.patch_embedding(x)                        # [B, dim, T/4, gh, gw]
        grid = tuple(x.shape[3:])
        x = rearrange(x, "b c gt gh gw -> b (gt gh gw) c")

        Trans_features, Score1 = self.transformer1(x, gra_sharp, grid)
        Trans_features2, Score2 = self.transformer2(Trans_features, gra_sharp, grid)
        Trans_features3, Score3 = self.transformer3(Trans_features2, gra_sharp, grid)

        gh, gw = grid
        features_last = rearrange(Trans_features3, "b (gt gh gw) c -> b c gt gh gw",
                                  gh=gh, gw=gw)           # [B, dim, T/4, gh, gw]

        features_last = self.upsample(features_last)       # [B, dim,   T/2, gh, gw]
        features_last = self.upsample2(features_last)      # [B, dim/2, T,   gh, gw]

        # Pool the token grid away; time survives.
        features_last = reduce(features_last, "b c t gh gw -> b c t", "mean")
        rPPG = self.ConvBlockLast(features_last)           # [B, 1, T']

        # Back to the window length if the tubes did not divide it.
        if rPPG.shape[-1] != frames:
            rPPG = F.interpolate(rPPG, size=frames, mode="linear", align_corners=False)

        return rPPG, Score1, Score2, Score3


class PhysFormer(nn.Module):
    """PhysFormer as a clip backbone: ``(B, in_channels, T, H, W)`` -> ``(B, 1, T)``.

    Wraps :class:`ViT_ST_ST_Compact3_TDC_gra_sharp` rather than merging with it,
    so the published network stays a self-contained module that can be checked
    against the original at ``in_channels=3``.

    ``gra_sharp`` is a forward-time argument of the original network, held here
    as the constant the paper and every published trainer use (2.0). It is the
    softmax temperature of the attention, so it belongs to the architecture, not
    to the training loop.
    """

    def __init__(self, in_channels=3, patches=4, dim=96, ff_dim=144, num_heads=4,
                 num_layers=12, theta=0.7, dropout_rate=0.1, gra_sharp=2.0):
        """``in_channels`` is the number of input channels (default 3); every
        other argument is a published hyperparameter at its published value."""
        super().__init__()
        self.in_channels = in_channels
        self.gra_sharp = float(gra_sharp)
        self.backbone = ViT_ST_ST_Compact3_TDC_gra_sharp(
            patches=patches, dim=dim, ff_dim=ff_dim, num_heads=num_heads,
            num_layers=num_layers, dropout_rate=dropout_rate, theta=theta,
            in_channels=in_channels,
        )

    def forward(self, video):
        """``(B, in_channels, T, H, W)`` -> ``(B, 1, T)``; attention maps are dropped."""
        rPPG, *_ = self.backbone(video, self.gra_sharp)
        return rPPG

    def output_layers(self):
        """The activation-free readout: the final 1x1 conv."""
        return (self.backbone.ConvBlockLast,)

    def extra_repr(self):
        return f"gra_sharp={self.gra_sharp}"
