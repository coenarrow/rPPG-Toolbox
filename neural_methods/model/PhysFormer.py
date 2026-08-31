"""PhysFormer: temporal-difference transformer for physiological measurement.

Yu et al., https://arxiv.org/abs/2111.12082 — a combination of ``Physformer.py``
and ``transformer_layer.py`` from the official implementation
(https://github.com/ZitongYu/PhysFormer).

The architecture is unchanged from the original. Per the migration contract's
fidelity principle, exactly three things differ:

* the **first layer** (``Stem0``'s conv) widens to ``in_channels``, so the
  network sees whichever camera channels the config asked for (R,G,B,I,D)
  times however many ``DATA_TYPE`` blocks the frame transform emits;
* the **final readout** (``ConvBlockLast``) widens from 1 to ``out_signals``
  output planes — still a bare ``Conv1d``, no activation, so an absolute-class
  signal like ABP can be predicted directly in mmHg;
* the losses applied to it are the trainer's, not the original's (see the
  migration retro: PhysFormer's published DLDL frequency/KL term is a
  *spectral* component of the per-signal loss registry, not a parallel
  criterion).

Everything between those two layers — the 3-D stem, the (4,4,4) tube
tokenization, the three temporal-difference transformer stages, the temporal
upsampling head — is the published network. At ``in_channels=3,
out_signals=1`` it is exactly the original.

Two hardcoded constants of the original are *derived* here instead, which
changes no behaviour at the published 128x128 input but turns a silent
wrong-shape crash into a named error elsewhere:

* the token grid ``(gh, gw)`` — the original wrote ``view(B, C, P//16, 4, 4)``,
  which is ``gh = gw = 4``, true only at 128x128 with 4x4 patches. It is now
  computed from ``image_size`` and ``patches``, and ``forward`` refuses a frame
  size that would tokenize to a different grid. That refusal is load-bearing
  rather than defensive: whenever the actual token count still divides by the
  declared ``gh*gw``, the reshape *succeeds* and reinterprets several time
  steps' tokens as one step's spatial grid, returning a prediction a quarter
  the window's length instead of raising.
* the temporal token count ``gt = T // ft``, which came from the input all
  along, so the window length is not baked into the model. ``T % ft == 0`` is
  the only constraint; it is declared as
  :attr:`PhysFormer.temporal_divisor` (which is what
  ``MultiSignalTrainer.check_window`` reads, and it reports the fix in
  seconds), and enforced in both the constructor and ``forward`` for the paths
  that never reach the trainer. The legacy trainer instead truncated the batch
  and said nothing.

All reshaping is einops.
"""

import math
from typing import Optional

import torch
from einops import einsum, rearrange, reduce
from torch import nn
from torch.nn import functional as F

from neural_methods.model.DictModel import DictModel

#: The stem's three ``MaxPool3d((1, 2, 2))`` stages, i.e. the spatial factor
#: the tube patch embedding sees on top of its own patch size.
STEM_SPATIAL_STRIDE = 8

#: The head's two ``Upsample(scale_factor=(2, 1, 1))`` stages. The temporal
#: patch size has to match it for the output to come back at the input length.
HEAD_TEMPORAL_UPSAMPLE = 4


def as_tuple(x):
    """A scalar becomes ``(x, x)``; any sequence (yacs hands out lists) passes through."""
    return tuple(x) if isinstance(x, (tuple, list)) else (x, x)


def _check_window_length(frames, patch_t):
    """Refuse a window the tube tokenization cannot divide, naming the fix.

    Called from both the constructor (against the build-time window) and
    ``forward`` (against the window that actually arrives), because those can
    differ — a test split may be configured with its own ``WINDOW_SECONDS``,
    and a notebook may call the module with a bare tensor.
    """
    if frames % patch_t:
        nearest = max(round(frames / patch_t), 1) * patch_t
        raise ValueError(
            f"PhysFormer tokenizes time in tubes of {patch_t} frames, so the "
            f"window length must be a multiple of {patch_t}; got {frames}. "
            f"Nearest valid: {nearest}."
        )


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
    which is what makes the attention temporal-difference aware.
    """

    def __init__(self, dim, num_heads, dropout, theta, grid):
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
        self.gh, self.gw = grid
        self.scores = None  # for visualization

    def forward(self, x, gra_sharp):
        """``(B, gt*gh*gw, dim)`` in, the same shape out (plus the score map)."""
        x = rearrange(x, "b (gt gh gw) c -> b c gt gh gw", gh=self.gh, gw=self.gw)
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

    def __init__(self, dim, ff_dim, grid):
        super().__init__()
        self.gh, self.gw = grid

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

    def forward(self, x):
        x = rearrange(x, "b (gt gh gw) c -> b c gt gh gw", gh=self.gh, gw=self.gw)
        x = self.fc1(x)
        x = self.STConv(x)
        x = self.fc2(x)
        return rearrange(x, "b c gt gh gw -> b (gt gh gw) c")


class Block_ST_TDC_gra_sharp(nn.Module):
    """One pre-norm transformer block."""

    def __init__(self, dim, num_heads, ff_dim, dropout, theta, grid):
        super().__init__()
        self.attn = MultiHeadedSelfAttention_TDC_gra_sharp(dim, num_heads, dropout,
                                                           theta, grid)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward_ST(dim, ff_dim, grid)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gra_sharp):
        Atten, Score = self.attn(self.norm1(x), gra_sharp)
        h = self.drop(self.proj(Atten))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x, Score


class Transformer_ST_TDC_gra_sharp(nn.Module):
    """One of the three transformer stages."""

    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, theta, grid):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block_ST_TDC_gra_sharp(dim, num_heads, ff_dim, dropout, theta, grid)
            for _ in range(num_layers)
        ])

    def forward(self, x, gra_sharp):
        for block in self.blocks:
            x, Score = block(x, gra_sharp)
        return x, Score


class ViT_ST_ST_Compact3_TDC_gra_sharp(nn.Module):
    """stem_3DCNN + ST-ViT with local depth-wise spatio-temporal MLP.

    ``in_channels`` and ``out_signals`` are the only two widths that differ from
    the published network; at ``3`` and ``1`` this is exactly the original.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        pretrained: bool = False,
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.2,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        in_channels: int = 3,
        out_signals: int = 1,
        frame: int = 160,
        theta: float = 0.2,
        image_size: Optional[int] = None,
    ):
        super().__init__()

        self.image_size = image_size
        self.frame = frame
        self.dim = dim
        self.in_channels = in_channels
        self.out_signals = out_signals

        # Tube sizes vs patch sizes. Only the spatial grid is *fixed* at
        # construction; the temporal one follows whatever window arrives at
        # forward time. The T in image_size is therefore not baked in — but it
        # is validated, so a build-time window the architecture cannot
        # tokenize is named here rather than surviving until the first batch.
        size = as_tuple(image_size)
        t, h, w = size if len(size) == 3 else (None, *size)
        patch_sizes = as_tuple(patches)
        ft, fh, fw = patch_sizes if len(patch_sizes) == 3 else (patches,) * 3
        self.patch_t = ft
        self.patch_hw = (fh, fw)
        self.grid = _spatial_grid(h, w, fh, fw)
        if t is not None:
            _check_window_length(t, ft)

        if ft != HEAD_TEMPORAL_UPSAMPLE:
            raise ValueError(
                f"PhysFormer's head upsamples time by {HEAD_TEMPORAL_UPSAMPLE}x "
                f"(two Upsample(scale_factor=(2,1,1)) stages), so the temporal "
                f"patch size must be {HEAD_TEMPORAL_UPSAMPLE} for the prediction "
                f"to come back at the window length; got {ft}."
            )

        # Patch embedding: one [ft x fh x fw] tube -> one token.
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw),
                                         stride=(ft, fh, fw))

        stage_layers = num_layers // 3
        self.transformer1 = Transformer_ST_TDC_gra_sharp(
            num_layers=stage_layers, dim=dim, num_heads=num_heads, ff_dim=ff_dim,
            dropout=dropout_rate, theta=theta, grid=self.grid)
        self.transformer2 = Transformer_ST_TDC_gra_sharp(
            num_layers=stage_layers, dim=dim, num_heads=num_heads, ff_dim=ff_dim,
            dropout=dropout_rate, theta=theta, grid=self.grid)
        self.transformer3 = Transformer_ST_TDC_gra_sharp(
            num_layers=stage_layers, dim=dim, num_heads=num_heads, ff_dim=ff_dim,
            dropout=dropout_rate, theta=theta, grid=self.grid)

        # --- Deviation 1: the first layer takes ``in_channels``, not 3. ---
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

        # --- Deviation 2: the readout emits one plane per predicted signal. ---
        # Style A (contract §2): a 1x1 Conv1d of width S *is* S independent
        # linear readouts of the shared temporal feature. Bias is kept (the
        # builder initialises it to each signal's physiological prior) and there
        # is deliberately no activation, so absolute-class signals are
        # expressible in their own units.
        self.ConvBlockLast = nn.Conv1d(dim // 2, out_signals, 1, stride=1, padding=0)

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
        """``(B, C_in, T, H, W)`` -> ``((B, S, T), Score1, Score2, Score3)``."""
        _, _, t, height, width = x.shape
        _check_window_length(t, self.patch_t)
        # The spatial grid is baked into the attention's 3-D projections, so a
        # frame size other than the one built for must be refused, not
        # tokenized. It would not necessarily fail on its own: whenever the
        # actual token count still divides by the declared gh*gw, the rearrange
        # succeeds and silently reinterprets several time steps' tokens as one
        # step's spatial grid, returning a prediction 1/4 the window's length.
        actual = _spatial_grid(height, width, *self.patch_hw)
        if actual != self.grid:
            raise ValueError(
                f"PhysFormer was built for a {self.grid[0]}x{self.grid[1]} token "
                f"grid, but {height}x{width} frames tokenize to "
                f"{actual[0]}x{actual[1]}. Resize the frames to the size the "
                "model was built for (the frame transform normally does this)."
            )

        x = self.Stem0(x)
        x = self.Stem1(x)
        x = self.Stem2(x)                                  # [B, dim, T, H/8, W/8]

        x = self.patch_embedding(x)                        # [B, dim, T/4, gh, gw]
        x = rearrange(x, "b c gt gh gw -> b (gt gh gw) c")

        Trans_features, Score1 = self.transformer1(x, gra_sharp)
        Trans_features2, Score2 = self.transformer2(Trans_features, gra_sharp)
        Trans_features3, Score3 = self.transformer3(Trans_features2, gra_sharp)

        gh, gw = self.grid
        features_last = rearrange(Trans_features3, "b (gt gh gw) c -> b c gt gh gw",
                                  gh=gh, gw=gw)           # [B, dim, T/4, gh, gw]

        features_last = self.upsample(features_last)       # [B, dim,   T/2, gh, gw]
        features_last = self.upsample2(features_last)      # [B, dim/2, T,   gh, gw]

        # Pool the token grid away; time survives.
        features_last = reduce(features_last, "b c t gh gw -> b c t", "mean")
        rPPG = self.ConvBlockLast(features_last)           # [B, S, T]

        return rPPG, Score1, Score2, Score3


def _spatial_grid(height, width, fh, fw):
    """Token grid ``(gh, gw)`` after the stem's 8x pooling and the patch stride.

    The original hardcoded ``(4, 4)`` — correct only for the published 128x128
    input with 4x4 patches. Deriving it means a different ``RESIZE`` either
    works or is refused by name, instead of failing deep inside a ``view``.
    """
    if height is None or width is None:
        raise ValueError(
            "PhysFormer needs image_size=(T, H, W) (or (H, W)) at construction: "
            "the token grid is fixed by the frame size, and the attention "
            "projections are 3-D convolutions over it."
        )
    grid = []
    for name, size, patch in (("H", height, fh), ("W", width, fw)):
        # One requirement, stated once: the stem pools 8x and the patch
        # embedding strides by `patch`, so the frame must divide by both. The
        # two-stage message this replaced named only the 8x, which sent anyone
        # who hit it at H=20 straight into a second, different error at H=24.
        step = STEM_SPATIAL_STRIDE * patch
        cells, remainder = divmod(size, step)
        if remainder or cells < 1:
            raise ValueError(
                f"PhysFormer needs RESIZE.{name} to be a positive multiple of "
                f"{step} ({STEM_SPATIAL_STRIDE}x stem pooling x {patch} patch); "
                f"got {size}. Nearest valid: {max(round(size / step), 1) * step}."
            )
        grid.append(cells)
    return tuple(grid)


class PhysFormer(DictModel):
    """PhysFormer on the Neckflix batch-dict contract.

    Wraps :class:`ViT_ST_ST_Compact3_TDC_gra_sharp` rather than merging with it,
    so the published network stays a self-contained module that can be checked
    against the original at ``in_channels=3, out_signals=1``.

    ``gra_sharp`` is a forward-time argument of the original network, held here
    as the constant the paper and every published trainer use (2.0). It is the
    softmax temperature of the attention, so it belongs to the architecture, not
    to the training loop.
    """

    #: Window lengths must be a multiple of this (the temporal patch size).
    #: Declared, not silently truncated — ``MultiSignalTrainer.check_window``
    #: reads it and refuses a bad ``WINDOW_SECONDS`` by name (contract §2). The
    #: legacy trainer instead truncated the batch and said nothing.
    temporal_divisor = HEAD_TEMPORAL_UPSAMPLE

    def __init__(self, channels=("R", "G", "B"), traces=("PPG",), frame_transform=None,
                 fs=0.0, image_size=(160, 128, 128), patches=4, dim=96, ff_dim=144,
                 num_heads=4, num_layers=12, theta=0.7, dropout_rate=0.1,
                 gra_sharp=2.0):
        """
        Args:
          channels: ordered camera channels the batch dict supplies.
          traces: ordered signals to predict, one readout plane each.
          frame_transform: raw-pixel preprocessing (resize + ``DATA_TYPE``);
            its channel multiplier is folded into the stem's input width.
          fs: the frame rate the model is trained at, carried in the checkpoint.
          image_size: ``(T, H, W)`` the model is built for. Only ``H``/``W``
            are structural — they fix the token grid, and a differently sized
            frame is refused at forward time. ``T`` is validated but not baked
            in, so one model serves any window that is a multiple of
            :attr:`temporal_divisor`.
        """
        super().__init__(channels=channels, traces=traces,
                         frame_transform=frame_transform, fs=fs)
        self.gra_sharp = float(gra_sharp)
        self.backbone = ViT_ST_ST_Compact3_TDC_gra_sharp(
            image_size=image_size, patches=patches, dim=dim, ff_dim=ff_dim,
            num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_rate,
            theta=theta, in_channels=self.in_channels, out_signals=self.out_signals,
        )

    def forward_video(self, video):
        """``(B, C_in, T, H, W)`` -> ``(B, S, T)``; attention maps are dropped."""
        rPPG, *_ = self.backbone(video, self.gra_sharp)
        return rPPG

    def output_layers(self):
        """The activation-free readout, for bias init and weight-decay exemption."""
        return (self.backbone.ConvBlockLast,)

    def extra_repr(self):
        return (f"{super().extra_repr()}, grid={self.backbone.grid}, "
                f"gra_sharp={self.gra_sharp}")
