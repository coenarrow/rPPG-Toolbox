"""FactorizePhys: matrix factorization as multidimensional attention for rPPG.

Joshi, Agaian and Cho, "FactorizePhys: Matrix Factorization for
Multidimensional Attention in Remote Physiological Sensing", NeurIPS 2024.

The architecture is the published ``FSAM_Res`` one: a 3-D convolutional
feature extractor over the temporal difference of the clip, a head whose
voxel embeddings are multiplied by the factorized attention mask of
:mod:`neural_methods.model.FactorizePhys.FSAM` and added back to themselves,
and a single readout plane. At ``in_channels=3`` and the defaults it is that
network layer for layer; the differences from the upstream file are the ones
this contract asks of every model:

- the stem takes ``in_channels`` inputs instead of the 1 / 3 / 4 if-chain,
  through one ``InstanceNorm3d(in_channels)`` (the norm carries no
  parameters, so nothing is added or renamed);
- the in-network ``torch.diff`` keeps a zero frame appended, so a clip of T
  frames leaves T rows. Upstream repeated the last frame in its trainer
  before the forward, which is the same arithmetic (``x_T - x_T = 0``) and
  the same convention as ``neural_methods.frame_transforms.diff_normalized``;
- the ``frames``, ``md_config``, ``device`` and ``debug`` arguments are gone:
  the window is read off the input, the published factorisation values are
  constructor defaults, modules follow their parameters, and the debug
  branches went with the rest of the legacy code. ``MD_INFERENCE: True`` in
  the paper's config, so FSAM ran in eval too and there is nothing left to
  switch on;
- the readout returns one trace, ``(B, 1, T)``, and gains a bias (upstream
  had none) because the trainer seeds each readout's bias with its trace's
  physiological prior. A multi-signal run is one complete copy of this
  network per trace (``MultiTraceModel``), never a widened readout;
- the approximation error of the factorisation is no longer returned: the
  upstream trainer only logged it, never adding it to the loss.

Any window length is accepted as it is: nothing in the network strides or
pools in time. Frames may be any size from ``MIN_FRAME``: the head's valid
convolutions take the feature map from 13x13 down to a point, so the
extractor's output is average-pooled spatially to 13x13 before it. At the
paper's 72x72 frames the extractor already emits 13x13 and the pool is
skipped, so the paper path is the published forward pass.

A clip backbone on the multi-signal contract: ``(B, C_in, T, H, W)`` in,
``(B, 1, T)`` out, preprocessing done by the dataset, the loss owned by the
trainer. All reshaping is einops.
"""

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from neural_methods.model.FactorizePhys.FSAM import FeaturesFactorizationModule

#: The published filter counts of the three stages.
nf = [8, 12, 16]

#: What the head's chain of valid 3x3x3 convolutions consumes: three in the
#: conv block and three in the final layer, two pixels each, ending at 1x1.
HEAD_SPATIAL = 13

#: The extractor is five valid convolutions, two of them strided: a frame
#: runs 72 -> 35 -> 33 -> 31 -> 15 -> 13, and 23 -> 11 -> 9 -> 7 -> 3 -> 1 is
#: the smallest frame it leaves anything of.
MIN_FRAME = 23


class ConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super().__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding=padding, bias=False),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_3d(x)


class rPPG_FeatureExtractor(nn.Module):
    """The 3-D convolutional stem. Shapes below are the paper's 72x72 frames."""

    def __init__(self, inCh, dropout_rate=0.1):
        super().__init__()
        # inCh, out_channel, kernel_size, stride, padding
        #                                                        Input: #B, inCh, 160, 72, 72
        self.FeatureExtractor = nn.Sequential(
            ConvBlock3D(inCh, nf[0], [3, 3, 3], [1, 1, 1], [1, 1, 1]),  #B, nf[0], 160, 72, 72
            ConvBlock3D(nf[0], nf[1], [3, 3, 3], [1, 2, 2], [1, 0, 0]), #B, nf[1], 160, 35, 35
            ConvBlock3D(nf[1], nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf[1], 160, 33, 33
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(nf[1], nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf[1], 160, 31, 31
            ConvBlock3D(nf[1], nf[2], [3, 3, 3], [1, 2, 2], [1, 0, 0]), #B, nf[2], 160, 15, 15
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf[2], 160, 13, 13
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x):
        return self.FeatureExtractor(x)


class BVP_Head(nn.Module):
    """Voxel embeddings in, one trace out, factorized attention in between."""

    def __init__(self, dropout_rate=0.1, use_fsam=True, residual=True):
        super().__init__()
        self.use_fsam = use_fsam
        self.residual = residual

        self.conv_block = nn.Sequential(
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf[2], 160, 11, 11
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf[2], 160, 9, 9
            ConvBlock3D(nf[2], nf[2], [3, 3, 3], [1, 1, 1], [1, 0, 0]), #B, nf[2], 160, 7, 7
            nn.Dropout3d(p=dropout_rate),
        )

        if self.use_fsam:
            self.fsam = FeaturesFactorizationModule(nf[2], align_channels=nf[2] // 2)
            self.fsam_norm = nn.InstanceNorm3d(nf[2])
            self.bias1 = nn.Parameter(torch.tensor(1.0))

        self.final_layer = nn.Sequential(
            ConvBlock3D(nf[2], nf[1], [3, 3, 3], [1, 1, 1], [1, 0, 0]),                        #B, nf[1], 160, 5, 5
            ConvBlock3D(nf[1], nf[0], [3, 3, 3], [1, 1, 1], [1, 0, 0]),                        #B, nf[0], 160, 3, 3
            nn.Conv3d(nf[0], 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0), bias=True),    #B, 1, 160, 1, 1
        )

    def output_layers(self):
        """The activation-free readout: the final layer's 1-plane conv."""
        return (self.final_layer[-1],)

    def forward(self, voxel_embeddings):
        """``(B, nf[2], T, H', W')`` -> ``(B, 1, T)``."""
        # Any frame size: the convolutions below are all valid and take
        # HEAD_SPATIAL down to a point, so whatever the extractor hands over
        # is pooled to that. At the paper's 72x72 frames it is already 13x13
        # and this is skipped.
        frames = voxel_embeddings.shape[2]
        if tuple(voxel_embeddings.shape[3:]) != (HEAD_SPATIAL, HEAD_SPATIAL):
            voxel_embeddings = F.adaptive_avg_pool3d(
                voxel_embeddings, (frames, HEAD_SPATIAL, HEAD_SPATIAL))

        voxel_embeddings = self.conv_block(voxel_embeddings)

        if self.use_fsam:
            # NMF factorises a non-negative matrix, so the embeddings are
            # shifted before the mask is taken, and again before it is applied.
            att_mask = self.fsam(voxel_embeddings - voxel_embeddings.min())
            x = torch.mul(voxel_embeddings - voxel_embeddings.min() + self.bias1,
                          att_mask - att_mask.min() + self.bias1)
            factorized_embeddings = self.fsam_norm(x)
            if self.residual:
                factorized_embeddings = voxel_embeddings + factorized_embeddings
            x = self.final_layer(factorized_embeddings)
        else:
            x = self.final_layer(voxel_embeddings)

        return rearrange(x, "b 1 t 1 1 -> b 1 t")


class FactorizePhys(nn.Module):
    def __init__(self, in_channels: int = 3, dropout: float = 0.1, use_fsam: bool = True):
        """Definition of FactorizePhys.

        Args:
          in_channels: the number of input channels. Default: 3.
          dropout: the published DROP_RATE. Default: 0.1.
          use_fsam: run the factorized attention module, the ablation the
            paper reports. Default: True, the FSAM_Res path.
        """
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.InstanceNorm3d(in_channels)
        self.rppg_feature_extractor = rPPG_FeatureExtractor(in_channels, dropout_rate=dropout)
        self.rppg_head = BVP_Head(dropout_rate=dropout, use_fsam=use_fsam)

    def output_layers(self):
        """The activation-free readout of this one copy."""
        return self.rppg_head.output_layers()

    def forward(self, x):
        """``(B, in_channels, T, H, W)`` -> ``(B, 1, T)``."""
        height, width = x.shape[3:]
        if min(height, width) < MIN_FRAME:
            raise ValueError(
                f"FactorizePhys's extractor strides and crops frames to nothing "
                f"below {MIN_FRAME}x{MIN_FRAME}; got {height}x{width}.")

        # The temporal difference the network is built on, one frame short of
        # the window; the zero frame puts it back, as upstream's repeated last
        # frame did.
        x = torch.diff(x, dim=2)
        x = torch.cat([x, torch.zeros_like(x[:, :, :1])], dim=2)
        x = self.norm(x)

        return self.rppg_head(self.rppg_feature_extractor(x))
