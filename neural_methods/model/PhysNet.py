"""PhysNet: an end-to-end spatio-temporal encoder-decoder for rPPG.

Yu et al., "Remote Photoplethysmograph Signal Measurement from Facial Videos
Using Spatio-Temporal Networks", BMVC 2019.

The architecture is unchanged from the original: the same 3D-conv stem, the
same encoder-decoder trunk that halves time twice and doubles it back twice
via transposed convolutions, the same single-plane readout. One thing differs
from the published network: the stem takes ``in_channels`` inputs (the
interface's channel count) instead of 3. The readout stays a single output
plane; a multi-signal run is one complete copy of this network per trace
(``MultiTraceModel``), never a widened readout on a shared trunk. At ``3``
this is exactly the original, layer for layer.

Any window length is accepted: the trunk halves time twice before the
transposed convolutions double it back twice, so the stem's output is
average-pooled in time to the nearest multiple of 4 and the prediction is
linearly interpolated back to the window length after it. Both are skipped
when the window already divides by 4, so on
``configs/interfaces/physnet_interface.yaml`` (128-frame windows) the forward
pass is the published one. Frames may be any size from 16x16, the smallest
the four 2x spatial pools leave anything of.

A clip backbone on the multi-signal contract: ``(B, C_in, T, H, W)`` in,
``(B, 1, T)`` out, preprocessing done by the dataset, the loss owned by the
trainer. All reshaping is einops.
"""

import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from neural_methods.model.shared import nearest_multiple, require_min_frame

#: Two ``MaxPool3d`` stages spatial-only plus two spatial-and-temporal
#: stages between the input and the bottleneck; a smaller frame pools to
#: nothing.
MIN_FRAME = 16

#: The trunk's temporal stride: two 2x halvings before the bottleneck, undone
#: by two 2x transposed-conv upsamples after it.
TEMPORAL_STRIDE = 4


class PhysNet(nn.Module):
    def __init__(self, in_channels=3):
        """Definition of PhysNet.

        Args:
          in_channels: the number of input channels. Default: 3.
        """
        super().__init__()
        self.in_channels = in_channels

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64,
                                kernel_size=[4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64,
                                kernel_size=[4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        # Spatial-only pooling to a point, time kept at whatever length the
        # trunk hands back: ``None`` is the identity in time, so one model
        # serves any window.
        self.poolspa = nn.AdaptiveAvgPool3d((None, 1, 1))

    def output_layers(self):
        """The activation-free readout: the final 1x1x1 conv."""
        return (self.ConvBlock10,)

    def forward(self, x):
        """``(B, in_channels, T, H, W)`` -> ``(B, 1, T)``."""
        frames, height, width = x.shape[2:]
        require_min_frame("PhysNet", MIN_FRAME, height, width)

        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)

        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)

        # Any window length: the trunk halves time twice then doubles it back
        # twice, so T must divide by 4. At the paper's 128-frame windows the
        # target equals T and this is skipped.
        t4 = nearest_multiple(frames, TEMPORAL_STRIDE)
        if t4 != frames:
            x = F.adaptive_avg_pool3d(x, (t4, x.shape[3], x.shape[4]))

        x = self.MaxpoolSpaTem(x)

        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.MaxpoolSpaTem(x)

        x = self.ConvBlock6(x)
        x = self.ConvBlock7(x)
        x = self.MaxpoolSpa(x)

        x = self.ConvBlock8(x)
        x = self.ConvBlock9(x)
        x = self.upsample(x)
        x = self.upsample2(x)

        x = self.poolspa(x)
        x = self.ConvBlock10(x)
        out = rearrange(x, "b 1 t 1 1 -> b 1 t")

        # Back to the window length if the stride did not divide it.
        if out.shape[-1] != frames:
            out = F.interpolate(out, size=frames, mode="linear", align_corners=False)
        return out
