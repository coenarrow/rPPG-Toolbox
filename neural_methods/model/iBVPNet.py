"""iBVPNet: a 3D convolutional encoder-decoder for rPPG, proposed with the
iBVP dataset.

Joshi, Jitesh, and Youngjun Cho. 2024. "iBVP Dataset: RGB-Thermal rPPG
Dataset with High Resolution Signal Quality Labels." Electronics 13, no. 7:
1334. https://doi.org/10.3390/electronics13071334

Two things differ from the published network. First, the stem takes
``in_channels`` inputs (the interface's channel count) instead of the
upstream 1/3/4-channel branching; instance norm is per (sample, channel), so
one ``InstanceNorm3d(in_channels)`` over every channel is numerically
identical to upstream's split RGB / thermal norms. Second, upstream fed the
model ``T + 1`` frames (the trainer duplicated the last frame) so that an
internal ``torch.diff`` landed back on the window length ``T``; here the
model takes exactly ``T`` frames, takes the diff itself (``T - 1`` rows), and
appends one zero frame so the trunk always sees ``T`` rows, matching the
window length without help from the caller.

Any window length is accepted: ``temporal_encoder`` halves time twice, so the
spatio-temporal encoder's output is average-pooled in time to the nearest
multiple of 4 before it (a no-op when it already divides by 4), and the
final adaptive max-pool already targets the window length directly, so no
interpolation is needed afterward. At ``T = 160`` (the upstream
``CHUNK_LENGTH``, 30 fps) this is the published network, bit for bit. Frames
may be any size from 64x64, the smallest the spatial pools and strided
decoder convs leave anything of.

A clip backbone on the multi-signal contract: ``(B, C_in, T, H, W)`` in,
``(B, 1, T)`` out, preprocessing done by the dataset, the loss owned by the
trainer. All reshaping is einops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from neural_methods.model.shared import nearest_multiple, require_min_frame

#: The two spatial-only ``MaxPool3d`` stages in ``spatio_temporal_encoder``,
#: the stride-1 spatial ``MaxPool3d`` in ``temporal_encoder``, and the two
#: stride-2 spatial convs in ``decoder_block``; a smaller frame pools to
#: nothing (empirically, since the strides do not divide evenly).
MIN_FRAME = 64

#: ``temporal_encoder``'s two 2x temporal halvings: the spatio-temporal
#: encoder's output is pooled to the nearest multiple of this before them.
TEMPORAL_STRIDE = 4

#: num_filters
nf = [8, 16, 24, 40, 64]


class ConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super().__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_3d(x)


class DeConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super().__init__()
        k_t, k_s1, k_s2 = kernel_size
        s_t, s_s1, s_s2 = stride
        self.deconv_block_3d = nn.Sequential(
            nn.ConvTranspose3d(in_channel, in_channel, (k_t, 1, 1), (s_t, 1, 1), padding),
            nn.Tanh(),
            nn.InstanceNorm3d(in_channel),

            nn.Conv3d(in_channel, out_channel, (1, k_s1, k_s2), (1, s_s1, s_s2), padding),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.deconv_block_3d(x)


class encoder_block(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        # in_channel, out_channel, kernel_size, stride, padding
        self.spatio_temporal_encoder = nn.Sequential(
            ConvBlock3D(in_channel, nf[0], [1, 3, 3], [1, 1, 1], [0, 1, 1]),
            ConvBlock3D(nf[0], nf[1], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            ConvBlock3D(nf[1], nf[2], [1, 3, 3], [1, 1, 1], [0, 1, 1]),
            ConvBlock3D(nf[2], nf[3], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            ConvBlock3D(nf[3], nf[4], [1, 3, 3], [1, 1, 1], [0, 1, 1]),
            ConvBlock3D(nf[4], nf[4], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
        )

        self.temporal_encoder = nn.Sequential(
            ConvBlock3D(nf[4], nf[4], [11, 1, 1], [1, 1, 1], [5, 0, 0]),
            ConvBlock3D(nf[4], nf[4], [11, 3, 3], [1, 1, 1], [5, 1, 1]),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
            ConvBlock3D(nf[4], nf[4], [11, 1, 1], [1, 1, 1], [5, 0, 0]),
            ConvBlock3D(nf[4], nf[4], [11, 3, 3], [1, 1, 1], [5, 1, 1]),
            nn.MaxPool3d((2, 2, 2), stride=(2, 1, 1)),
            ConvBlock3D(nf[4], nf[4], [7, 1, 1], [1, 1, 1], [3, 0, 0]),
            ConvBlock3D(nf[4], nf[4], [7, 3, 3], [1, 1, 1], [3, 1, 1]),
        )

    def forward(self, x):
        st_x = self.spatio_temporal_encoder(x)

        # Any window length: temporal_encoder halves time twice, so T must
        # divide by 4. At the paper's 160-frame windows the target equals T
        # and this is skipped.
        frames = st_x.shape[2]
        t4 = nearest_multiple(frames, TEMPORAL_STRIDE)
        if t4 != frames:
            st_x = F.adaptive_avg_pool3d(st_x, (t4, st_x.shape[3], st_x.shape[4]))

        return self.temporal_encoder(st_x)


class decoder_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_block = nn.Sequential(
            DeConvBlock3D(nf[4], nf[3], [7, 3, 3], [2, 2, 2], [2, 1, 1]),
            DeConvBlock3D(nf[3], nf[2], [7, 3, 3], [2, 2, 2], [2, 1, 1]),
        )

    def forward(self, x):
        return self.decoder_block(x)


class iBVPNet(nn.Module):
    def __init__(self, in_channels=3):
        """Definition of iBVPNet.

        Args:
          in_channels: the number of input channels. Default: 3.
        """
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.InstanceNorm3d(in_channels)
        self.encoder = encoder_block(in_channels)
        self.decoder = decoder_block()
        self.readout = nn.Conv3d(nf[2], 1, [1, 1, 1], stride=1, padding=0)

    def output_layers(self):
        """The activation-free readout: the final 1x1x1 conv."""
        return (self.readout,)

    def forward(self, x):
        """``(B, in_channels, T, H, W)`` -> ``(B, 1, T)``."""
        frames, height, width = x.shape[2:]
        require_min_frame("iBVPNet", MIN_FRAME, height, width)

        # Diff along time, T -> T - 1, then append a zero frame so the trunk
        # sees T rows, same as the window length in and out.
        x = torch.diff(x, dim=2)
        zero_frame = torch.zeros_like(x[:, :, :1, :, :])
        x = torch.cat([x, zero_frame], dim=2)

        x = self.norm(x)
        x = self.encoder(x)
        x = self.decoder(x)

        # Spatial adaptive pooling to a point, time straight to the window
        # length: the frame the diff and zero frame produced.
        x = F.adaptive_max_pool3d(x, (frames, 1, 1))
        x = self.readout(x)
        return rearrange(x, "b 1 t 1 1 -> b 1 t")
