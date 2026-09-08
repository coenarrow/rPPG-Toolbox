"""Temporal Shift Convolutional Attention Network (TS-CAN).
Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement
NeurIPS, 2020
Xin Liu, Josh Fromm, Shwetak Patel, Daniel McDuff
"""

import torch
import torch.nn as nn
from einops import rearrange


class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5


class TSM(nn.Module):
    """Temporal shift over segments of ``frame_depth`` consecutive frames of
    the same clip.

    Operates on ``(B, T, C, H, W)``, one clip per row of ``B``. Each clip's
    ``T`` frames are cut into chunks of ``frame_depth``, and every chunk is
    shifted independently, so the shift never crosses a clip boundary. A
    trailing partial chunk (``T`` not a multiple of ``frame_depth``) is
    shifted as its own shorter segment — the published shift already
    zero-pads at segment ends, so a short segment is well defined. At
    ``T % frame_depth == 0`` this computes exactly the published,
    single-chunk-size shift.
    """

    def __init__(self, frame_depth: int = 20, fold_div: int = 3):
        super().__init__()
        self.frame_depth = frame_depth
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t, c, _, _ = x.shape
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        for start in range(0, t, self.frame_depth):
            end = min(start + self.frame_depth, t)
            segment = x[:, start:end]
            shifted = torch.zeros_like(segment)
            shifted[:, :-1, :fold] = segment[:, 1:, :fold]                  # shift left
            shifted[:, 1:, fold:2 * fold] = segment[:, :-1, fold:2 * fold]  # shift right
            shifted[:, :, 2 * fold:] = segment[:, :, 2 * fold:]             # not shifted
            out[:, start:end] = shifted
        return out


class TSCAN(nn.Module):

    def __init__(self,
                 in_channels=3,
                 nb_filters1=32,
                 nb_filters2=64,
                 kernel_size=3,
                 dropout_rate1=0.25,
                 dropout_rate2=0.5,
                 pool_size=(2, 2),
                 nb_dense=128,
                 frame_depth=20,
                 img_size=36):
        """Definition of TS-CAN.
        Args:
          in_channels: the number of input channels of EACH branch (motion,
            appearance). Default: 3
          frame_depth: the segment length the temporal shift shifts within.
            Default: 20
          img_size: height/width of each frame. Default: 36.
        Returns:
          TSCAN model.

        Two things differ from the published network. First, the first conv
        of each branch takes ``in_channels`` inputs (the interface's channel
        count) instead of 3, as DeepPhys does. Second, the temporal shift
        (``TSM``) is adaptive to any clip length ``T``; at a ``T`` that is a
        multiple of ``frame_depth`` this computes exactly the published
        shift. At the defaults this is the original network, layer for
        layer.
        """
        super(TSCAN, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        # TSM layers
        self.TSM_1 = TSM(frame_depth=frame_depth)
        self.TSM_2 = TSM(frame_depth=frame_depth)
        self.TSM_3 = TSM(frame_depth=frame_depth)
        self.TSM_4 = TSM(frame_depth=frame_depth)
        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Apperance branch convs
        self.apperance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(
            self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(
            self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()
        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)
        # Dense layers
        h1 = (img_size - 2) // 2          # conv2 (valid) then pool /2
        h2 = (h1 - 2) // 2                # conv4 (valid) then pool /2
        features = self.nb_filters2 * h2 * h2
        self.final_dense_1 = nn.Linear(features, self.nb_dense, bias=True)
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)

    def output_layers(self):
        """The activation-free readout."""
        return (self.final_dense_2,)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """``(B, 2 * in_channels, T, H, W)`` -> ``(B, 1, T)``: motion block
        first, appearance second, folded to one 2D frame per row for the
        published network and unfolded back around each temporal shift."""
        b = video.shape[0]
        diff_input = video[:, :self.in_channels]
        raw_input = video[:, self.in_channels:2 * self.in_channels]

        diff_input = rearrange(diff_input, "b c t h w -> b t c h w")
        raw_input = rearrange(raw_input, "b c t h w -> (b t) c h w")

        diff_input = self.TSM_1(diff_input)
        diff_input = rearrange(diff_input, "b t c h w -> (b t) c h w")
        d1 = torch.tanh(self.motion_conv1(diff_input))
        d1 = rearrange(d1, "(b t) c h w -> b t c h w", b=b)
        d1 = self.TSM_2(d1)
        d1 = rearrange(d1, "b t c h w -> (b t) c h w")
        d2 = torch.tanh(self.motion_conv2(d1))

        r1 = torch.tanh(self.apperance_conv1(raw_input))
        r2 = torch.tanh(self.apperance_conv2(r1))

        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        d4 = rearrange(d4, "(b t) c h w -> b t c h w", b=b)
        d4 = self.TSM_3(d4)
        d4 = rearrange(d4, "b t c h w -> (b t) c h w")
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = rearrange(d5, "(b t) c h w -> b t c h w", b=b)
        d5 = self.TSM_4(d5)
        d5 = rearrange(d5, "b t c h w -> (b t) c h w")
        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))

        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = rearrange(d8, "n c h w -> n (c h w)")
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        out = self.final_dense_2(d11)

        return rearrange(out, "(b t) s -> b s t", b=b)
