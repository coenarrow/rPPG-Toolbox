"""EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Vitals Measurement
Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2023)
Xin Liu, Brial Hill, Ziheng Jiang, Shwetak Patel, Daniel McDuff
"""

import torch
import torch.nn as nn
from einops import rearrange

from neural_methods.model.TS_CAN import TSM, Attention_mask


class EfficientPhys(nn.Module):

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
        """Definition of EfficientPhys.
        Args:
          in_channels: the number of input channels of the single branch (the
            interface's channel count). Default: 3
          frame_depth: the segment length the temporal shift shifts within.
            Default: 20
          img_size: height/width of each frame. Default: 36.
        Returns:
          EfficientPhys model.

        Two things differ from the published network. First, the first conv
        and the input batch norm take ``in_channels`` inputs (the interface's
        channel count) instead of 3, as DeepPhys does. Second, the temporal
        shift (``TSM``, the one TS-CAN uses) is adaptive to any clip length
        ``T``; at a ``T`` that is a multiple of ``frame_depth`` this computes
        exactly the published shift. At the defaults this is the original
        network, layer for layer.
        """
        super(EfficientPhys, self).__init__()
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
        # Attention layers: EfficientPhys gates the motion branch with itself,
        # so there is no appearance branch to pair it with.
        self.apperance_att_conv1 = nn.Conv2d(
            self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(
            self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()
        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)
        # Dense layers
        h1 = (img_size - 2) // 2          # conv2 (valid) then pool /2
        h2 = (h1 - 2) // 2                # conv4 (valid) then pool /2
        features = self.nb_filters2 * h2 * h2
        self.final_dense_1 = nn.Linear(features, self.nb_dense, bias=True)
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)
        # The frame difference is the network's own first stage, not dataset
        # preprocessing: EfficientPhys is end-to-end from standardised frames.
        self.batch_norm = nn.BatchNorm2d(self.in_channels)

    def output_layers(self):
        """The activation-free readout."""
        return (self.final_dense_2,)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """``(B, in_channels, T, H, W)`` -> ``(B, 1, T)``: difference each
        clip along its own time axis, then fold to one 2D frame per row for
        the published network and unfold back around each temporal shift.

        ``torch.diff`` leaves ``T - 1`` rows, so a zero frame is appended to
        each clip. That is numerically what upstream produced by appending a
        copy of the chunk's last frame before differencing (the last
        difference is then the frame minus itself): one output per input
        frame, and no difference taken across a clip boundary.
        """
        b = video.shape[0]
        frames = rearrange(video, "b c t h w -> b t c h w")
        diff_input = torch.diff(frames, dim=1)
        diff_input = torch.cat((diff_input, torch.zeros_like(frames[:, :1])), dim=1)
        diff_input = rearrange(diff_input, "b t c h w -> (b t) c h w")
        diff_input = self.batch_norm(diff_input)

        diff_input = rearrange(diff_input, "(b t) c h w -> b t c h w", b=b)
        diff_input = self.TSM_1(diff_input)
        diff_input = rearrange(diff_input, "b t c h w -> (b t) c h w")
        d1 = torch.tanh(self.motion_conv1(diff_input))
        d1 = rearrange(d1, "(b t) c h w -> b t c h w", b=b)
        d1 = self.TSM_2(d1)
        d1 = rearrange(d1, "b t c h w -> (b t) c h w")
        d2 = torch.tanh(self.motion_conv2(d1))

        g1 = torch.sigmoid(self.apperance_att_conv1(d2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        d4 = rearrange(d4, "(b t) c h w -> b t c h w", b=b)
        d4 = self.TSM_3(d4)
        d4 = rearrange(d4, "b t c h w -> (b t) c h w")
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = rearrange(d5, "(b t) c h w -> b t c h w", b=b)
        d5 = self.TSM_4(d5)
        d5 = rearrange(d5, "b t c h w -> (b t) c h w")
        d6 = torch.tanh(self.motion_conv4(d5))

        g2 = torch.sigmoid(self.apperance_att_conv2(d6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = rearrange(d8, "n c h w -> n (c h w)")
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        out = self.final_dense_2(d11)

        return rearrange(out, "(b t) s -> b s t", b=b)
