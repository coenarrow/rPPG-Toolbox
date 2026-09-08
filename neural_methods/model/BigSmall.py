"""BigSmall: a two-resolution network for physiological measurement.

BigSmall: Efficient Multi-Task Learning For Physiological Measurements
Girish Narayanswamy, Yujia (Nancy) Liu, Yuzhe Yang, Chengqian (Jack) Ma,
Xin Liu, Daniel McDuff, Shwetak Patel

https://arxiv.org/abs/2303.11573

The architecture is the published one: a "big" branch of six convolutions on
a high-resolution frame, run on one frame per segment of ``frame_depth`` and
held for the rest of that segment, summed with a "small" branch of four
convolutions on a tiny frame carrying the published wrapping temporal shift
(WTSM — here the shared ``TSM`` with ``wrap=True``), then one dense
readout. Three things differ from the paper.

First, the first conv of each branch takes ``in_channels`` inputs (the
interface's channel count) instead of 3, as DeepPhys and TS-CAN do. Second,
the multi-task head is gone: upstream emitted action units, respiration and
BVP from the shared features, and here a model predicts one trace with one
complete copy of the network per trace (``MultiTraceModel``), so only the BVP
readout remains — respiration is a trace like any other on the standard
interface.

Third, the small branch's resolution is derived here rather than
preprocessed. Upstream the dataset resized twice, a 144x144 Standardized
frame for the big branch and a 9x9 DiffNormalized frame for the small one; an
interface here states one ``RESIZE``, so the small branch average-pools its
own preprocessing block down to ``small_size`` itself. At the paper's 144x144
that pool is an exact 16x16 mean, the same reduction the upstream resize
made; the difference is that it happens after the DiffNormalized
preprocessing instead of before it.

Any frame size is accepted, from 16x16 (the smallest the big branch's three
pools leave anything of): the big branch's map is resampled to
``small_size`` before the two branches are summed, which at 144x144 is the
identity because the published pools already land on 9x9. Any window length
is accepted too: the temporal shift is in-clip and adaptive, and a window
that is not a multiple of ``frame_depth`` ends in a shorter segment, shifted
and held on its own.

A clip backbone on the multi-signal contract: ``(B, C_in, T, H, W)`` in,
``(B, 1, T)`` out, preprocessing done by the dataset, the loss owned by the
trainer. All reshaping is einops.
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat

from neural_methods.model.shared import TSM, require_min_frame

#: The big branch pools 2x, 2x then 4x; a smaller frame pools to nothing.
MIN_FRAME = 16


class BigSmall(nn.Module):

    def __init__(self,
                 in_channels=3,
                 nb_filters1=32,
                 nb_filters2=64,
                 kernel_size=3,
                 dropout_rate1=0.25,
                 dropout_rate2=0.5,
                 dropout_rate3=0.5,
                 pool_size1=(2, 2),
                 pool_size2=(4, 4),
                 nb_dense=128,
                 frame_depth=3,
                 small_size=9):
        """Definition of BigSmall.

        Args:
          in_channels: the number of input channels of EACH branch (big,
            small). Default: 3
          frame_depth: the segment length the big branch holds one frame for
            and the temporal shift shifts within. Default: 3
          small_size: height/width the small branch's block is pooled to, and
            the size the two branches are summed at. Default: 9, the
            published small resolution.
        Returns:
          BigSmall model.

        At the defaults, and at the paper's 144x144 frames, this is the
        original network's big branch, small branch and BVP head, layer for
        layer.
        """
        super(BigSmall, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.dropout_rate3 = dropout_rate3
        self.pool_size1 = pool_size1
        self.pool_size2 = pool_size2
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        self.frame_depth = frame_depth
        self.small_size = small_size

        # Big Convolutional Layers
        self.big_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv5 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv6 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)

        # Big Avg Pooling / Dropout Layers
        self.big_avg_pooling1 = nn.AvgPool2d(self.pool_size1)
        self.big_dropout1 = nn.Dropout(self.dropout_rate1)
        self.big_avg_pooling2 = nn.AvgPool2d(self.pool_size1)
        self.big_dropout2 = nn.Dropout(self.dropout_rate2)
        self.big_avg_pooling3 = nn.AvgPool2d(self.pool_size2)
        self.big_dropout3 = nn.Dropout(self.dropout_rate3)

        # The branches are summed, so the big branch's map is resampled to
        # the small branch's size. At 144x144 the pools above already land on
        # 9x9 and this is the identity.
        self.big_to_small = nn.AdaptiveAvgPool2d(self.small_size)

        # The small branch's resolution, taken from the interface's frame
        # here instead of from a second preprocessing pass. At 144x144 this
        # is an exact 16x16 mean.
        self.small_pooling = nn.AdaptiveAvgPool2d(self.small_size)

        # TSM layers: the published WTSM, the shared shift wrapping at the
        # ends of each segment.
        self.TSM_1 = TSM(frame_depth=self.frame_depth, wrap=True)
        self.TSM_2 = TSM(frame_depth=self.frame_depth, wrap=True)
        self.TSM_3 = TSM(frame_depth=self.frame_depth, wrap=True)
        self.TSM_4 = TSM(frame_depth=self.frame_depth, wrap=True)

        # Small Convolutional Layers
        self.small_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.small_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.small_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.small_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)

        # BVP Fully Connected Layers
        features = self.nb_filters2 * self.small_size * self.small_size   # 5184 published
        self.bvp_fc1 = nn.Linear(features, self.nb_dense, bias=True)
        self.bvp_fc2 = nn.Linear(self.nb_dense, 1, bias=True)

    def output_layers(self):
        """The activation-free readout."""
        return (self.bvp_fc2,)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """``(B, 2 * in_channels, T, H, W)`` -> ``(B, 1, T)``: the big block
        first on the channel axis, the small block second, each folded to one
        2D frame per row for the published network."""
        b, _, t, height, width = video.shape
        require_min_frame("BigSmall", MIN_FRAME, height, width)
        big_input = video[:, :self.in_channels]
        small_input = video[:, self.in_channels:2 * self.in_channels]

        # The big branch sees the first frame of each segment only, as
        # upstream does; a trailing partial segment contributes its own.
        big_input = rearrange(big_input[:, :, ::self.frame_depth],
                              "b c s h w -> (b s) c h w")

        # Big Conv block 1
        b1 = nn.functional.relu(self.big_conv1(big_input))
        b2 = nn.functional.relu(self.big_conv2(b1))
        b3 = self.big_avg_pooling1(b2)
        b4 = self.big_dropout1(b3)

        # Big Conv block 2
        b5 = nn.functional.relu(self.big_conv3(b4))
        b6 = nn.functional.relu(self.big_conv4(b5))
        b7 = self.big_avg_pooling2(b6)
        b8 = self.big_dropout2(b7)

        # Big Conv block 3
        b9 = nn.functional.relu(self.big_conv5(b8))
        b10 = nn.functional.relu(self.big_conv6(b9))
        b11 = self.big_avg_pooling3(b10)
        b12 = self.big_dropout3(b11)
        b13 = self.big_to_small(b12)

        # Each segment's one big frame is held for the whole segment, so the
        # branches line up frame for frame. The last segment is short when
        # frame_depth does not divide the window.
        b14 = repeat(b13, "(b s) c h w -> b (s r) c h w", b=b, r=self.frame_depth)
        b15 = rearrange(b14[:, :t], "b t c h w -> (b t) c h w")

        # Small branch, pooled to its published resolution and unfolded back
        # around each temporal shift.
        s0 = rearrange(small_input, "b c t h w -> (b t) c h w")
        s0 = rearrange(self.small_pooling(s0), "(b t) c h w -> b t c h w", b=b)

        # Small Conv block 1
        s1 = rearrange(self.TSM_1(s0), "b t c h w -> (b t) c h w")
        s2 = nn.functional.relu(self.small_conv1(s1))
        s2 = rearrange(s2, "(b t) c h w -> b t c h w", b=b)
        s3 = rearrange(self.TSM_2(s2), "b t c h w -> (b t) c h w")
        s4 = nn.functional.relu(self.small_conv2(s3))
        s4 = rearrange(s4, "(b t) c h w -> b t c h w", b=b)

        # Small Conv block 2
        s5 = rearrange(self.TSM_3(s4), "b t c h w -> (b t) c h w")
        s6 = nn.functional.relu(self.small_conv3(s5))
        s6 = rearrange(s6, "(b t) c h w -> b t c h w", b=b)
        s7 = rearrange(self.TSM_4(s6), "b t c h w -> (b t) c h w")
        s8 = nn.functional.relu(self.small_conv4(s7))

        # Shared Layers
        shared = rearrange(b15 + s8, "n c h w -> n (c h w)")

        # BVP Output Layers
        bvpfc1 = nn.functional.relu(self.bvp_fc1(shared))
        out = self.bvp_fc2(bvpfc1)

        return rearrange(out, "(b t) s -> b s t", b=b)
