"""DeepPhys - 2D Convolutional Attention Network.
DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks
ECCV, 2018
Weixuan Chen, Daniel McDuff
"""

import torch
import torch.nn as nn
from einops import rearrange

#: Sanctioned multi-signal head styles (migration contract §2). ``widened`` is
#: style A: the original final layer's output width goes from 1 to S, which is
#: S independent linear readouts of the shared feature and nothing else changed.
#: ``per_signal`` is style B: one copy of the original dense head per signal
#: over the same trunk. Because the head reads the flattened spatial map, a
#: per-signal head is a per-signal learned spatial weighting — which is what
#: justifies it where signals come from different regions (ABP from the
#: carotid, CVP from the jugular). Start with ``widened``; escalate only on
#: per-signal metric evidence of interference.
HEAD_STYLES = ('widened', 'per_signal')


class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config


class DeepPhys(nn.Module):

    def __init__(self, in_channels=3, out_signals=1, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, img_size=36,
                 head_style='widened'):
        """Definition of DeepPhys.
        Args:
          in_channels: the number of input channel. Default: 3
          out_signals: the number of output signals. Default: 1
          img_size: height/width of each frame. Default: 36.
          head_style: 'widened' (default) or 'per_signal'; see HEAD_STYLES.
        Returns:
          DeepPhys model.

        Only two things differ from the published network: the first conv of
        each branch takes ``in_channels`` inputs, and the final dense layer
        emits ``out_signals`` outputs. At ``3``/``1`` with the default head
        style this is exactly the original, layer for layer and name for name.
        """
        super(DeepPhys, self).__init__()
        if head_style not in HEAD_STYLES:
            raise ValueError(f"Unknown head_style {head_style!r}; known: {HEAD_STYLES}")
        self.head_style = head_style
        self.in_channels = in_channels
        self.out_signals = out_signals
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Apperance branch convs
        self.apperance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
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
        if head_style == 'widened':
            # The original two layers, under their original names; only the
            # width of the last one moved.
            self.final_dense_1 = nn.Linear(features, self.nb_dense, bias=True)
            self.final_dense_2 = nn.Linear(self.nb_dense, out_signals, bias=True)
        else:
            self.head_dense_1 = nn.ModuleList(
                nn.Linear(features, self.nb_dense, bias=True) for _ in range(out_signals))
            self.head_dense_2 = nn.ModuleList(
                nn.Linear(self.nb_dense, 1, bias=True) for _ in range(out_signals))

    def output_layers(self):
        """The activation-free readout(s): one widened layer, or one per signal."""
        if self.head_style == 'widened':
            return (self.final_dense_2,)
        return tuple(self.head_dense_2)

    def _read_out(self, features):
        """``(B, features)`` -> ``(B, S)``, through whichever head style is built."""
        if self.head_style == 'widened':
            hidden = self.dropout_4(torch.tanh(self.final_dense_1(features)))
            return self.final_dense_2(hidden)
        per_signal = [
            dense_2(self.dropout_4(torch.tanh(dense_1(features))))
            for dense_1, dense_2 in zip(self.head_dense_1, self.head_dense_2)
        ]
        return torch.cat(per_signal, dim=1)

    def forward(self, inputs, params=None):

        diff_input = inputs[:, :self.in_channels, :, :]
        raw_input = inputs[:, self.in_channels:2 * self.in_channels, :, :]

        d1 = torch.tanh(self.motion_conv1(diff_input))
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

        d5 = torch.tanh(self.motion_conv3(d4))
        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))

        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = rearrange(d8, "b c h w -> b (c h w)")
        out = self._read_out(d9)

        return out

