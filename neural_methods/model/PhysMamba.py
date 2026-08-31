"""PhysMamba: slow-fast temporal-difference Mamba for physiological measurement.

Luo et al., https://doi.org/10.48550/arXiv.2409.12031

The architecture is unchanged from the original: the same convolutional stem,
the same two-stream slow/fast temporal-difference Mamba blocks with lateral
fusion, the same upsampling head. Two things changed for the Neckflix pipeline:

* it is a :class:`~neural_methods.model.DictModel.DictModel`, so it consumes
  the loader's batch dict and returns it with a ``predictions`` entry keyed by
  signal name (a plain tensor in still gets a plain tensor out, which is how
  the upstream tuple-contract trainers keep working);
* the input channel count and the number of predicted signals are
  constructor-configurable, instead of hardcoded 3-in / 1-out.

All reshaping is einops.
"""

import math

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import trunc_normal_, DropPath
from torch.nn import functional as F

from neural_methods.model.DictModel import DictModel
from neural_methods.model.mamba_compat import make_mamba


class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        return x*attention

class LateralConnection(nn.Module):
    def __init__(self, fast_channels=32, slow_channels=64):
        super(LateralConnection, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(fast_channels, slow_channels, [3, 1, 1], stride=[2, 1, 1], padding=[1,0,0]),
            nn.BatchNorm3d(slow_channels),
            nn.ReLU(),
        )

    def forward(self, slow_path, fast_path):
        fast_path = self.conv(fast_path)
        return fast_path + slow_path

class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.2):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = rearrange(kernel_diff, "cout cin -> cout cin 1 1 1")
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, channel_token = False):
        super(MambaLayer, self).__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        drop_path = 0
        self.mamba = make_mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba=True,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_patch_token(self, x):
        """Every spatio-temporal position is one token of the SSM sequence."""
        assert x.shape[1] == self.dim, f"expected {self.dim} channels, got {x.shape[1]}"
        frames, height, width = x.shape[2:]
        x_flat = rearrange(x, "b d t h w -> b (t h w) d")
        x_norm = self.norm1(x_flat)
        x_mamba = self.mamba(x_norm)
        x_out = self.norm2(x_flat + self.drop_path(x_mamba))
        return rearrange(x_out, "b (t h w) d -> b d t h w", t=frames, h=height, w=width)

    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
        out = self.forward_patch_token(x)
        return out

def conv_block(in_channels, out_channels, kernel_size, stride, padding, bn=True, activation='relu'):
    layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]
    if bn:
        layers.append(nn.BatchNorm3d(out_channels))
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'elu':
        layers.append(nn.ELU(inplace=True))
    return nn.Sequential(*layers)


class PhysMamba(DictModel):
    #: The slow stream strides T by 4 and the two upsamples put it back; a
    #: window that is not a multiple of 4 comes out a different length than it
    #: went in.
    temporal_divisor = 4

    def __init__(self, channels=("R", "G", "B"), traces=("PPG",), frame_transform=None,
                 fs=0.0, theta=0.5, drop_rate1=0.25, drop_rate2=0.5):
        """Definition of PhysMamba.

        Args:
          channels: ordered camera channels the batch dict supplies.
          traces: ordered signals to predict, one output plane each.
          frame_transform: raw-pixel preprocessing (resize + ``DATA_TYPE``);
            its channel multiplier is folded into the stem's input width.
          fs: frame rate the model is trained at, recorded on the checkpoint.
        """
        super().__init__(channels=channels, traces=traces,
                         frame_transform=frame_transform, fs=fs)

        self.ConvBlock1 = conv_block(self.in_channels, 16, [1, 5, 5], stride=1, padding=[0, 2, 2])
        self.ConvBlock2 = conv_block(16, 32, [3, 3, 3], stride=1, padding=1)
        self.ConvBlock3 = conv_block(32, 64, [3, 3, 3], stride=1, padding=1)
        self.ConvBlock4 = conv_block(64, 64, [4, 1, 1], stride=[4, 1, 1], padding=0)
        self.ConvBlock5 = conv_block(64, 32, [2, 1, 1], stride=[2, 1, 1], padding=0)
        self.ConvBlock6 = conv_block(32, 32, [3, 1, 1], stride=1, padding=[1, 0, 0], activation='elu')

        # Temporal Difference Mamba Blocks
        # Slow Stream
        self.Block1 = self._build_block(64, theta)
        self.Block2 = self._build_block(64, theta)
        self.Block3 = self._build_block(64, theta)
        # Fast Stream
        self.Block4 = self._build_block(32, theta)
        self.Block5 = self._build_block(32, theta)
        self.Block6 = self._build_block(32, theta)

        # Upsampling
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(1,0,0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(96, 48, [3, 1, 1], stride=1, padding=(1,0,0)),
            nn.BatchNorm3d(48),
            nn.ELU(),
        )

        self.ConvBlockLast = nn.Conv3d(48, self.out_signals, [1, 1, 1], stride=1, padding=0)
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        self.fuse_1 = LateralConnection(fast_channels=32, slow_channels=64)
        self.fuse_2 = LateralConnection(fast_channels=32, slow_channels=64)

        self.drop_1 = nn.Dropout(drop_rate1)
        self.drop_2 = nn.Dropout(drop_rate1)
        self.drop_3 = nn.Dropout(drop_rate2)
        self.drop_4 = nn.Dropout(drop_rate2)
        self.drop_5 = nn.Dropout(drop_rate2)
        self.drop_6 = nn.Dropout(drop_rate2)

        # Spatial-only pooling: ``None`` keeps the temporal axis at whatever
        # length the window happens to be, so one model serves any CHUNK_LENGTH.
        self.poolspa = nn.AdaptiveAvgPool3d((None, 1, 1))

    def output_layers(self):
        """The final 1x1x1 conv is the readout: one output plane per trace."""
        return (self.ConvBlockLast,)

    def _build_block(self, channels, theta):
        return nn.Sequential(
            CDC_T(channels, channels, theta=theta),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            MambaLayer(dim=channels),
            ChannelAttention3D(in_channels=channels, reduction=2),
        )

    def forward_video(self, x):
        """``(B, C, T, H, W)`` -> ``(B, S, T)``."""
        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpa(x)

        # Process streams
        s_x = self.ConvBlock4(x) # Slow stream
        f_x = self.ConvBlock5(x) # Fast stream

        # First set of blocks and fusion
        s_x1 = self.Block1(s_x)
        s_x1 = self.MaxpoolSpa(s_x1)
        s_x1 = self.drop_1(s_x1)

        f_x1 = self.Block4(f_x)
        f_x1 = self.MaxpoolSpa(f_x1)
        f_x1 = self.drop_2(f_x1)

        s_x1 = self.fuse_1(s_x1,f_x1) # LateralConnection

        # Second set of blocks and fusion
        s_x2 = self.Block2(s_x1)
        s_x2 = self.MaxpoolSpa(s_x2)
        s_x2 = self.drop_3(s_x2)

        f_x2 = self.Block5(f_x1)
        f_x2 = self.MaxpoolSpa(f_x2)
        f_x2 = self.drop_4(f_x2)

        s_x2 = self.fuse_2(s_x2,f_x2) # LateralConnection

        # Third blocks and upsampling
        s_x3 = self.Block3(s_x2)
        s_x3 = self.upsample1(s_x3)
        s_x3 = self.drop_5(s_x3)

        f_x3 = self.Block6(f_x2)
        f_x3 = self.ConvBlock6(f_x3)
        f_x3 = self.drop_6(f_x3)

        # Final fusion and upsampling
        x_fusion = torch.cat((f_x3, s_x3), dim=1)
        x_final = self.upsample2(x_fusion)

        x_final = self.poolspa(x_final)
        x_final = self.ConvBlockLast(x_final)

        return rearrange(x_final, "b s t 1 1 -> b s t")
