"""Adapter that gives a plain backbone the batch-dict contract.

Some architectures are per-frame 2-D networks (DeepPhys, TS-CAN) and some are
3-D video networks. Both can satisfy :class:`DictModel` without being touched,
by folding the temporal axis into the batch axis on the way in and back out
again on the way out — which is the only thing this wrapper does.

A backbone that natively subclasses :class:`DictModel` (PhysMamba) does not
need this.
"""
import torch.nn as nn
from einops import rearrange

from neural_methods.model.DictModel import DictModel

INPUT_MODES = ('frames2d', 'video3d')


class SignalDictWrapper(DictModel):
    """Wrap ``backbone`` so it speaks ``(B, C, T, H, W) -> (B, S, T)``.

    ``input_mode='frames2d'``: the backbone sees ``(B*T, C, H, W)`` and returns
    ``(B*T, S)`` — one independent prediction per frame.
    ``input_mode='video3d'``: the backbone sees the clip unchanged and returns
    ``(B, S, T)`` itself.
    """

    def __init__(self, backbone, channels, traces, input_mode, frame_transform=None,
                 fs=0.0):
        super().__init__(channels=channels, traces=traces,
                         frame_transform=frame_transform, fs=fs)
        if input_mode not in INPUT_MODES:
            raise ValueError(f"Unknown input_mode {input_mode!r}; known: {INPUT_MODES}")
        self.backbone = backbone
        self.input_mode = input_mode

    def output_layers(self):
        """Whatever the wrapped backbone declares — the readout is its, not ours."""
        declared = getattr(self.backbone, "output_layers", None)
        return declared() if callable(declared) else ()

    def forward_video(self, video):
        if self.input_mode == 'frames2d':
            frames = rearrange(video, "b c t h w -> (b t) c h w")
            per_frame = self.backbone(frames)                       # (B*T, S)
            return rearrange(per_frame, "(b t) s -> b s t", b=video.shape[0])
        return self.backbone(video)                                 # (B, S, T)

    def extra_repr(self):
        return f"{super().extra_repr()}, input_mode={self.input_mode!r}"
