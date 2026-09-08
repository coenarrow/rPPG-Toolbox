"""A backbone template on the multi-signal contract. Copy, rename, replace.

This is not a real architecture. It is the smallest ``nn.Module`` that
satisfies everything ``src.models.MultiTraceModel`` and ``src.trainer`` ask
of a backbone, with each requirement marked. Copy it to
``neural_methods/model/<Name>.py``, rename the class, and swap the layers for
the published network. Then register it in ``src/models.py`` and write
``configs/models/<name>.yaml`` — the steps are in ``docs/adding_a_model.md``.

What a backbone is, in one paragraph: a function from one preprocessed
clip (or frame) to one trace. It never sees the batch dict, never knows which
trace it is predicting, never normalises its input and never computes a
loss. The wrapper makes one copy per trace and owns the dict; the dataset
does the preprocessing; the trainer owns the loss.
"""

import torch
import torch.nn as nn
from einops import rearrange


class TemplateNet(nn.Module):
    """``(B, C_in, T, H, W) -> (B, 1, T)``: a clip backbone.

    For a per-frame backbone (``per_frame=True`` in the builder) the contract
    is instead ``(N, C_in, H, W) -> (N, 1)``; see ``DeepPhys.py``.
    """

    # INTERIM ONLY, for a migration that has not got there yet. The
    # destination is an adaptive stage that accepts any window length and is
    # an exact no-op at the paper's; none of the ten migrated models declares
    # either of these any more (docs/adding_a_model.md, "Any frame size, any
    # window length"). Until yours has one, declare that the window must be a
    # multiple of something ...
    temporal_divisor = 1
    # ... or that it must be exactly one clip length. Delete both once the
    # adaptive stage is in; while they are here the trainer reads them off the
    # first copy and refuses a WINDOW_SECONDS that does not fit, naming the
    # fix, rather than truncating.
    # temporal_length = 128

    def __init__(self, in_channels: int = 3, hidden: int = 16):
        """``in_channels`` is REQUIRED and is the only width the wrapper
        always passes: ``len(interface.CHANNELS) * len(input_blocks)``.
        Anything else the interface determines (an ``img_size`` for a dense
        layer, say) is an argument the builder passes. Every published
        hyperparameter is a default here, never a YAML key."""
        super().__init__()
        self.in_channels = in_channels
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(hidden),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1)),      # keep T, drop H and W
        )
        # REQUIRED: the readout is activation-free and one wide, with a bias
        # of exactly one element. The trainer seeds that bias with the
        # trace's physiological prior and exempts it from weight decay.
        self.readout = nn.Conv1d(hidden, 1, kernel_size=1, bias=True)

    def output_layers(self):
        """REQUIRED. The activation-free readout(s) of this one copy."""
        return (self.readout,)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """``(B, C_in, T, H, W) -> (B, 1, T)``.

        The input is already resized and preprocessed by the dataset: the
        interface's channels stacked in order, once per named preprocessing
        block, blocks concatenated in the order the builder lists them. A
        backbone reading two blocks (DeepPhys) slices them off the channel
        axis itself.
        """
        x = self.features(video)
        x = rearrange(x, "b c t 1 1 -> b c t")      # einops, never view/permute
        return self.readout(x)                       # (B, 1, T), three dims
