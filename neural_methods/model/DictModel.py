"""Base class for models that speak the Neckflix batch dict.

The contract, in one place, so every architecture below it stays exactly the
architecture it was:

* ``forward(batch)`` takes the loader's dict and returns *the same dict* with a
  ``predictions`` entry added — nothing is dropped on the way through, so at
  any point in training or evaluation a single object carries the frames, the
  labels, the masks, the metadata and the predictions, each identifiable by key.
* Subclasses implement ``forward_video(video)``: a plain
  ``(B, C_in, T, H, W)`` tensor in, a raw ``(B, S, T)`` tensor out. No dicts, no
  masks, no metadata — that is what keeps the retrofit to an existing
  architecture a signature change rather than a rewrite.
* Channel and signal *order* is owned here (``self.channels`` / ``self.traces``),
  never inferred from dict iteration order.

``C_in`` is ``len(channels) * frame_transform.channel_multiplier``: a
``DATA_TYPE`` of two transforms feeds each backbone two channel blocks of the
same clip, matching upstream toolbox semantics.
"""

import torch
import torch.nn as nn
from einops import rearrange

from neural_methods.batch import (
    FRAMES, PREDICTIONS, require_batch_dict, split_signals, stack_frames,
)
from neural_methods.frame_transforms import FrameTransform
from neural_methods.signals import validate_channels, validate_traces


class DictModel(nn.Module):
    """Dict in, dict out; subclasses only implement the tensor-level forward."""

    #: Temporal constraints on the window length T, **declared, never silently
    #: handled**: a stride/upsample round trip that only closes on a multiple of
    #: k sets ``temporal_divisor = k``; an architecturally fixed length sets
    #: ``temporal_length``. The builder checks the config's derived T against
    #: these at construction time, which is where a bad window should fail —
    #: the legacy trainers truncated the batch instead, and a silently shortened
    #: window is a silently different experiment.
    temporal_divisor = 1
    temporal_length = None

    def __init__(self, channels=("R", "G", "B"), traces=("PPG",), frame_transform=None,
                 fs=0.0):
        super().__init__()
        self.channels = tuple(validate_channels(list(channels)))
        self.traces = tuple(validate_traces(list(traces)))
        self.frame_transform = frame_transform if frame_transform is not None \
            else FrameTransform(("Raw",))
        # A buffer, not a plain attribute, so the rate rides in the state dict:
        # a checkpoint knows what it was trained at, and at inference the data
        # is decimated to the model's rate rather than the other way round.
        self.register_buffer("_fs", torch.tensor(float(fs)))

    @property
    def fs(self) -> float:
        """Frame rate this model's dynamics were learned at, in Hz."""
        return float(self._fs)

    @property
    def in_channels(self) -> int:
        """Channel count the backbone is built for, after the frame transform."""
        return len(self.channels) * self.frame_transform.channel_multiplier

    @property
    def out_signals(self) -> int:
        return len(self.traces)

    def output_layers(self):
        """The activation-free readout module(s), in ``self.traces`` order.

        Either one layer whose output width is ``S`` (head style A, the
        default) or ``S`` per-signal copies (style B). Exactly two pieces of
        trainer-side machinery need to find them: the physiological bias
        initialisation, and the weight-decay exemption that stops decay from
        dragging a raw-mmHg prediction toward zero. Returning ``()`` opts a
        model out of both.
        """
        return ()

    def prepare_frames(self, batch) -> torch.Tensor:
        """``batch['frames']`` -> the transformed ``(B, C_in, T, H, W)`` tensor."""
        video = stack_frames(require_batch_dict(batch)[FRAMES], self.channels)
        return self.frame_transform(video)

    def forward_video(self, video):
        """``(B, C_in, T, H, W)`` -> ``(B, S, T)``. Implemented by each architecture."""
        raise NotImplementedError

    def predict(self, batch) -> dict:
        """Just the predictions dict, for callers that do not want the whole batch."""
        return split_signals(self.forward_video(self.prepare_frames(batch)), self.traces)

    def forward(self, batch):
        """Dict in, dict out — or tensor in, tensor out for the legacy datasets.

        The tensor branch exists so the upstream tuple-contract trainers (PURE,
        UBFC-rPPG, ...) keep working against exactly the shapes they always
        passed: ``(B, C, T, H, W)`` in, ``(B, T)`` out for a single-signal
        model. New code passes the batch dict, and gets the batch dict back.
        """
        if torch.is_tensor(batch):
            raw = self.forward_video(self.frame_transform(batch))
            return rearrange(raw, "b 1 t -> b t") if self.out_signals == 1 else raw
        return {**require_batch_dict(batch), PREDICTIONS: self.predict(batch)}

    def extra_repr(self) -> str:
        return f"channels={list(self.channels)}, traces={list(self.traces)}"

