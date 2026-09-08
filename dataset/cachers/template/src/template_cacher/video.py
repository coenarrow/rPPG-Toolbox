"""Frame decoding and per-frame resize.

FILL IN: ``decode_video``. Decode streaming, frame by frame, resizing each
before it is stacked -- a full-resolution recording rarely fits in memory.
"""
from pathlib import Path

import numpy as np

from template_cacher.scan import StreamInfo


def decode_video(
    stream: StreamInfo,
    modality: str,
    num_frames: int,
    resize: tuple[int, int] | None = None,
) -> tuple[np.ndarray, float]:
    """Decode up to ``num_frames`` frames as ``(C, T, H, W)`` plus the container rate.

    FILL IN. Requirements:

    * The array is ``(C, T, H, W)`` in exactly that order, ``C`` fixed by the
      modality's place in the contract vocabulary (``rgb`` 3; ``gr``, ``ir``,
      ``depth``, ``t`` 1). Any dtype: keep the sensor's native one (uint8
      RGB, uint16 depth/IR, float thermal) -- the contract does not care
      and the consumer casts.
    * ``resize`` is ``(H, W)``; ``None`` keeps the native size. Pick the
      interpolation per modality: area for intensity images, **nearest for
      depth** so invalid-depth zeros are never blended into false distances.
    * The float returned is the container's own frame rate, used only to
      warn when it disagrees with ``NOMINAL_FPS``; the perspective attr is
      the nominal rate.
    * ``stream.source`` may be a single container file or a directory of
      image files (``PURE`` is PNGs); the shape contract is the same.

    Decoding fewer frames than ``num_frames`` is allowed -- the CLI truncates
    the timestamps and traces to match and warns.
    """
    raise NotImplementedError("decode_video: frames as (C, T, H, W)")
