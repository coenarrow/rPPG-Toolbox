import numpy as np
from einops import rearrange

from src.evaluation.post_process import _detrend


def detrend(input_signal, lambda_value):
    """Smoothness-priors detrending — see :func:`src.evaluation.post_process._detrend`.

    Kept as a name here because every method imports it from this module; the
    implementation is shared so the O(n) banded solve benefits both the
    supervised and unsupervised paths.
    """
    return _detrend(input_signal, lambda_value)


def rgb_trace(frames):
    """Per-frame spatial mean, ``(T, 3)`` float64.

    Accepts a clip ``(T, H, W, 3)`` **or** an already-reduced ``(T, 3)`` trace.
    Every traditional method begins by collapsing each frame to its mean RGB,
    so a caller running several methods over one clip reduces once and passes
    the trace to all of them — the difference between decoding-and-averaging
    seven times and once.
    """
    array = np.asarray(frames, dtype=np.float64)
    if array.ndim == 2:
        return array
    if array.ndim == 4:
        return array.mean(axis=(1, 2))
    raise ValueError(
        f"Expected frames (T, H, W, 3) or an RGB trace (T, 3), got shape {array.shape}"
    )


def process_video(frames):
    """Spatial mean as a single-batch series: ``(1, 3, T)``.

    The leading singleton is what GREEN/LGI/PBV/OMIT index against; keeping it
    here means those methods need no shape handling of their own.
    """
    return rearrange(rgb_trace(frames), "t c -> 1 c t")
