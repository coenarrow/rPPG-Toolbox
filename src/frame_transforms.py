"""Frame preprocessing: the interface's ``INPUT_PREPROCESSING`` vocabulary.

The cache holds raw pixel values; the dataset (``src.inputs``) resizes each
window to the interface's ``RESIZE`` and then applies every named transform
to each channel's plane separately, with that plane's own statistics, so a
channel's block never depends on which other channels the interface demands.
Every function here takes and returns one ``(T, H, W)`` float plane.

The formulas are the upstream toolbox's ``standardized_data`` and
``diff_normalize_data``, statistics taken per window. Every reshape is einops.
"""

import torch
import torch.nn.functional as F
from einops import rearrange

_EPS = 1e-7


def standardized(plane: torch.Tensor) -> torch.Tensor:
    """Z-score a plane with its own mean and std."""
    out = (plane - plane.mean()) / plane.std().clamp_min(_EPS)
    return torch.nan_to_num(out)


def diff_normalized(plane: torch.Tensor) -> torch.Tensor:
    """Frame-to-frame difference normalised by its own std, zero-padded to T.

    ``d_t = (x_{t+1} - x_t) / (x_{t+1} + x_t + eps)``, divided by the window's
    std, with a zero frame appended so the temporal length is unchanged.
    """
    later, earlier = plane[1:], plane[:-1]
    diff = (later - earlier) / (later + earlier + _EPS)
    diff = torch.nan_to_num(diff / diff.std().clamp_min(_EPS))
    return torch.cat([diff, torch.zeros_like(plane[:1])], dim=0)


#: ``INPUT_PREPROCESSING`` vocabulary: name -> transform of one plane.
FRAME_TRANSFORMS = {
    "Raw": lambda plane: plane,
    "Standardized": standardized,
    "DiffNormalized": diff_normalized,
}


def resize_video(plane: torch.Tensor, size) -> torch.Tensor:
    """Bilinear spatial resize of ``(T, H, W)`` to ``size = (H', W')``.

    Frames are folded into the batch axis so ``interpolate`` sees plain 2-D
    images; a plane already at ``size`` is returned as is.
    """
    height, width = size
    if tuple(plane.shape[-2:]) == (height, width):
        return plane
    frames = rearrange(plane, "t h w -> t 1 h w")
    resized = F.interpolate(frames, size=(height, width), mode="bilinear",
                            align_corners=False)
    return rearrange(resized, "t 1 h w -> t h w")
