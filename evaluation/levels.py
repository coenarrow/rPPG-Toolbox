"""The aggregation hierarchy, and the contiguity rule that defines a section.

The standards specify different levels — ISO 81060-3 wants a per-subject mean
and SD, IEEE 1708 a per-subject MAE — so a report that hardcodes one level
loses the others. Each level here names its grouping keys and nothing else.
"""

from dataclasses import dataclass

import numpy as np

#: Coarsening order. A run populates every level it can reach; a LOSO fold
#: simply has one participant.
LEVELS = ("beat", "window", "section", "recording", "participant", "cohort")

#: What identifies one unit at each level.
GROUPING = {
    "beat": ("recording_id", "camera_id", "section_index", "beat_index"),
    "window": ("recording_id", "camera_id", "start_frame"),
    "section": ("recording_id", "camera_id", "section_index"),
    "recording": ("recording_id", "camera_id"),
    "participant": ("participant",),
    "cohort": (),
}


@dataclass(frozen=True)
class Section:
    """A maximal contiguous run of windows, stitched back into one trace."""

    recording_id: str
    camera_id: str
    index: int
    windows: list
    prediction: np.ndarray
    label: np.ndarray
    attrs: dict


def unit_id(level, **fields) -> str:
    """Stable identity for one unit, from that level's grouping keys."""
    keys = GROUPING[level]
    return "|".join(str(fields[key]) for key in keys) if keys else "all"


def sections(windows) -> list:
    """Split windows into maximal contiguous runs per (recording, camera).

    Contiguity is ``next.start_frame == previous.start_frame + len(previous)``.
    Overlapping windows (a non-zero ``STRIDE_SECONDS`` under the window length)
    therefore never stitch — each becomes its own single-window section, which
    is the safe degradation: stitching overlapping windows would double-count
    the shared samples.

    Only signals loaded ``raw`` reconstruct correctly, so callers pass
    absolute-class windows. A per-window z-scored signal would step at every
    seam.
    """
    by_camera = {}
    for window in windows:
        by_camera.setdefault((window.recording_id, window.camera_id), []).append(window)

    result, index = [], 0
    for (recording_id, camera_id), group in sorted(by_camera.items()):
        group.sort(key=lambda w: w.start_frame)
        run = [group[0]]
        for previous, window in zip(group, group[1:]):
            if window.start_frame == previous.start_frame + len(previous.label):
                run.append(window)
            else:
                result.append(_stitch(recording_id, camera_id, index, run))
                index += 1
                run = [window]
        result.append(_stitch(recording_id, camera_id, index, run))
        index += 1
    return result


def _stitch(recording_id, camera_id, index, run) -> Section:
    return Section(
        recording_id=recording_id,
        camera_id=camera_id,
        index=index,
        windows=list(run),
        prediction=np.concatenate([w.prediction for w in run]),
        label=np.concatenate([w.label for w in run]),
        attrs=dict(run[0].attrs),
    )
