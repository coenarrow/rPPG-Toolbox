"""Trace vocabulary, trace/timestamp reading, and alignment to the frames.

FILL IN: ``TRACE_UNITS`` and ``read_stream_traces``. ``align_stream`` and
``trim_trailing_nans`` are dataset-agnostic and stay.
"""
from dataclasses import dataclass

import numpy as np

from template_cacher.scan import StreamInfo

#: FILL IN. The traces this dataset carries and the units it states for
#: them. Keys are the lowercase trace group names from the contract's
#: vocabulary (``ecg``, ``abp``, ``cvp``, ``ppg``, ``rr``); values are the
#: ``units`` attr written on each (``"mmHg"``, ``"mV"``, ``"arb"`` for a
#: signal with no physical unit, ...). Every trace group must carry one.
#:
#: Take the units from the dataset's own documentation or headers, and if a
#: source file states a unit, check it against this map rather than
#: trusting either alone -- a mislabelled clinical trace has no symptom.
TRACE_UNITS: dict[str, str] = {}


@dataclass
class AlignedStream:
    """Timestamps and traces cut to one stream's aligned length."""
    num_frames: int
    timestamps_us: np.ndarray        # (T,) int64, microseconds
    traces: dict[str, np.ndarray]    # name -> (T,) float64, index-aligned to the frames

    def truncated(self, num_frames: int) -> "AlignedStream":
        """Copy cut to the first ``num_frames`` frames."""
        return AlignedStream(
            num_frames=num_frames,
            timestamps_us=self.timestamps_us[:num_frames],
            traces={name: v[:num_frames] for name, v in self.traces.items()},
        )

    def interior_nans(self) -> dict[str, int]:
        """NaN samples per trace inside the aligned span (tails are already trimmed)."""
        counts = {name: int(np.isnan(v).sum()) for name, v in self.traces.items()}
        return {name: n for name, n in counts.items() if n}


def read_stream_traces(stream: StreamInfo) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per-frame timestamps and frame-rate traces for one stream.

    FILL IN. Return ``(timestamps_us, traces)`` where

    * ``timestamps_us`` is ``(T_video,)`` int64 microseconds, one per frame,
      strictly increasing, on a clock shared by every stream of the
      recording (the contract judges first-frame alignment across the
      modalities of a perspective by these);
    * ``traces`` maps a name in ``TRACE_UNITS`` to a ``(T,)`` float64 array
      **already sampled at the frame timestamps** -- one value per frame,
      in the units ``TRACE_UNITS`` states. Trailing NaN tails are fine
      (``align_stream`` trims them); interior NaNs are kept and reported.

    A dataset whose traces are not frame-aligned at source resamples them
    here (interpolate the native-rate trace at ``timestamps_us``). A
    dataset with no timestamps at all synthesises them from the nominal
    frame rate -- and says so in its cache spec.
    """
    raise NotImplementedError("read_stream_traces: timestamps + frame-rate traces")


def trim_trailing_nans(values: np.ndarray) -> np.ndarray:
    """Drop the trailing all-NaN tail; interior NaNs are kept."""
    finite = np.flatnonzero(~np.isnan(values))
    if len(finite) == 0:
        return values[:0]
    return values[: finite[-1] + 1]


def align_stream(
    video_frames: int,
    timestamps_us: np.ndarray,
    traces: dict[str, np.ndarray],
) -> AlignedStream:
    """Cut timestamps and traces to ``min(video frames, trace lengths)``.

    Each modality must be internally consistent -- ``timestamps_us``,
    ``video`` and every trace agree on ``T`` -- so the shortest wins and
    everything is truncated to it. Modalities may still differ from one
    another; the reader reconciles that.
    """
    if len(timestamps_us) < video_frames:
        raise ValueError(f"{len(timestamps_us)} timestamps < {video_frames} video frames")
    trimmed = {name: trim_trailing_nans(v) for name, v in traces.items()}
    num_frames = min([video_frames] + [len(v) for v in trimmed.values()])
    return AlignedStream(
        num_frames=num_frames,
        timestamps_us=np.asarray(timestamps_us[:num_frames], dtype=np.int64),
        traces={name: v[:num_frames] for name, v in trimmed.items()},
    )
