"""PURE's one trace: the oximeter waveform, put on the frame grid by timestamp."""
from dataclasses import dataclass

import numpy as np

from pure_cacher.scan import StreamInfo, load_json

#: The finger pulse oximeter's waveform has no physical unit.
TRACE_UNITS: dict[str, str] = {"ppg": "arb"}


@dataclass
class AlignedStream:
    """Timestamps and traces cut to one stream's aligned length."""
    num_frames: int
    timestamps_us: np.ndarray        # (T,) int64, microseconds
    traces: dict[str, np.ndarray]    # name -> (T,) float64, index-aligned to the frames

    def truncated(self, num_frames: int) -> "AlignedStream":
        return AlignedStream(
            num_frames=num_frames,
            timestamps_us=self.timestamps_us[:num_frames],
            traces={name: v[:num_frames] for name, v in self.traces.items()},
        )

    def interior_nans(self) -> dict[str, int]:
        counts = {name: int(np.isnan(v).sum()) for name, v in self.traces.items()}
        return {name: n for name, n in counts.items() if n}


def read_stream_traces(stream: StreamInfo) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per-frame timestamps and the waveform sampled at them.

    Timestamps are the ``/Image`` capture times, epoch nanoseconds in the
    source, written as epoch **microseconds** (int64) so the absolute capture
    time survives. The waveform is recorded at ~60 Hz on the same clock and
    is linearly interpolated at the frame times; frames outside the
    oximeter's span get NaN rather than a clamped edge value (the two clocks
    start and stop within a frame or two of each other, so this is at most
    a few samples, and ``align_stream`` trims a NaN tail).
    """
    name = stream.source.name
    data = load_json(stream.source.parent / f"{name}.json")
    image_ns = np.array([r["Timestamp"] for r in data["/Image"]], dtype=np.int64)
    package = data["/FullPackage"]
    package_ns = np.array([r["Timestamp"] for r in package], dtype=np.int64)
    waveform = np.array([r["Value"]["waveform"] for r in package], dtype=np.float64)
    if not np.all(np.diff(image_ns) > 0):
        raise ValueError(f"{name}: /Image timestamps are not strictly increasing")
    order = np.argsort(package_ns, kind="stable")
    package_ns, waveform = package_ns[order], waveform[order]

    # Interpolate in seconds relative to the first frame: epoch nanoseconds as
    # float64 would carry only ~256 ns of resolution, and np.interp wants
    # floating x anyway.
    t0 = image_ns[0]
    frame_s = (image_ns - t0) / 1e9
    sample_s = (package_ns - t0) / 1e9
    ppg = np.interp(frame_s, sample_s, waveform)
    ppg[(frame_s < sample_s[0]) | (frame_s > sample_s[-1])] = np.nan
    return image_ns // 1000, {"ppg": ppg}


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
    """Cut timestamps and traces to ``min(video frames, trace lengths)``."""
    if len(timestamps_us) < video_frames:
        raise ValueError(f"{len(timestamps_us)} timestamps < {video_frames} video frames")
    trimmed = {name: trim_trailing_nans(v) for name, v in traces.items()}
    num_frames = min([video_frames] + [len(v) for v in trimmed.values()])
    return AlignedStream(
        num_frames=num_frames,
        timestamps_us=np.asarray(timestamps_us[:num_frames], dtype=np.int64),
        traces={name: v[:num_frames] for name, v in trimmed.items()},
    )
