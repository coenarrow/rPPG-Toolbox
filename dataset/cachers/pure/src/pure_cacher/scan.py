"""Discover PURE recordings and probe their metadata. No decoding happens here.

Raw layout (the public release, one directory per recording ``SS-TT``):

    {input_dir}/01-01/01-01/Image<ns>.png ...   frames, nanosecond capture time in the name
    {input_dir}/01-01/01-01.json                /Image timestamps + /FullPackage oximeter records

The JSON is both the metadata and the trace source, so it is read here for
the root attrs and again in ``traces.py`` for the waveform.
"""
import json
import re
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_RECORDING = re.compile(r"^\d\d-\d\d$")
_FRAME = re.compile(r"^Image(\d+)\.png$")

#: The six recording setups, per the dataset's published description.
#: The token is what the directory name carries; the name is a convenience.
SETUP_NAMES = {
    "01": "steady",
    "02": "talking",
    "03": "slow_translation",
    "04": "fast_translation",
    "05": "small_rotation",
    "06": "medium_rotation",
}

#: Per-sample oximeter fields in ``/FullPackage`` besides ``waveform``,
#: keyed exactly as the JSON spells them. Numeric ones are summarised
#: (min/max/mean) into the root attrs; boolean ones are counted.
OXIMETER_NUMERIC = ("pulseRate", "o2saturation", "signalStrength", "barGraph")
OXIMETER_FLAGS = ("beep", "droppingo2Sat", "probeError", "searching", "searchingToLong")


@dataclass
class StreamInfo:
    """One video stream of one perspective: where it is and how long it is."""
    source: Path            # the directory of Image<ns>.png frames
    num_frames: int


@dataclass
class RecordingInfo:
    """Everything the scan learned about one recording."""
    name: str
    attrs: dict
    perspectives: dict[str, dict[str, StreamInfo]] = field(default_factory=dict)
    source_resolution: list[int] | None = None


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def frame_timestamp_ns(filename: str) -> int:
    """``Image1392643993642815000.png`` -> 1392643993642815000."""
    match = _FRAME.match(filename)
    if match is None:
        raise ValueError(f"not a PURE frame filename: {filename!r}")
    return int(match.group(1))


def list_frames(image_dir: Path) -> list[Path]:
    """The ``Image<ns>.png`` files in chronological order.

    Sorted by the timestamp in the name, not lexicographically, and matched by
    pattern rather than ``*.png`` so stray files (``Thumbs.db`` sits beside the
    frames in at least one recording) are never decoded as frames.
    """
    frames = [p for p in image_dir.iterdir() if _FRAME.match(p.name)]
    return sorted(frames, key=lambda p: frame_timestamp_ns(p.name))


def png_resolution(path: Path) -> list[int]:
    """``[H, W]`` from the PNG IHDR chunk; no decode."""
    with open(path, "rb") as f:
        header = f.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    width, height = struct.unpack(">II", header[16:24])
    return [int(height), int(width)]


def _stats(values: list) -> dict:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"min": None, "max": None, "mean": None}
    return {"min": float(x.min()), "max": float(x.max()), "mean": round(float(x.mean()), 3)}


def build_root_attrs(name: str, data: dict) -> dict:
    """Root attrs for one recording, from its name and its JSON sidecar.

    Everything the raw data states about the recording is kept, at the
    recording level: identity tokens, capture time and measured rates, and
    a summary of every oximeter field the ``/FullPackage`` records carry
    beside the waveform (the waveform itself is the ``ppg`` trace). Field
    names are the JSON's own.
    """
    subject, setup = name.split("-")
    image_ts = np.array([r["Timestamp"] for r in data["/Image"]], dtype=np.int64)
    package = data["/FullPackage"]
    package_ts = np.array([r["Timestamp"] for r in package], dtype=np.int64)
    duration_s = float(image_ts[-1] - image_ts[0]) / 1e9
    oximeter_duration_s = float(package_ts[-1] - package_ts[0]) / 1e9
    return {
        "participant": subject,                       # "01": the dataset's own token, verbatim
        "recording": name,
        "subject": subject,
        "setup": setup,
        "setup_name": SETUP_NAMES.get(setup),
        "capture_start_utc": datetime.fromtimestamp(
            image_ts[0] / 1e9, tz=timezone.utc).isoformat(timespec="milliseconds"),
        "duration_s": round(duration_s, 3),
        "frame_rate_measured": round((len(image_ts) - 1) / duration_s, 3) if duration_s else None,
        "oximeter": {
            "num_samples": len(package),
            "sample_rate_measured": (
                round((len(package) - 1) / oximeter_duration_s, 3) if oximeter_duration_s else None),
            "start_offset_ms": round(float(package_ts[0] - image_ts[0]) / 1e6, 1),
            "end_offset_ms": round(float(package_ts[-1] - image_ts[-1]) / 1e6, 1),
            **{key: _stats([r["Value"][key] for r in package]) for key in OXIMETER_NUMERIC},
            "flag_counts": {key: int(sum(bool(r["Value"][key]) for r in package))
                            for key in OXIMETER_FLAGS},
        },
        "alignment": "timestamp",                     # how ppg was put on the frame grid
    }


def discover_recordings(input_dir: Path) -> list[str]:
    """Names of the ``SS-TT`` recording directories under ``input_dir``.

    A directory with neither the image directory nor the JSON is not a
    recording and is left out silently: the public release ships ``06-02``
    as an empty directory.
    """
    input_dir = Path(input_dir)
    dirs = [p for p in sorted(input_dir.iterdir()) if p.is_dir() and _RECORDING.match(p.name)]
    if not dirs:
        raise FileNotFoundError(f"No SS-TT recording directories under {input_dir}")
    return [d.name for d in dirs if (d / d.name).is_dir() or (d / f"{d.name}.json").exists()]


def scan_recording(input_dir: Path, name: str) -> RecordingInfo:
    """Probe one recording: frame listing, JSON sidecar, one PNG header."""
    rec_dir = Path(input_dir) / name
    image_dir, json_path = rec_dir / name, rec_dir / f"{name}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"{name}: missing {json_path.name}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"{name}: missing image directory {image_dir}")
    frames = list_frames(image_dir)
    if not frames:
        raise ValueError(f"{name}: no Image<ns>.png frames in {image_dir}")
    data = load_json(json_path)
    # The frame files and the JSON's /Image list describe the same frames.
    # Their timestamps must agree exactly; a drift means frames were
    # added or removed after the sidecar was written.
    file_ts = np.array([frame_timestamp_ns(p.name) for p in frames], dtype=np.int64)
    image_ts = np.array([r["Timestamp"] for r in data["/Image"]], dtype=np.int64)
    if not np.array_equal(file_ts, image_ts):
        raise ValueError(
            f"{name}: {len(frames)} frame files vs {len(image_ts)} /Image records, "
            "or their timestamps differ; the sidecar does not describe these frames"
        )
    return RecordingInfo(
        name=name,
        attrs=build_root_attrs(name, data),
        perspectives={"1": {"rgb": StreamInfo(source=image_dir, num_frames=len(frames))}},
        source_resolution=png_resolution(frames[0]),
    )
