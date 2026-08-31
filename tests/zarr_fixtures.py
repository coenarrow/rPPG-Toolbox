"""Synthetic Neckflix zarr-v3 stores for loader tests.

Mirrors the store schema written by the Neckflix preprocessor
(ghcr.io/coenarrow/neckflix >= 1.0.0); the cache contract is documented in
docs/architecture.md.
"""

import numpy as np
import zarr

TOOL_VERSION = "1.0.0"

# stream group -> (channel count, dtype), matching the preprocessor output.
STREAM_SPECS = {
    "rgb": (3, np.uint8),
    "ir": (1, np.uint16),
    "depth": (1, np.uint16),
}

# Distinct, deterministic base offsets so traces are tellable-apart in tests.
TRACE_OFFSETS = {"abp": 100.0, "cvp": 5.0, "ecg": 0.5}


def default_attrs(name):
    """Root attrs derived from a recording name like ``P030_S01_R1_0_D``.

    ``posture`` is the second-to-last underscore token; the trailing token is
    part of the name only and maps to no attr.
    """
    parts = name.split("_")
    return {
        "recording": name,
        "participant": parts[0][1:],   # unprefixed, e.g. "030"
        "session": parts[1],
        "repeat": parts[2],
        "posture": parts[-2],
        "source_resolution": [650, 650],
        "resized_to": None,
        "tool_version": TOOL_VERSION,
        "complete": True,
    }


def make_store(
    cache_dir,
    name="P030_S01_R1_0_D",
    *,
    attrs=None,
    perspectives=("1",),
    streams=("rgb", "ir", "depth"),
    traces=("abp", "cvp"),
    num_frames=12,
    hw=(8, 8),
    fps=30.0,
    frame_fill=None,
    trace_values=None,
    trace_lengths=None,
    events=False,
    extra_groups=(),
):
    """Write one synthetic store under ``cache_dir``; return its path.

    attrs          : dict merged over the defaults; a value of None REMOVES the key.
    num_frames     : int, or {stream: int} for per-stream frame counts.
    frame_fill     : {(persp, stream): int} constant pixel value (default: a
                     deterministic arange pattern).
    trace_values   : {(persp, stream, trace): np.ndarray} explicit trace data.
    trace_lengths  : {(persp, stream, trace): int} truncates that trace copy.
    events         : also write a root-level events/ group (arrays, no video child).
    extra_groups   : iterable of (persp, group_name) video-less groups placed
                     INSIDE an existing perspective.
    """
    path = cache_dir / f"{name}.zarr"
    root = zarr.open_group(str(path), mode="w")
    merged = default_attrs(name)
    for key, value in (attrs or {}).items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    root.attrs.update(merged)

    frames_per_stream = (
        dict(num_frames) if isinstance(num_frames, dict)
        else {s: num_frames for s in streams}
    )
    h, w = hw
    for persp in perspectives:
        pgroup = root.create_group(persp)
        for stream in streams:
            n_ch, dtype = STREAM_SPECS[stream]
            t = frames_per_stream.get(stream, 12)
            sgroup = pgroup.create_group(stream)
            video = sgroup.create_group("video")
            fill = (frame_fill or {}).get((persp, stream))
            if fill is None:
                data = (np.arange(n_ch * t * h * w) % 251).reshape(n_ch, t, h, w)
                data = data.astype(dtype)
            else:
                data = np.full((n_ch, t, h, w), fill, dtype=dtype)
            video.create_array("frames", data=data)
            video.create_array(
                "timestamps_us",
                data=(np.arange(t) * round(1e6 / fps)).astype(np.int64),
            )
            video.attrs.update({"fps": float(fps), "num_frames": int(t)})
            for trace in traces:
                length = (trace_lengths or {}).get((persp, stream, trace), t)
                values = (trace_values or {}).get((persp, stream, trace))
                if values is None:
                    values = TRACE_OFFSETS.get(trace, 1.0) + np.arange(
                        length, dtype=np.float64
                    )
                tgroup = sgroup.create_group(trace)
                tgroup.create_array("data", data=np.asarray(values, dtype=np.float64))

    if events:
        egroup = root.create_group("events")
        egroup.create_array("x", data=np.zeros(4, dtype=np.uint16))
        egroup.create_array("y", data=np.zeros(4, dtype=np.uint16))
        egroup.create_array("p", data=np.zeros(4, dtype=np.int8))
        egroup.create_array("t", data=np.zeros(4, dtype=np.int64))
    for persp, gname in extra_groups:
        root[persp].create_group(gname)
    return path


def make_unreadable_store(cache_dir, name="P099_S01_R1_0_D"):
    """A directory that globs as *.zarr but fails zarr.open_group."""
    path = cache_dir / f"{name}.zarr"
    path.mkdir()
    (path / "zarr.json").write_text("this is not json")
    return path


def base_cfg(cache_dir, **overrides):
    """A complete, valid loader cfg dict; override any key per test."""
    cfg = {
        "cache_dir": str(cache_dir),
        "channels": ["R", "G", "B", "I", "D"],
        "labels": ["ABP", "CVP"],
        "target_fps": 30.0,
        "window_seconds": 4 / 30,
        "stride_seconds": 4 / 30,
        "window_size": 4,
        "window_stride": 4,
        "random_windows": False,
        "filters": {},
        "label_norms": {"ABP": "zscore", "CVP": "zscore"},
        "allow_missing": False,
        "min_channels": 1,
        "min_labels": 1,
    }
    cfg.update(overrides)
    # Keep the physical window and the frame count consistent: tests set
    # window_size (the readable quantity here), and the seconds follow from it
    # at the fixture rate, exactly as the config translation would derive them.
    fps = cfg["target_fps"]
    if "window_seconds" not in overrides:
        cfg["window_seconds"] = cfg["window_size"] / fps
    if "stride_seconds" not in overrides:
        cfg["stride_seconds"] = cfg["window_stride"] / fps
    if "label_norms" not in overrides:
        cfg["label_norms"] = {label: "zscore" for label in cfg["labels"]}
    return cfg
