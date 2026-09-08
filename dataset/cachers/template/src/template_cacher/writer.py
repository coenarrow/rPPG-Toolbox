"""Zarr store creation: the contract's schema and the compression settings.

Dataset-agnostic. Nothing here should need editing for a new dataset; the
layout it writes is docs/cache-contract.md, and the settings match the
Neckflix cacher so every cache in the family reads alike.
"""
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec
from zarr.codecs.numcodecs import Delta

_COMPRESSOR = BloscCodec(cname="zstd", clevel=9, shuffle="bitshuffle")

#: FILL IN. The nominal frame rate written on every video perspective. The
#: contract's ``fps`` attr is the nominal identity of the perspective, not a
#: measured rate; the reader reconciles per-store drift against it.
NOMINAL_FPS: float = 30.0


def init_store(store_path: Path, attrs: dict) -> zarr.Group:
    """Create (or wipe and recreate) a recording store and write its root attrs.

    Refuses a non-string ``participant`` here, at write time: the contract
    requires a string and the validator would fail the store later anyway,
    but this is the one clause a cacher gets wrong silently (an int from a
    CSV column), and every store in a run would carry the fault.
    """
    participant = attrs.get("participant")
    if not isinstance(participant, str):
        raise TypeError(
            f"root attr 'participant' must be a string, got "
            f"{type(participant).__name__} {participant!r}. Take the dataset's "
            "identifier verbatim as text; do not parse it to a number."
        )
    root = zarr.open_group(store_path, mode="w")
    for key, value in attrs.items():
        root.attrs[key] = value
    return root


def write_perspective_group(root: zarr.Group, name: str, fps: float | None) -> zarr.Group:
    """Create a perspective group and stamp its nominal frame rate.

    ``fps`` lives on the perspective, not on each video group: a perspective's
    modalities are pixel-aligned and share a rate. ``None`` is the contract's
    spelling for a perspective with no frame rate (an event camera).
    """
    group = root.create_group(name)
    group.attrs["fps"] = fps
    return group


def write_timestamps(parent: zarr.Group, timestamps_us: np.ndarray) -> None:
    """Write ``timestamps_us/data`` under ``parent``."""
    parent.create_group("timestamps_us").create_array(
        "data", data=np.asarray(timestamps_us, dtype=np.int64)
    )


def write_video_group(
    modality_group: zarr.Group,
    frames: np.ndarray,
    timestamps_us: np.ndarray,
    chunk_size: int = 32,
) -> None:
    """Write ``video/data`` and the sibling ``timestamps_us/data``.

    Both sit directly under the modality: the timestamps describe the
    modality's timeline, not the pixel array. Chunks are ``(C, 32, H, W)`` so
    a training window is one or two chunk reads; the Delta filter plus
    blosc-zstd is what the Neckflix cacher uses, kept identical so one reader
    configuration serves every cache.
    """
    chunks = (frames.shape[0], chunk_size) + frames.shape[2:]
    video_group = modality_group.create_group("video")
    video_group.create_array(
        "data",
        data=frames,  # (C, T, H, W)
        chunks=chunks,
        compressors=[_COMPRESSOR],
        filters=[Delta(dtype=str(frames.dtype))],
    )
    write_timestamps(modality_group, timestamps_us)


def write_trace(parent: zarr.Group, name: str, values: np.ndarray, units: str) -> None:
    """Write ``{name}/data`` with its required ``units`` attr under ``parent``.

    A frame-rate trace shares its modality's clock, so nothing else is
    needed: its ``(T,)`` length is index-aligned to the modality's frames.
    """
    group = parent.create_group(name)
    group.create_array("data", data=np.asarray(values, dtype=np.float64))
    group.attrs["units"] = units


def mark_complete(root: zarr.Group, modalities: list[str], perspectives: list[str]) -> None:
    """Stamp a store as fully written. Always the last write.

    Records what the run *asked for*, not what was written, so a recording
    whose source lacks a modality is not rebuilt on every run forever. This
    is the cacher's own skip/rebuild bookkeeping -- the contract has no
    ``complete`` clause and the reader never looks at it.
    """
    root.attrs["complete"] = {
        "modalities": sorted(modalities),
        "perspectives": sorted(perspectives),
    }


def covers(
    store_path: Path,
    modalities: list[str],
    perspectives: list[str],
    resize: tuple[int, int] | None = None,
) -> bool:
    """True iff the store is complete AND spans everything this run wants.

    ``resize`` must match exactly: frames are already downsampled on disk,
    so a store built at 64x64 cannot serve a request for 200x200. Anything
    unreadable (a truncated ``zarr.json`` from a killed run) is not a
    covering store, so the caller rebuilds instead of raising.
    """
    if not Path(store_path).exists():
        return False
    try:
        attrs = dict(zarr.open_group(store_path, mode="r").attrs)
    except Exception:
        return False
    done = attrs.get("complete")
    if not isinstance(done, dict):
        return False
    if attrs.get("resized_to") != (list(resize) if resize else None):
        return False
    return (
        set(modalities) <= set(done.get("modalities", []))
        and set(perspectives) <= set(done.get("perspectives", []))
    )
