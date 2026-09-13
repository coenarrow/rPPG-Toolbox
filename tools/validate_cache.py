"""Cache validator — the admission mechanism, made executable.

The contract: docs/cache-contract.md (normative; every clause it tabulates
under "What the validator checks" is a check below). Run this after
generating a cache; a store this passes is admissible, full stop — there is
no ``complete``/``tool_version`` gate any more.

    uv run python tools/validate_cache.py <cache-dir | store.zarr ...>
"""
import argparse
import math
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import zarr

from src.signal_transforms import MODALITY_CHANNELS, TRACE_KEYS

#: Modalities whose frame representation the contract has not pinned yet. Their
#: channel count goes unchecked; the CLI says so rather than passing silently.
UNPINNED = tuple(m for m, channels in MODALITY_CHANNELS.items() if channels is None)


class Violation(NamedTuple):
    where: str          # "store/perspective/modality" style path
    message: str

    def __str__(self):
        return f"{self.where}: {self.message}"


def _array_at(group, name):
    """The zarr Array at ``<name>/data``, or None if there is not one there.

    v2 puts every array at ``<name>/data``. A store that writes a bare array at
    ``<name>`` (the v1 shape) or a group at ``<name>/data`` has to come back as
    a violation like any other: ``main()`` sweeps a whole cache directory, so a
    validator that raises takes every store after the malformed one with it.
    """
    if name not in group or not isinstance(group[name], zarr.Group):
        return None
    node = group[name]
    if "data" not in node or not isinstance(node["data"], zarr.Array):
        return None
    return node["data"]


def _check_modality(out, where, modality, group):
    """Within-modality checks; returns the first timestamp, or None."""
    # v2 puts every array at <name>/data, so a bare array child is a trace (or a
    # video) written in the v1 shape. The walks below see groups only, so
    # without this the store passes clean. Reported before the required-array
    # check so a v1-shaped video says what is actually wrong with it.
    for key in group.array_keys():
        out.append(Violation(
            f"{where}/{key}",
            "array child of a modality; v2 puts every array at <name>/data"))
    video, stamps_array = _array_at(group, "video"), _array_at(group, "timestamps_us")
    for required, array in (("timestamps_us", stamps_array), ("video", video)):
        if array is None:
            out.append(Violation(where, f"missing {required}/data"))
    if video is None or stamps_array is None:
        return None
    if video.ndim != 4:
        out.append(Violation(where, f"video/data is {video.ndim}-D, want (C, T, H, W)"))
        return None
    expected = MODALITY_CHANNELS[modality]
    if expected is not None and video.shape[0] != len(expected):
        out.append(Violation(
            where, f"video/data has C={video.shape[0]}, {modality} wants "
                   f"{len(expected)} ({', '.join(expected)})"))
    stamps = stamps_array[:]
    frames = video.shape[1]
    if not frames:
        out.append(Violation(where, "video/data has T=0; the recording is empty"))
    if stamps.shape != (frames,):
        out.append(Violation(
            where, f"timestamps_us length {stamps.shape} vs T={frames}"))
        stamps = None
    elif frames > 1 and not np.all(np.diff(stamps) > 0):
        out.append(Violation(where, "timestamps_us not strictly increasing"))
    for key in group.group_keys():
        if key in ("timestamps_us", "video"):
            continue
        sub = f"{where}/{key}"
        if key not in TRACE_KEYS:
            out.append(Violation(
                sub, f"unknown trace group; vocabulary: {sorted(TRACE_KEYS)}"))
            continue
        trace = _array_at(group, key)
        if trace is None:
            out.append(Violation(sub, "missing data array"))
            continue
        if trace.shape != (frames,):
            out.append(Violation(
                sub, f"trace length {trace.shape} is not index-aligned to "
                     f"video T={frames}"))
        if not np.issubdtype(trace.dtype, np.floating):
            out.append(Violation(sub, f"trace dtype {trace.dtype}, want float"))
        if "units" not in group[key].attrs:
            out.append(Violation(sub, "missing required 'units' attr"))
    return float(stamps[0]) if stamps is not None and stamps.size else None


def _nominal_fps(out, where, attrs):
    """The perspective's nominal frame rate, or None when it has none.

    ``fps`` is required, but its *value* may be null (or NaN): a perspective
    with no frame rate, which is what an event camera is. Anything else must
    be a positive finite number. None is returned for the no-rate case so the
    caller skips every check that needs a rate.
    """
    if "fps" not in attrs:
        out.append(Violation(where, "missing required perspective attr 'fps'"))
        return None
    fps = attrs["fps"]
    if fps is None or (isinstance(fps, float) and math.isnan(fps)):
        return None
    if (isinstance(fps, bool) or not isinstance(fps, (int, float))
            or not math.isfinite(fps) or fps <= 0):
        out.append(Violation(
            where, f"perspective attr 'fps' is {fps!r}, want a positive "
                   f"number or null (no frame rate)"))
        return None
    return float(fps)


def validate_store(path) -> list:
    """Every contract clause, itemised. Empty list = admissible."""
    path = Path(path)
    out = []
    try:
        root = zarr.open_group(str(path), mode="r")
    except Exception as error:                       # unreadable = one violation
        return [Violation(path.name, f"cannot open as a zarr group: {error}")]
    if "participant" not in root.attrs:
        out.append(Violation(path.name, "missing required root attr 'participant'"))
    elif not isinstance(root.attrs["participant"], str):
        # Any identifier in any format, but a string: the split machinery
        # matches it exactly, and an int 15 never equals a configured "015".
        participant = root.attrs["participant"]
        out.append(Violation(
            path.name, f"root attr 'participant' is {type(participant).__name__} "
                       f"{participant!r}, want a string"))
    perspectives = list(root.group_keys())
    if not perspectives:
        out.append(Violation(path.name, "store has no perspective groups"))
    for perspective in perspectives:
        cam = root[perspective]
        where = f"{path.name}/{perspective}"
        fps = _nominal_fps(out, where, cam.attrs)
        modalities = list(cam.group_keys())
        trace_sets, first_stamps = {}, {}
        for modality in modalities:
            sub = f"{where}/{modality}"
            if modality not in MODALITY_CHANNELS:
                out.append(Violation(
                    sub, f"unknown modality; vocabulary: "
                         f"{sorted(MODALITY_CHANNELS)}"))
                continue
            first = _check_modality(out, sub, modality, cam[modality])
            trace_sets[modality] = frozenset(
                k for k in cam[modality].group_keys()
                if k not in ("timestamps_us", "video"))
            if first is not None:
                first_stamps[modality] = first
        if len(set(trace_sets.values())) > 1:
            listing = "; ".join(f"{m}: {sorted(s)}" for m, s in trace_sets.items())
            out.append(Violation(
                where, f"modalities carry different trace sets ({listing})"))
        if fps is not None and len(first_stamps) > 1:
            budget_us = 1e6 / fps
            spread = max(first_stamps.values()) - min(first_stamps.values())
            if spread >= budget_us:
                out.append(Violation(
                    where, f"first frames misaligned by {spread:.0f}us, over "
                           f"the 1/fps budget of {budget_us:.0f}us"))
    return out


def unpinned_modalities(path) -> list:
    """Modalities present in the store whose channel count went unchecked.

    Not a violation — the contract has not pinned their frame representation —
    but the CLI prints it, so a clean PASS never quietly means "not looked at".
    """
    try:
        root = zarr.open_group(str(Path(path)), mode="r")
    except Exception:
        return []
    return sorted({f"{perspective}/{modality}"
                   for perspective in root.group_keys()
                   for modality in root[perspective].group_keys()
                   if modality in UNPINNED})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+",
                        help="cache directories and/or individual .zarr stores")
    args = parser.parse_args(argv)
    stores = []
    for raw in args.paths:
        path = Path(raw)
        # A store is itself a directory, so is_dir() alone globs *inside* it and
        # finds nothing -- the documented single-store form reported "No *.zarr
        # stores found" and exited 1. Only a non-store directory is a cache.
        if path.is_dir() and path.suffix != ".zarr":
            stores.extend(sorted(path.glob("*.zarr")))
        else:
            stores.append(path)
    if not stores:
        print("No *.zarr stores found.")
        return 1
    failed = 0
    for store in stores:
        violations = validate_store(store)
        if violations:
            failed += 1
            print(f"FAIL {store.name}")
            for violation in violations:
                print(f"  - {violation}")
        else:
            print(f"PASS {store.name}")
        for unpinned in unpinned_modalities(store):
            print(f"  ~ {unpinned}: channel count unchecked; the contract has "
                  f"not pinned this modality's frame representation yet")
    print(f"{len(stores) - failed}/{len(stores)} stores pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
