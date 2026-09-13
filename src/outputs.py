"""What inference leaves behind: one readable directory per participant.

``write_records`` turns the per-window records ``Trainer.test`` returns into::

    <out_dir>/
      meta.json                        what was run, on what, at what rate
      windows.csv                      one row per window: where it sits and
                                       which channels / traces it carried
      <recording>/<perspective>/<TRACE>.csv   one wide table per trace

Windows never cross a recording or a camera, so each (recording, perspective)
has its own time axis and its own tables. A trace table has one row per
frame the windows cover: ``frame``, ``t`` (seconds at the interface's
``FS``), ``label`` (physical units; blank where the trace is padded or no
window covers the frame), ``mean`` / ``std`` / ``n`` over the windows
covering that frame, then one ``w<start_frame>`` column per window holding
its prediction inside its span and blank outside. Overlapping windows sit
side by side, ``mean`` is the combined trace and ``std`` its repeatability.

Everything is plain CSV and JSON; nothing downstream needs torch to read it.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.interface import InterfaceConfig

RECORDS_DIR = "test_records"
META_NAME = "meta.json"
WINDOWS_NAME = "windows.csv"
FLOAT_FORMAT = "%.6g"
#: The columns every trace table starts with; the ``w<start_frame>`` columns follow.
FIXED_COLUMNS = ("frame", "t", "label", "mean", "std", "n")


def _array(value) -> np.ndarray:
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value, dtype=np.float64)


def _start(record) -> int:
    return int(record["metadata"]["start_frame"])


def window_rows(records, interface: InterfaceConfig) -> pd.DataFrame:
    """``windows.csv``: one row per record in the order given, with a presence
    flag per interface channel and trace."""
    rows = []
    for i, record in enumerate(records):
        meta, start = record["metadata"], _start(record)
        row = {"window": i, "dataset": str(meta["dataset"]),
               "recording": str(meta["recording"]),
               "participant": str(meta["participant"]),
               "perspective": str(meta["perspective"]),
               "start_frame": start, "end_frame": start + interface.window_frames}
        row.update({f"channel_{ch}": bool(record["channel_mask"][ch])
                    for ch in interface.CHANNELS})
        row.update({f"label_{sig}": bool(record["label_mask"][sig])
                    for sig in interface.TRACES})
        rows.append(row)
    return pd.DataFrame(rows)


def trace_table(records, sig: str, fs: float) -> pd.DataFrame:
    """The wide table of one trace over one (recording, perspective)'s
    windows, which must be sorted by ``start_frame``."""
    starts = [_start(r) for r in records]
    length = _array(records[0]["predictions"][sig]).size
    first = starts[0]
    frames = np.arange(first, starts[-1] + length)
    label = np.full(frames.size, np.nan)
    columns = {}
    for record, start in zip(records, starts):
        lo = start - first
        column = np.full(frames.size, np.nan)
        column[lo:lo + length] = _array(record["predictions"][sig])
        columns[f"w{start}"] = column
        if bool(record["label_mask"][sig]):
            # Overlapping windows carry the same label; the first to cover
            # a frame writes it.
            span = label[lo:lo + length]
            missing = np.isnan(span)
            span[missing] = _array(record["labels"][sig])[missing]
    stack = np.stack(list(columns.values()), axis=1)     # (frames, windows)
    covered = ~np.isnan(stack)
    n = covered.sum(axis=1)
    # By hand rather than nanmean/nanstd: a frame no window covers (possible
    # after --limit-windows) is a blank, not a RuntimeWarning.
    mean = np.where(n > 0, np.nansum(stack, axis=1) / np.maximum(n, 1), np.nan)
    deviation = np.where(covered, stack - mean[:, None], 0.0) ** 2
    std = np.where(n > 1, np.sqrt(deviation.sum(axis=1) / np.maximum(n - 1, 1)), np.nan)
    return pd.DataFrame({"frame": frames, "t": frames / fs, "label": label,
                         "mean": mean, "std": std, "n": n, **columns})


def write_records(records, out_dir, interface: InterfaceConfig, meta: dict) -> Path:
    """Write the directory described in the module docstring; returns it.

    ``meta`` is what the caller knows and the records do not (the dataset and
    participant, the checkpoint's run directory, the command, the git state);
    the interface's rate, window, channels and traces are added here.
    """
    if not records:
        raise ValueError("no records to write")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    window_rows(records, interface).to_csv(out_dir / WINDOWS_NAME, index=False)

    groups: dict[tuple, list] = {}
    for record in records:
        m = record["metadata"]
        groups.setdefault((str(m["recording"]), str(m["perspective"])), []).append(record)
    for (recording, perspective), group in groups.items():
        group.sort(key=_start)          # DDP gathers shards in rank order
        folder = out_dir / recording / perspective
        folder.mkdir(parents=True, exist_ok=True)
        for sig in interface.TRACES:
            trace_table(group, sig, interface.FS).to_csv(
                folder / f"{sig}.csv", index=False, float_format=FLOAT_FORMAT)

    payload = {
        "fs": interface.FS,
        "window_frames": interface.window_frames,
        "stride_frames": interface.stride_frames,
        "channels": list(interface.CHANNELS),
        "traces": list(interface.TRACES),
        "label_preprocessing": dict(interface.LABEL_PREPROCESSING),
        **meta,
        "n_windows": len(records),
    }
    with open(out_dir / META_NAME, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return out_dir
