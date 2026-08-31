"""One window record, from either source, already in physical units.

``MultiSignalTrainer`` holds its scored windows in memory and also writes them
to ``*_outputs.pickle``. A LOSO sweep produces one pickle per fold. All three
arrive here and leave as the same ``RunRecords``, with the normalisation
already inverted — nothing downstream of this module sees normalised space.
"""

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import torch

from dataset.data_loader.label_transforms import INVERSES


@dataclass(frozen=True)
class WindowRecord:
    """One scored window of one signal, in that signal's physical unit."""

    signal: str
    recording_id: str
    camera_id: str
    start_frame: int
    prediction: np.ndarray
    label: np.ndarray
    attrs: dict


@dataclass(frozen=True)
class RunRecords:
    """Every window of a run (or of a whole sweep), plus what describes them."""

    windows: list
    fs: float
    traces: tuple
    label_norms: dict

    def signals(self) -> list:
        return sorted({window.signal for window in self.windows})


def to_physical(values, stats, mode) -> np.ndarray:
    """Invert one window's normalisation with the stats that produced it.

    Exact rather than approximate: the stats ride in the record. For a ``raw``
    signal the inverse is the identity and the numbers were already mmHg.
    """
    inverse = INVERSES[mode]
    tensor_stats = {key: torch.tensor(float(value)) for key, value in stats.items()}
    tensor = torch.as_tensor(np.asarray(values), dtype=torch.float32)
    return inverse(tensor, tensor_stats).numpy().astype(np.float64)


def from_saved(raw_windows, *, fs, traces, label_norms) -> RunRecords:
    """Build ``RunRecords`` from the trainer's own per-window dicts."""
    windows = []
    for raw in raw_windows:
        mode = label_norms[raw["signal"]]
        windows.append(WindowRecord(
            signal=raw["signal"],
            recording_id=str(raw["recording_id"]),
            camera_id=str(raw["camera_id"]),
            start_frame=int(raw["start_frame"]),
            prediction=to_physical(raw["prediction"], raw["label_stats"], mode),
            label=to_physical(raw["label"], raw["label_stats"], mode),
            attrs=dict(raw.get("attrs") or {}),
        ))
    return RunRecords(windows=windows, fs=float(fs), traces=tuple(traces),
                      label_norms=dict(label_norms))


def _pickle_paths(target) -> list:
    """A pickle, a directory to search, or an iterable of either."""
    if isinstance(target, (str, Path)):
        path = Path(target)
        if path.is_file():
            return [path]
        found = sorted(path.rglob("*_outputs.pickle"))
        if not found:
            raise FileNotFoundError(f"No *_outputs.pickle under {path}")
        return found
    return [p for item in target for p in _pickle_paths(item)]


def load(target) -> RunRecords:
    """Load one run or a whole sweep into a single ``RunRecords``.

    Pooling folds is the only way to reach the participant and cohort levels:
    a LOSO fold has exactly one test participant. Folds must agree on the
    sampling rate and on every shared signal's normalisation, or the pooled
    numbers would silently mix units.
    """
    windows, fs, traces, label_norms = [], None, [], {}
    for path in _pickle_paths(target):
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if fs is None:
            fs = float(payload["fs"])
        elif float(payload["fs"]) != fs:
            raise ValueError(
                f"{path} was scored at {payload['fs']} Hz but earlier files at "
                f"{fs} Hz; pooling them would mix time bases")
        for signal, mode in payload["label_norms"].items():
            if label_norms.setdefault(signal, mode) != mode:
                raise ValueError(
                    f"{path} normalised {signal} as {mode!r}, earlier files as "
                    f"{label_norms[signal]!r}; pooling would mix units")
        for trace in payload["traces"]:
            if trace not in traces:
                traces.append(trace)
        windows.extend(payload["windows"])
    return from_saved(windows, fs=fs, traces=tuple(traces), label_norms=label_norms)
