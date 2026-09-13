"""The evaluation of one recording and camera: its beats, readings and
heart rates, written beside the trace tables it read.

A reading is one blood-pressure determination: a non-overlapping stretch of
the combined trace, ``reading_seconds`` long, cut from the first covered
frame to the last, a trailing remainder shorter than half a reading
dropped. The length is the caller's (``scripts/eval.py --reading-seconds``):

* ISO 81060-2:2018 clause 6.2.4 b), p. 21 — the invasive reference reading
  averages the beat-by-beat values over at least 30 s (the default);
* ISO 81060-3:2022 clause 5.1.3 a) 1), p. 14 — the segment matches the
  device's minimum output period, typically 5 s to 10 s (A.2, p. 27);
* IEEE 1708-2025 clause 4.4.2, p. 24 — three 60 s recordings per test.

Per reading and signal the row carries the beat counts on both sides and
how many matched; for absolute-class signals the mean and SD over the
beats of each level (max / mean / min: systolic / MAP / diastolic for ABP),
the error of the means (prediction minus reference, the sign every
standard uses) and the ISO 81060-2 clause 6.2.5, p. 22, dead-band error
(zero inside the reference mean ± SD, else the distance to the nearer
limit); and the per-sample agreement of the combined prediction with the
label over the reading, the IEEE 1708 waveform metrics (equations (3) and
(4), p. 28). The prediction is the trace table's ``mean`` column: the
average of every strided window covering the frame.

The three files carry no dataset / participant / recording / perspective
columns: the folder's place in the records directory says where it sits,
so this module never needs to know.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.beats import BEAT_COLUMNS, LEVELS, analyse, beat_rows
from src.evaluation.rate import MIN_FRAMES, RATE_METRICS, reading_rates
from src.outputs import FLOAT_FORMAT
from src.signal_transforms import is_absolute, is_cardiac

BEATS_NAME, READINGS_NAME, RATES_NAME = "beats.csv", "readings.csv", "rates.csv"
FILES = (BEATS_NAME, READINGS_NAME, RATES_NAME)
WAVEFORM_METRICS = ("mad", "rmse", "r", "ccc")
READING_COLUMNS = (
    "signal", "reading", "t_start", "t_end", "reading_seconds",
    "n_ref_beats", "n_pred_beats", "n_matched",
    *(f"{side}_{s}_{stat}" for s in LEVELS for side in ("ref", "pred")
      for stat in ("mean", "sd")),
    *(f"err_{s}" for s in LEVELS),
    *(f"err_{s}_deadband" for s in LEVELS),
    *(f"waveform_{m}" for m in WAVEFORM_METRICS),
)
RATE_COLUMNS = ("reading", "source", *RATE_METRICS)
TRACE_COLUMNS = ["frame", "t", "label", "mean", "std", "n"]
_NAN = float("nan")


# ---------------------------------------------------------------------------
# Agreement between two sequences
# ---------------------------------------------------------------------------
def pearson(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return _NAN
    return float(np.corrcoef(a, b)[0, 1])


def ccc(a, b) -> float:
    """Lin's concordance: correlation penalised by disagreement in level."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    r = pearson(a, b)
    if not np.isfinite(r):
        return _NAN
    va, vb = a.var(ddof=1), b.var(ddof=1)
    denominator = va + vb + (a.mean() - b.mean()) ** 2
    return float(2 * r * np.sqrt(va * vb) / denominator) if denominator > 0 else _NAN


def _waveform(ref: np.ndarray, pred: np.ndarray) -> dict:
    error = pred - ref
    return {"waveform_mad": float(np.abs(error).mean()),
            "waveform_rmse": float(np.sqrt((error ** 2).mean())),
            "waveform_r": pearson(pred, ref), "waveform_ccc": ccc(pred, ref)}


# ---------------------------------------------------------------------------
# The trace tables and the readings they are cut into
# ---------------------------------------------------------------------------
def read_trace(folder: Path, sig: str) -> pd.DataFrame:
    """The fixed columns of one trace table (``src/outputs.py``)."""
    return pd.read_csv(Path(folder) / f"{sig}.csv", usecols=TRACE_COLUMNS)[TRACE_COLUMNS]


def reading_bounds(covered: np.ndarray, fs: float, reading_seconds: float) -> list:
    """``[(start, end), ...]`` sample slices of consecutive readings from the
    first covered frame to the last; a trailing remainder shorter than half
    a reading is dropped."""
    frames = np.flatnonzero(covered)
    if frames.size == 0:
        return []
    first, last = int(frames[0]), int(frames[-1]) + 1
    length = max(1, int(round(reading_seconds * fs)))
    bounds = []
    for start in range(first, last, length):
        end = min(start + length, last)
        if end - start >= length / 2:
            bounds.append((start, end))
    return bounds


def _filled(x: np.ndarray) -> np.ndarray:
    """NaN samples replaced by the finite mean, for the filters."""
    finite = np.isfinite(x)
    return np.where(finite, x, x[finite].mean())


def _errors(s: str, row: dict) -> dict:
    err = row[f"pred_{s}_mean"] - row[f"ref_{s}_mean"]
    sd = row[f"ref_{s}_sd"]
    if np.isfinite(err) and np.isfinite(sd):
        dead = 0.0 if abs(err) <= sd else err - np.sign(err) * sd
    else:
        dead = err
    return {f"err_{s}": err, f"err_{s}_deadband": dead}


def _beat_level_columns(ref_levels: np.ndarray, pred_levels: np.ndarray) -> dict:
    """Mean and SD over the beats of each level, each side, and the errors."""
    row = {}
    for k, s in enumerate(LEVELS):
        for side, levels in (("ref", ref_levels), ("pred", pred_levels)):
            values = levels[:, k]
            row[f"{side}_{s}_mean"] = float(values.mean()) if values.size else _NAN
            row[f"{side}_{s}_sd"] = float(values.std(ddof=1)) if values.size > 1 else _NAN
        row.update(_errors(s, row))
    return row


def _sample_level_columns(ref: np.ndarray, pred: np.ndarray) -> dict:
    """For an absolute signal without beats (SpO2): the level is the sample
    mean, its SD the sample SD, max and min the sample extremes."""
    row = {}
    for side, values in (("ref", ref), ("pred", pred)):
        row[f"{side}_max_mean"], row[f"{side}_max_sd"] = float(values.max()), _NAN
        row[f"{side}_mean_mean"] = float(values.mean())
        row[f"{side}_mean_sd"] = float(values.std(ddof=1)) if values.size > 1 else _NAN
        row[f"{side}_min_mean"], row[f"{side}_min_sd"] = float(values.min()), _NAN
    for s in LEVELS:
        row.update(_errors(s, row))
    return row


# ---------------------------------------------------------------------------
# One folder to its three files
# ---------------------------------------------------------------------------
def score_recording(folder, meta: dict, reading_seconds: float) -> dict:
    """Score every trace of one recording-and-camera folder; write and return
    ``{"beats", "readings", "rates"}``."""
    folder = Path(folder)
    fs, traces = float(meta["fs"]), [str(sig) for sig in meta["traces"]]
    tables = {sig: read_trace(folder, sig) for sig in traces}
    # The tables' own time axis, not the row index: under --limit-windows the
    # first covered frame need not be frame 0, so index / fs and the table's
    # "t" column disagree by the covered window's start offset. t_start,
    # t_end and t_ref are on the "t" axis so they line up with the trace
    # tables they sit beside.
    times = tables[traces[0]]["t"].to_numpy(dtype=np.float64)
    covered = np.zeros(len(tables[traces[0]]), dtype=bool)
    for table in tables.values():
        covered |= table["n"].to_numpy() > 0
    readings, beats, rates = [], [], []
    first_beat = {sig: 0 for sig in traces}
    for index, (start, end) in enumerate(reading_bounds(covered, fs, reading_seconds)):
        t0, cardiac = float(times[start]), {}
        t_end = float(times[end - 1]) + 1 / fs
        for sig, table in tables.items():
            label = table["label"].to_numpy(dtype=np.float64)[start:end]
            pred = table["mean"].to_numpy(dtype=np.float64)[start:end]
            row = {"signal": sig, "reading": index, "t_start": t0, "t_end": t_end,
                   "reading_seconds": reading_seconds}
            ok = np.isfinite(label) & np.isfinite(pred)
            if ok.sum() >= MIN_FRAMES:
                row.update(_waveform(label[ok], pred[ok]))
                if is_cardiac(sig):
                    label_f, pred_f = _filled(label), _filled(pred)
                    cardiac[sig] = (label_f, pred_f)
                    found = analyse(label_f, pred_f, fs, sig)
                    row.update({"n_ref_beats": found.ref_peaks.size,
                                "n_pred_beats": found.pred_peaks.size,
                                "n_matched": found.n_matched})
                    beats.append(beat_rows(found, fs, t0, sig, first_beat[sig]))
                    first_beat[sig] += found.ref_peaks.size
                    if is_absolute(sig):
                        row.update(_beat_level_columns(found.ref_levels, found.pred_levels))
                elif is_absolute(sig):
                    row.update(_sample_level_columns(label[ok], pred[ok]))
            readings.append(row)
        rates.extend({"reading": index, **r} for r in reading_rates(cardiac, fs))
    frames = {
        "beats": (pd.concat(beats, ignore_index=True) if beats
                  else pd.DataFrame(columns=list(BEAT_COLUMNS))),
        "readings": pd.DataFrame(readings, columns=list(READING_COLUMNS)),
        "rates": pd.DataFrame(rates, columns=list(RATE_COLUMNS)),
    }
    for name, key in ((BEATS_NAME, "beats"), (READINGS_NAME, "readings"), (RATES_NAME, "rates")):
        frames[key].to_csv(folder / name, index=False, float_format=FLOAT_FORMAT)
    return frames
