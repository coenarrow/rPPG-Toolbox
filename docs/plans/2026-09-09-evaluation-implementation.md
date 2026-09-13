# Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** replace the dead per-window evaluation with a three-layer
`src/evaluation/` package and a real `scripts/eval.py` that score what
`scripts/infer.py` writes, compute the ISO 81060-2, ISO 81060-3 and IEEE
1708 statistics, and write one PDF report.

**Architecture:** layer one (`recording.py`, `beats.py`, `rate.py`) turns
one recording-and-camera folder of trace tables into `beats.csv`,
`readings.csv` and `rates.csv` beside them. Layer two (`aggregate.py`)
globs those files across any number of records directories and derives the
summary, criteria and coverage tables from them alone. Layer three
(`plots.py`, `report.py`) draws seaborn figures and writes the PdfPages
report and the text digest. Nothing in the package imports torch.

**Tech Stack:** Python 3 via `uv run`, numpy, pandas, scipy
(`find_peaks`, `butter`, `filtfilt`, `periodogram`), matplotlib (Agg,
`PdfPages`), seaborn. Tests with pytest.

**Spec:** `docs/plans/2026-09-09-evaluation-design.md`. Read it first; the
plan argues from it. The standards themselves are under `standards/`; the
clause and page numbers in the code comments below come from them.

## Global Constraints

Copied from `CLAUDE.md` and the spec. Every task's requirements include
these.

1. **Everything shared is written once.** One beat detector, one reading
   segmenter, one agreement function, one plot set. A second implementation
   of anything is a bug in the first one's design.
2. **Legacy code is deleted, not adapted.** No shim around the old
   `test_records.pt` path. `src/evaluation/evaluate.py`,
   `src/evaluation/records.py` and `tests/test_evaluate.py` are deleted in
   Task 8, not rewritten.
3. **Tests are a cost.** This plan adds exactly: two dozen-line unit tests
   in a new `tests/test_evaluation.py` (the beat detector on a sine, the
   ISO 81060-2 Table 1 lookup), one eval step in the existing chain test in
   `tests/test_scripts.py`, and the moved `to_physical` test. Nothing else.
   Verification for every other step is the run command shown in that step.
4. **Dependencies go through `uv add`, never pip.** `pyproject.toml` and
   `uv.lock` are the source of truth.
5. **No torch in `src/evaluation/`.** `grep -rn torch src/evaluation` must
   print nothing when the plan is done.
6. **Run pytest as `uv run pytest -p no:faulthandler ...`** on Windows; a
   harmless `0xc0000139` DLL message appears otherwise when `src.models` is
   imported.
7. **Do not commit.** The working tree's index already holds the user's
   uncommitted train/infer refactor (many staged files). A plain
   `git commit` would sweep it in. The user commits; each task ends with a
   verification step instead of a commit step.
8. **The repo has a real run to check against:**
   `runs/BIGSMALL_PURE.01_202609091150/test_records/` holds six recordings
   (`01-01` to `01-06`), one camera (`1`), one trace (`PPG`), 25 fps, 180
   frame windows at stride 25, 396 windows. PURE has no ABP, so the BP
   criteria come out empty on it; the synthetic fixture in the chain test
   carries ABP and CVP ramps (no beats) and only checks wiring.
9. **Column-name conventions** used by every task: levels are `max`,
   `mean`, `min` (registry generic names; reports render them through
   `beat_labels()`); errors are prediction minus reference; tag columns are
   `dataset`, `participant`, `recording`, `perspective`; MAPD is called
   `mapd` for levels and heart rate alike.

## File map

| File | Task | Responsibility |
| --- | --- | --- |
| `pyproject.toml`, `uv.lock` | 1 | seaborn becomes a runtime dependency |
| `src/signal_transforms.py` | 1 | `beat` entry per cardiac signal, `beat_config()` |
| `src/evaluation/beats.py` (new) | 2 | detect, refine, level, match beats; `Beats`, `beat_rows` |
| `tests/test_evaluation.py` (new) | 2, 5 | the two unit tests |
| `src/evaluation/rate.py` | 3 | `reading_rates()` replaces `window_rates()` |
| `src/evaluation/recording.py` (new) | 4 | layer one: folder to three CSVs |
| `src/evaluation/aggregate.py` (new) | 5 | layer two: pool, summary, criteria, coverage, changes |
| `src/evaluation/plots.py` (rewrite) | 6 | every seaborn figure |
| `src/evaluation/report.py` (new) | 7 | PdfPages report and digest |
| `scripts/eval.py` (rewrite) | 8 | the entry point |
| `src/evaluation/__init__.py`, deletions, `tests/test_to_physical.py`, `tests/test_scripts.py` | 8 | wiring, cleanup, smoke test |
| `README.md`, `docs/adding_a_model.md`, `docs/evaluation.md` (new), `docs/plans/2026-09-08-model-migrations.md` | 9 | docs |

---

### Task 1: seaborn as a runtime dependency, and the registry's beat config

**Files:**
- Modify: `pyproject.toml` (dev group line `"seaborn>=0.13.2",     # prototype notebooks only`)
- Modify: `src/signal_transforms.py:58-70` (the `SIGNALS` dict) and after `beat_labels()` at line 123

**Interfaces:**
- Produces: `beat_config(sig) -> dict` with keys `polarity` (int, +1 or -1) and `prominence` (float fraction); `KeyError` for a non-cardiac signal.

- [ ] **Step 1: Move seaborn to the runtime dependencies**

Run from the repo root:

```bash
uv remove --group dev seaborn
uv add "seaborn>=0.13.2"
```

Expected: `pyproject.toml` lists `"seaborn>=0.13.2",` under `dependencies`
and no longer under `dev`; `uv.lock` updated. If `uv add` complains about
the `mamba-ssm` or `triton` overrides, rerun with `--frozen` removed and
report the error verbatim; do not edit `uv.lock` by hand.

- [ ] **Step 2: Add the beat entry to every cardiac signal**

In `src/signal_transforms.py`, replace the four cardiac entries of
`SIGNALS` so they read:

```python
    "PPG":  {"class": SHAPE,    "unit": "a.u.", "prior": 0.0,  "cardiac": True,
             "beat": {"polarity": 1, "prominence": 0.3}},
    "ECG":  {"class": SHAPE,    "unit": "uV",   "prior": 0.0,  "cardiac": True,
             "beat": {"polarity": 1, "prominence": 0.3}},
    "ABP":  {"class": ABSOLUTE, "unit": "mmHg", "prior": 90.0, "cardiac": True,
             "beat": {"polarity": 1, "prominence": 0.3},
             "beat_labels": {"max": "systolic", "mean": "MAP", "min": "diastolic"}},
    # CVP has no systole: its waveform is a/c/v waves, and the quantity that
    # matters clinically is the mean. Same machinery as ABP, other words.
    # Its beats are detected with the same detector and a lower prominence;
    # the recall / precision columns of readings.csv say whether that works.
    "CVP":  {"class": ABSOLUTE, "unit": "mmHg", "prior": 8.0,  "cardiac": True,
             "beat": {"polarity": 1, "prominence": 0.2},
             "beat_labels": {"max": "peak", "mean": "mean", "min": "trough"}},
```

Extend the comment block above `SIGNALS` (the `#:` lines) with one line:
`#: ``beat`` is how the detector reads the trace (``src/evaluation/beats.py``).`

- [ ] **Step 3: Add the accessor after `beat_labels()`**

```python
def beat_config(sig) -> dict:
    """How the beat detector reads this cardiac signal: ``polarity`` (+1 when
    a beat is a peak of the trace, -1 a trough) and ``prominence`` (the
    fraction of the cleaned trace's range a candidate must stand out by).
    KeyError for a signal that does not beat."""
    entry = SIGNALS[canonical_signal(sig)]
    if not entry["cardiac"]:
        raise KeyError(f"{sig} is not cardiac; it has no beats")
    return dict(entry["beat"])
```

- [ ] **Step 4: Verify**

```bash
uv run python -c "import seaborn; from src.signal_transforms import beat_config; print(seaborn.__version__, beat_config('ABP'), beat_config('bvp'))"
uv run pytest -p no:faulthandler tests/test_signal_transforms.py -q
```

Expected: a version, `{'polarity': 1, 'prominence': 0.3}` twice, and the
existing registry tests pass.

---

### Task 2: `beats.py`, the one beat detector

**Files:**
- Create: `src/evaluation/beats.py`
- Create: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `clean(trace, fs)`, `BAND`, `MIN_FRAMES` from `src/evaluation/rate.py` (exist); `beat_config`, `is_absolute` from `src/signal_transforms.py`.
- Produces:
  - `LEVELS = ("max", "mean", "min")`
  - `BEAT_COLUMNS` tuple
  - `detect_beats(trace, fs, sig) -> np.ndarray` of int sample indices
  - `beat_levels(trace, peaks, fs, polarity) -> np.ndarray` shape `(n_beats, 3)`
  - `match_beats(ref_peaks, pred_peaks, fs) -> np.ndarray` shape `(n_ref,)`, index into `pred_peaks` or -1
  - `Beats` dataclass with `ref_peaks`, `pred_peaks`, `ref_levels`, `pred_levels`, `match`, property `n_matched`
  - `analyse(label, pred, fs, sig) -> Beats`
  - `beat_rows(beats, fs, t0, sig, first_beat) -> pd.DataFrame` with `BEAT_COLUMNS`

- [ ] **Step 1: Write the module**

```python
"""One beat detector for every cardiac trace, on the label and on the prediction.

The clinical standards score readings of systolic, diastolic and mean
pressure, one per cardiac cycle before averaging (ISO 81060-2:2018 clause
6.2.4, p. 21-22; ISO 81060-3:2022 clause 5.1.3, p. 14; IEEE 1708-2025 clause
3.1, p. 16), so the beat is the atomic unit of the evaluation. This module
finds the beats of one trace, reads each beat's max / mean / min off the raw
waveform, and pairs the predicted beats with the reference ones so misses
and spurious beats are counted, not hidden.

Detection: the trace is cleaned as the heart-rate estimator cleans it
(detrend, zero-phase bandpass to the heart-rate band), multiplied by the
signal's polarity from the registry, and ``scipy.signal.find_peaks`` runs
with the minimum distance between beats set by the top of the band and the
prominence a registry fraction of the cleaned range. Each candidate is then
moved to the extremum of the raw trace within a quarter of the median beat
interval, so systolic is read off the real waveform, not the filtered one.

Levels: a beat spans trough to trough around its peak (the troughs being the
minima of the polarity-signed trace between neighbouring peaks). Over that
span ``max`` is the peak, ``min`` the lower trough and ``mean`` the area
under the curve divided by the duration, which is the MAP definition in ISO
81060-2 clause 6.2.4 e), p. 22.

Matching: each predicted beat goes to the nearest reference beat within 40
percent of the median reference interval, closest pairs first, one to one.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.evaluation.rate import BAND, MIN_FRAMES, clean
from src.signal_transforms import beat_config, is_absolute

LEVELS = ("max", "mean", "min")
#: A predicted beat matches a reference beat within this fraction of the
#: median reference inter-beat interval.
MATCH_FRACTION = 0.4
#: A candidate peak is moved to the raw extremum within this fraction of the
#: median inter-beat interval.
REFINE_FRACTION = 0.25
BEAT_COLUMNS = ("signal", "beat", "t_ref", "t_pred",
                *(f"{side}_{s}" for side in ("ref", "pred") for s in LEVELS))
_NAN = float("nan")


def _median_interval(peaks: np.ndarray, fs: float) -> float:
    """Samples between beats; one second when there are too few beats to tell."""
    return float(np.median(np.diff(peaks))) if peaks.size > 1 else float(fs)


def detect_beats(trace, fs: float, sig: str) -> np.ndarray:
    """Sample indices of the beats of one finite trace, sorted, unique."""
    trace = np.asarray(trace, dtype=np.float64)
    if trace.size < MIN_FRAMES or not np.all(np.isfinite(trace)):
        raise ValueError(
            f"detect_beats needs a finite trace of at least {MIN_FRAMES} samples")
    config = beat_config(sig)
    signed = trace * config["polarity"]
    cleaned = clean(signed, fs)
    span = float(np.ptp(cleaned))
    if span <= 0:
        return np.array([], dtype=int)
    candidates, _ = find_peaks(cleaned, distance=max(1, int(fs / BAND[1])),
                               prominence=config["prominence"] * span)
    if candidates.size == 0:
        return candidates.astype(int)
    radius = max(1, int(round(REFINE_FRACTION * _median_interval(candidates, fs))))
    refined = []
    for c in candidates:
        lo, hi = max(0, c - radius), min(signed.size, c + radius + 1)
        refined.append(lo + int(np.argmax(signed[lo:hi])))
    return np.unique(np.asarray(refined, dtype=int))


def beat_levels(trace, peaks: np.ndarray, fs: float, polarity: int) -> np.ndarray:
    """``(n_beats, 3)`` of max, mean, min over each beat's trough-to-trough
    span, read off the raw trace. Empty when there are no beats."""
    trace = np.asarray(trace, dtype=np.float64)
    if peaks.size == 0:
        return np.empty((0, len(LEVELS)))
    signed = trace * polarity
    interval = int(round(_median_interval(peaks, fs)))
    starts = np.concatenate(([max(0, peaks[0] - interval)], peaks[:-1]))
    ends = np.concatenate((peaks[1:], [min(trace.size - 1, peaks[-1] + interval)]))
    rows = []
    for start, peak, end in zip(starts, peaks, ends):
        lo = start + int(np.argmin(signed[start:peak + 1]))
        hi = peak + int(np.argmin(signed[peak:end + 1]))
        span = trace[lo:hi + 1]
        rows.append((float(span.max()), float(span.mean()), float(span.min())))
    return np.asarray(rows, dtype=np.float64)


def match_beats(ref_peaks: np.ndarray, pred_peaks: np.ndarray, fs: float) -> np.ndarray:
    """For every reference beat the index of its predicted beat, or -1."""
    match = np.full(ref_peaks.size, -1, dtype=int)
    if ref_peaks.size == 0 or pred_peaks.size == 0:
        return match
    tolerance = MATCH_FRACTION * _median_interval(ref_peaks, fs)
    gaps = np.abs(ref_peaks[:, None] - pred_peaks[None, :]).astype(np.float64)
    pairs = np.argwhere(gaps <= tolerance)
    if pairs.size == 0:
        return match
    order = np.argsort(gaps[pairs[:, 0], pairs[:, 1]], kind="stable")
    used = np.zeros(pred_peaks.size, dtype=bool)
    for i, j in pairs[order]:
        if match[i] < 0 and not used[j]:
            match[i], used[j] = j, True
    return match


@dataclass
class Beats:
    """The beats of one reading: reference and predicted, their levels, and
    which predicted beat each reference beat matched (-1 for a miss)."""
    ref_peaks: np.ndarray
    pred_peaks: np.ndarray
    ref_levels: np.ndarray
    pred_levels: np.ndarray
    match: np.ndarray

    @property
    def n_matched(self) -> int:
        return int((self.match >= 0).sum())


def analyse(label, pred, fs: float, sig: str) -> Beats:
    """Detect, level and match the beats of one finite label / prediction pair."""
    polarity = beat_config(sig)["polarity"]
    ref_peaks, pred_peaks = detect_beats(label, fs, sig), detect_beats(pred, fs, sig)
    return Beats(ref_peaks, pred_peaks,
                 beat_levels(label, ref_peaks, fs, polarity),
                 beat_levels(pred, pred_peaks, fs, polarity),
                 match_beats(ref_peaks, pred_peaks, fs))


def beat_rows(beats: Beats, fs: float, t0: float, sig: str, first_beat: int = 0) -> pd.DataFrame:
    """One row per reference beat: its time, its matched predicted beat's time
    (blank on a miss) and, for absolute-class signals, both beats' levels.
    ``t0`` is the reading's start in seconds; ``first_beat`` numbers the rows
    on from the recording's earlier readings."""
    levels = is_absolute(sig)
    rows = []
    for i, (peak, j) in enumerate(zip(beats.ref_peaks, beats.match)):
        row = {"signal": sig, "beat": first_beat + i, "t_ref": t0 + peak / fs,
               "t_pred": t0 + beats.pred_peaks[j] / fs if j >= 0 else _NAN}
        for k, s in enumerate(LEVELS):
            row[f"ref_{s}"] = float(beats.ref_levels[i, k]) if levels else _NAN
            row[f"pred_{s}"] = (float(beats.pred_levels[j, k])
                                if levels and j >= 0 else _NAN)
        rows.append(row)
    return pd.DataFrame(rows, columns=list(BEAT_COLUMNS))
```

- [ ] **Step 2: Write the unit test**

Create `tests/test_evaluation.py`:

```python
"""Two pure functions of the evaluation: the beat detector and the ISO
81060-2 Table 1 lookup. The chain test in test_scripts.py covers the rest."""
import numpy as np

from src.evaluation.beats import beat_levels, detect_beats, match_beats


def test_beats_of_a_sine_are_found_levelled_and_matched():
    fs, hz = 50.0, 1.2                       # 72 bpm
    t = np.arange(20 * fs) / fs
    abp = 100 + 20 * np.sin(2 * np.pi * hz * t)
    peaks = detect_beats(abp, fs, "ABP")
    assert abs(peaks.size - 24) <= 1         # 20 s at 1.2 Hz, an edge beat may go
    np.testing.assert_allclose(np.diff(peaks), fs / hz, atol=2)
    levels = beat_levels(abp, peaks, fs, 1)
    np.testing.assert_allclose(levels[:, 0], 120, atol=0.5)     # systolic
    np.testing.assert_allclose(levels[1:-1, 1], 100, atol=1)    # MAP
    np.testing.assert_allclose(levels[:, 2], 80, atol=0.5)      # diastolic
    match = match_beats(peaks, peaks + 3, fs)
    assert (match == np.arange(peaks.size)).all()
    assert (match_beats(peaks, np.array([], dtype=int), fs) == -1).all()
```

- [ ] **Step 3: Run it**

```bash
uv run pytest -p no:faulthandler tests/test_evaluation.py -q
```

Expected: 1 passed. If the beat count is off by more than one, the
prominence fraction is too high for the filtered edges; print `peaks` and
check the first and last are the ones missing before changing anything.

---

### Task 3: `rate.py` estimates per reading, not per window

**Files:**
- Modify: `src/evaluation/rate.py` (module docstring lines 1-30; delete `_array`, `cardiac_traces`, `window_rates` at lines 113-158; add `reading_rates`)

**Interfaces:**
- Produces: `reading_rates(traces: dict[str, tuple[np.ndarray, np.ndarray]], fs) -> list[dict]`, each dict with keys `source`, `ref_hr`, `pred_hr`, `err_hr`, `snr`, `macc`. `RATE_METRICS`, `FUSED`, `MEDIAN`, `BAND`, `MIN_FRAMES`, `clean` unchanged.

- [ ] **Step 1: Replace the tail of the module**

Delete everything from the comment `# One window to its rows` to the end
of the file (the `_array`, `cardiac_traces` and `window_rates` functions)
and put this in its place:

```python
# ---------------------------------------------------------------------------
# One reading to its rows
# ---------------------------------------------------------------------------
def reading_rates(traces: dict, fs: float) -> list:
    """``[{source, ref_hr, pred_hr, err_hr, snr, macc}, ...]`` for one reading,
    given ``{signal: (label, prediction)}`` over the cardiac traces it
    carries, both finite: one row per trace, then ``FUSED`` and ``MEDIAN``
    when there are two or more to combine. Empty for a reading too short
    to filter."""
    rows, ref_powers, pred_powers = [], [], []
    freqs = None
    for sig, (ref, pred) in traces.items():
        ref = np.asarray(ref, dtype=np.float64)
        pred = np.asarray(pred, dtype=np.float64)
        if ref.size < MIN_FRAMES:
            return []
        ref, pred = clean(ref, fs), clean(pred, fs)
        freqs, ref_power = spectrum(ref, fs)
        _, pred_power = spectrum(pred, fs)
        ref_hr, pred_hr = rate_of(freqs, ref_power), rate_of(freqs, pred_power)
        rows.append({"source": sig, "ref_hr": ref_hr, "pred_hr": pred_hr,
                     "err_hr": pred_hr - ref_hr, "snr": snr(freqs, pred_power, ref_hr),
                     "macc": _compute_macc(pred, ref)})
        ref_powers.append(ref_power)
        pred_powers.append(pred_power)
    if len(rows) < 2:
        return rows

    ref_fused, pred_fused = fuse(freqs, ref_powers), fuse(freqs, pred_powers)
    ref_hr, pred_hr = rate_of(freqs, ref_fused), rate_of(freqs, pred_fused)
    rows.append({"source": FUSED, "ref_hr": ref_hr, "pred_hr": pred_hr,
                 "err_hr": pred_hr - ref_hr, "snr": snr(freqs, pred_fused, ref_hr),
                 "macc": _NAN})
    ref_hr = float(np.median([row["ref_hr"] for row in rows[:-1]]))
    pred_hr = float(np.median([row["pred_hr"] for row in rows[:-1]]))
    rows.append({"source": MEDIAN, "ref_hr": ref_hr, "pred_hr": pred_hr,
                 "err_hr": pred_hr - ref_hr, "snr": _NAN, "macc": _NAN})
    return rows
```

Also remove the now-unused `from src.signal_transforms import is_cardiac`
import at the top, and in the module docstring change "Per window, for
every trace" to "Per reading, for every trace" and the closing sentence
"the records are already in physical units" to "the trace tables are
already in physical units".

- [ ] **Step 2: Verify**

```bash
uv run python -c "
import numpy as np
from src.evaluation.rate import reading_rates
t = np.arange(750) / 25.0
abp = 100 + 20 * np.sin(2 * np.pi * 1.2 * t); ppg = np.sin(2 * np.pi * 1.2 * t + 1)
rows = reading_rates({'ABP': (abp, abp + np.random.default_rng(0).normal(0, 2, 750)), 'PPG': (ppg, ppg)}, 25.0)
print([(r['source'], round(r['ref_hr'], 1), round(r['pred_hr'], 1)) for r in rows])"
grep -n "window_rates\|cardiac_traces\|is_cardiac" src/evaluation/rate.py
```

Expected: four rows `ABP`, `PPG`, `FUSED`, `MEDIAN`, every rate within 3
bpm of 72; the grep prints nothing. `src/evaluation/evaluate.py` still
imports `window_rates` and is now broken; it is deleted in Task 8.

---

### Task 4: `recording.py`, layer one

**Files:**
- Create: `src/evaluation/recording.py`

**Interfaces:**
- Consumes: Task 2's `analyse`, `beat_rows`, `BEAT_COLUMNS`, `LEVELS`; Task 3's `reading_rates`, `RATE_METRICS`, `MIN_FRAMES`; `FLOAT_FORMAT` from `src/outputs.py`; `is_absolute`, `is_cardiac` from the registry.
- Produces:
  - `BEATS_NAME = "beats.csv"`, `READINGS_NAME = "readings.csv"`, `RATES_NAME = "rates.csv"`, `FILES`
  - `WAVEFORM_METRICS = ("mad", "rmse", "r", "ccc")`, `READING_COLUMNS`, `RATE_COLUMNS`
  - `pearson(a, b) -> float`, `ccc(a, b) -> float`
  - `read_trace(folder, sig) -> pd.DataFrame` with columns `frame, t, label, mean, std, n`
  - `reading_bounds(covered, fs, reading_seconds) -> list[tuple[int, int]]`
  - `score_recording(folder, meta, reading_seconds) -> dict` with keys `beats`, `readings`, `rates` (DataFrames), having written the three CSVs into `folder`

- [ ] **Step 1: Write the module**

```python
"""Layer one of the evaluation: one recording and camera to its beats,
readings and heart rates, written beside the trace tables it read.

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
columns; layer two (``aggregate.py``) adds those from the folder's place
in the records directory, so this layer never needs to know where it sits.
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
    "signal", "reading", "t_start", "t_end",
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
    covered = np.zeros(len(tables[traces[0]]), dtype=bool)
    for table in tables.values():
        covered |= table["n"].to_numpy() > 0
    readings, beats, rates = [], [], []
    first_beat = {sig: 0 for sig in traces}
    for index, (start, end) in enumerate(reading_bounds(covered, fs, reading_seconds)):
        t0, cardiac = start / fs, {}
        for sig, table in tables.items():
            label = table["label"].to_numpy(dtype=np.float64)[start:end]
            pred = table["mean"].to_numpy(dtype=np.float64)[start:end]
            row = {"signal": sig, "reading": index, "t_start": t0, "t_end": end / fs}
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
```

- [ ] **Step 2: Run it on the real BigSmall run**

```bash
uv run python -c "
import json
from pathlib import Path
from src.evaluation.recording import score_recording
d = Path('runs/BIGSMALL_PURE.01_202609091150/test_records')
meta = json.load(open(d / 'meta.json'))
out = score_recording(d / '01-01' / '1', meta, 30.0)
print(out['readings'][['reading', 't_start', 't_end', 'n_ref_beats', 'n_pred_beats', 'n_matched', 'waveform_r']])
print(out['rates'])
print(out['beats'].head())"
ls runs/BIGSMALL_PURE.01_202609091150/test_records/01-01/1/
```

Expected: two PPG readings (the recording covers about 67 s at 25 fps:
30 s, 30 s, a 7 s remainder dropped), `n_ref_beats` around 30 to 45 each
(PURE resting heart rates), `n_matched` no larger than either count, one
`rates` row per reading with `source == "PPG"` and `ref_hr` between 45 and
100, `beats.csv`, `readings.csv` and `rates.csv` now listed in the folder.
Level columns are blank because PPG is shape-class. Delete the three files
afterwards so Task 5's staleness check is exercised from scratch:

```bash
rm runs/BIGSMALL_PURE.01_202609091150/test_records/01-01/1/{beats,readings,rates}.csv
```

---

### Task 5: `aggregate.py`, layer two

**Files:**
- Create: `src/evaluation/aggregate.py`
- Modify: `tests/test_evaluation.py` (append one test)

**Interfaces:**
- Consumes: Task 4's `FILES`, `READINGS_NAME`, `BEATS_NAME`, `RATES_NAME`, `WAVEFORM_METRICS`, `pearson`, `ccc`, `score_recording`; Task 2's `LEVELS`; `META_NAME`, `RECORDS_DIR` from `src/outputs.py`; `beat_labels`, `is_absolute` from the registry.
- Produces:
  - `TAGS`, `GROUP_COLUMNS` (dict grouping name to column), `SUMMARY_COLUMNS`, `CRITERIA_COLUMNS`, `COVERAGE_COLUMNS`, `CHANGE_COLUMNS`, `HR`, `WAVEFORM`, `BP_SIGNAL = "ABP"`
  - `find_records_dirs(paths) -> list[Path]`
  - `Pool` dataclass: `readings`, `beats`, `rates` (tagged DataFrames), `metas` (list of dict), `folders` (list of `(Path, meta, tags)`)
  - `load_pool(records_dirs, reading_seconds) -> Pool`
  - `participant_categories(readings) -> dict[str, str]`
  - `with_groups(frame, categories) -> pd.DataFrame` adding columns `all`, `recording_camera`, `bp_category`
  - `baseline_change(readings, s) -> pd.Series` (reference level minus the participant's first reading, aligned to `readings.index`)
  - `agreement(ref, pred) -> dict`
  - `summary_table(readings, rates, categories) -> pd.DataFrame`
  - `iso2_table1(mean_error) -> float`, `corrected_sd(errors, subjects) -> dict`, `type_t_errors(part, column) -> pd.Series`, `change_events(readings, change_seconds) -> pd.DataFrame`, `ieee_grade(mad, bias) -> str`, `ieee_waveform_grade(mad, r) -> str`
  - `criteria_table(readings, categories, change_seconds) -> tuple[pd.DataFrame, pd.DataFrame]` (criteria, change events)
  - `coverage_table(readings) -> pd.DataFrame`

- [ ] **Step 1: Write the module**

```python
"""Layer two of the evaluation: many records directories to one summary,
the clinical standards' criteria, and the coverage tables.

Everything here is derived from the files layer one (``recording.py``)
wrote beside each recording's trace tables, tagged with where they sit:
``dataset`` and ``participant`` from the records directory's ``meta.json``,
``recording`` and ``perspective`` from the folder names. A layer-one file
is (re)made only when missing or cut at a different reading length, so
pooling a LOSO sweep is reading small CSVs. Nothing here reads a trace.

Groupings: ``all``, ``dataset``, ``participant``, ``recording`` (recording
and camera together) and ``bp_category`` (the participant's, from their
mean reference systolic / diastolic ABP readings, binned by IEEE 1708-2025
Table 3, p. 22).

The standards' criteria are computed on ABP only (``BP_SIGNAL``); the
summary metrics on every signal. Counts the standards fix and we cannot
(subjects, readings per subject, change events per subject) are reported
as achieved values with ``pass`` left blank.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.beats import LEVELS
from src.evaluation.recording import (
    BEATS_NAME, FILES, RATES_NAME, READINGS_NAME, WAVEFORM_METRICS, ccc,
    pearson, score_recording,
)
from src.outputs import META_NAME, RECORDS_DIR
from src.signal_transforms import beat_labels, is_absolute

TAGS = ("dataset", "participant", "recording", "perspective")
#: Grouping name -> the column ``with_groups`` groups it by.
GROUP_COLUMNS = {"all": "all", "dataset": "dataset", "participant": "participant",
                 "recording": "recording_camera", "bp_category": "bp_category"}
SUMMARY_COLUMNS = ("group_by", "group", "signal", "statistic", "metric", "value", "n")
CRITERIA_COLUMNS = ("standard", "clause", "signal", "measurand", "group_by", "group",
                    "metric", "value", "limit", "pass", "note")
COVERAGE_COLUMNS = ("standard", "clause", "signal", "measurand", "band", "share",
                    "required", "pass")
CHANGE_COLUMNS = (*TAGS, "signal", "measurand", "t_start", "t_end",
                  "delta_ref", "delta_pred", "e_percent")
AGREEMENT_METRICS = ("bias", "sd", "loa_low", "loa_high", "mad", "mapd", "rmse",
                     "cp5", "cp10", "cp15", "pearson", "ccc")
HR, WAVEFORM = "hr", "waveform"
BP_SIGNAL = "ABP"
#: ISO 81060-3 clause 5.3.2 g)-i), p. 20: the smallest change that counts,
#: per level of ABP (systolic, MAP, diastolic).
CHANGE_THRESHOLDS = {"max": 15.0, "mean": 12.0, "min": 10.0}
#: ISO 81060-2 Table 1, p. 11: the largest SD of the per-subject mean
#: errors criterion 2 allows, indexed by |pooled mean error| in 0.1 mmHg
#: steps from 0.0 to 5.0.
ISO2_TABLE1 = (
    6.95, 6.95, 6.95, 6.95, 6.93, 6.92, 6.91, 6.90, 6.89, 6.88,
    6.87, 6.86, 6.84, 6.82, 6.80, 6.78, 6.76, 6.73, 6.71, 6.68,
    6.65, 6.62, 6.58, 6.55, 6.51, 6.47, 6.43, 6.39, 6.34, 6.30,
    6.25, 6.20, 6.14, 6.09, 6.03, 5.97, 5.89, 5.83, 5.77, 5.70,
    5.64, 5.56, 5.49, 5.41, 5.33, 5.25, 5.16, 5.08, 5.01, 4.90,
    4.79,
)
#: IEEE 1708 Table 5, p. 27: (low, high, required percent) of readings
#: whose change from the participant's baseline falls in [low, high).
IEEE_CHANGE_BINS = ((-20, -10, 10), (-10, -5, 20), (-5, 5, 0), (5, 15, 20), (15, 25, 25))
_NAN = float("nan")


# ---------------------------------------------------------------------------
# Finding and loading the records
# ---------------------------------------------------------------------------
def find_records_dirs(paths) -> list:
    """Each path as a records directory: itself when it holds ``meta.json``,
    its ``test_records/`` when it is a run directory."""
    dirs = []
    for path in map(Path, paths):
        if (path / META_NAME).is_file():
            dirs.append(path)
        elif (path / RECORDS_DIR / META_NAME).is_file():
            dirs.append(path / RECORDS_DIR)
        else:
            raise FileNotFoundError(
                f"{path} is neither a records directory (no {META_NAME}) nor a "
                f"run directory holding {RECORDS_DIR}/")
    return dirs


@dataclass
class Pool:
    """Every recording's layer-one tables, tagged, plus where they came from."""
    readings: pd.DataFrame
    beats: pd.DataFrame
    rates: pd.DataFrame
    metas: list
    folders: list           # (folder, meta, tags) per recording and camera


def recording_folders(records_dir: Path, meta: dict) -> list:
    first = str(meta["traces"][0])
    return sorted(path.parent for path in Path(records_dir).glob(f"*/*/{first}.csv"))


def _stale(folder: Path, reading_seconds: float, fs: float) -> bool:
    if not all((folder / name).is_file() for name in FILES):
        return True
    readings = pd.read_csv(folder / READINGS_NAME)
    if readings.empty:
        return False
    first = readings.iloc[0]
    return abs(float(first["t_end"] - first["t_start"]) - reading_seconds) > 1.0 / fs


def load_pool(records_dirs, reading_seconds: float) -> Pool:
    """Layer one where needed, then every folder's three tables, tagged."""
    readings, beats, rates, metas, folders = [], [], [], [], []
    for records_dir in map(Path, records_dirs):
        meta = json.loads((records_dir / META_NAME).read_text(encoding="utf-8"))
        metas.append(meta)
        for folder in recording_folders(records_dir, meta):
            if _stale(folder, reading_seconds, float(meta["fs"])):
                score_recording(folder, meta, reading_seconds)
            tags = {"dataset": str(meta["dataset"]), "participant": str(meta["participant"]),
                    "recording": folder.parent.name, "perspective": folder.name}
            folders.append((folder, meta, tags))
            for name, sink in ((READINGS_NAME, readings), (BEATS_NAME, beats),
                               (RATES_NAME, rates)):
                sink.append(pd.read_csv(folder / name).assign(**tags))
    if not readings:
        raise ValueError("no recordings found under "
                         + ", ".join(str(d) for d in records_dirs))
    concat = lambda frames: pd.concat(frames, ignore_index=True)   # noqa: E731
    return Pool(concat(readings), concat(beats), concat(rates), metas, folders)


# ---------------------------------------------------------------------------
# Groupings
# ---------------------------------------------------------------------------
def bp_category(sbp: float, dbp: float) -> str:
    """IEEE 1708 Table 3, p. 22 (2017 ACC/AHA); the higher category wins."""
    if not (np.isfinite(sbp) and np.isfinite(dbp)):
        return "unknown"
    if sbp >= 140 or dbp >= 90:
        return "stage2"
    if sbp >= 130 or dbp >= 80:
        return "stage1"
    if sbp >= 120:
        return "elevated"
    return "normal"


def participant_categories(readings: pd.DataFrame) -> dict:
    """``{participant: category}`` from their mean reference systolic and
    diastolic ABP readings; ``unknown`` without ABP."""
    abp = readings[readings["signal"] == BP_SIGNAL]
    if abp.empty:
        return {}
    means = abp.groupby("participant")[["ref_max_mean", "ref_min_mean"]].mean()
    return {str(p): bp_category(row["ref_max_mean"], row["ref_min_mean"])
            for p, row in means.iterrows()}


def with_groups(frame: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """The frame with the columns ``GROUP_COLUMNS`` groups by."""
    out = frame.copy()
    out["participant"] = out["participant"].astype(str)
    out["all"] = "all"
    out["recording_camera"] = (out["recording"].astype(str) + "/"
                               + out["perspective"].astype(str))
    out["bp_category"] = out["participant"].map(categories).fillna("unknown")
    return out


def _first_readings(readings: pd.DataFrame) -> pd.DataFrame:
    """Each participant's first reading of each signal, in recording order:
    the calibration-free baseline (IEEE 1708 clause 4.4.2, p. 24)."""
    ordered = readings.sort_values(["participant", "recording", "perspective", "reading"])
    return ordered.groupby(["participant", "signal"], sort=False).head(1)


def baseline_change(readings: pd.DataFrame, s: str) -> pd.Series:
    """Reference level of each reading minus its participant's baseline."""
    first = _first_readings(readings).set_index(["participant", "signal"])[f"ref_{s}_mean"]
    keys = pd.MultiIndex.from_arrays([readings["participant"].astype(str),
                                      readings["signal"]])
    baseline = first.reindex(keys).to_numpy()
    return pd.Series(readings[f"ref_{s}_mean"].to_numpy() - baseline, index=readings.index)


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------
def _finite_mean(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else _NAN


def agreement(ref, pred) -> dict:
    """Bias, SD, limits of agreement, MAD, MAPD, RMSE, CP5/10/15, Pearson,
    CCC and n over the finite pairs. IEEE 1708 clause 4.6.2, p. 32, names
    MAD, MAPD, MD (bias), SD and CP_L as the report columns."""
    ref, pred = np.asarray(ref, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    ok = np.isfinite(ref) & np.isfinite(pred)
    ref, pred = ref[ok], pred[ok]
    err = pred - ref
    n = int(err.size)
    if n == 0:
        return {**{m: _NAN for m in AGREEMENT_METRICS}, "n": 0}
    bias = float(err.mean())
    sd = float(err.std(ddof=1)) if n > 1 else _NAN
    absolute = np.abs(err)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(ref != 0, absolute / np.abs(ref) * 100, np.nan)
    return {"bias": bias, "sd": sd, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd,
            "mad": float(absolute.mean()), "mapd": _finite_mean(relative),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "cp5": float((absolute <= 5).mean() * 100),
            "cp10": float((absolute <= 10).mean() * 100),
            "cp15": float((absolute <= 15).mean() * 100),
            "pearson": pearson(ref, pred), "ccc": ccc(ref, pred), "n": n}


def summary_table(readings: pd.DataFrame, rates: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """Per grouping, signal and statistic (``max`` / ``mean`` / ``min`` levels
    for absolute signals, ``waveform`` for every signal, ``hr`` per heart-rate
    source), the agreement metrics, one row each."""
    readings, rates = with_groups(readings, categories), with_groups(rates, categories)
    rows = []

    def add(group_by, group, signal, statistic, metrics):
        n = metrics.pop("n")
        rows.extend({"group_by": group_by, "group": str(group), "signal": signal,
                     "statistic": statistic, "metric": metric, "value": value, "n": n}
                    for metric, value in metrics.items())

    for group_by, column in GROUP_COLUMNS.items():
        for group, chunk in readings.groupby(column, sort=False):
            for sig, part in chunk.groupby("signal", sort=False):
                if is_absolute(sig):
                    for s in LEVELS:
                        add(group_by, group, sig, s,
                            agreement(part[f"ref_{s}_mean"], part[f"pred_{s}_mean"]))
                metrics = {m: _finite_mean(part[f"waveform_{m}"]) for m in WAVEFORM_METRICS}
                metrics["n"] = int(part["waveform_mad"].notna().sum())
                add(group_by, group, sig, WAVEFORM, metrics)
        for group, chunk in rates.groupby(column, sort=False):
            for source, part in chunk.groupby("source", sort=False):
                metrics = agreement(part["ref_hr"], part["pred_hr"])
                metrics["snr"], metrics["macc"] = _finite_mean(part["snr"]), _finite_mean(part["macc"])
                add(group_by, group, source, HR, metrics)
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


# ---------------------------------------------------------------------------
# The standards' statistics
# ---------------------------------------------------------------------------
def iso2_table1(mean_error: float) -> float:
    """ISO 81060-2 Table 1, p. 11: the largest SD of the per-subject mean
    errors criterion 2 allows, for a pooled mean error rounded to 0.1 mmHg;
    NaN beyond ±5.0 (criterion 1 fails there anyway). The standard's own
    example: a mean of ±4.2 mmHg allows 5.49."""
    if not np.isfinite(mean_error):
        return _NAN
    tenths = int(round(abs(mean_error) * 10))
    return ISO2_TABLE1[tenths] if tenths < len(ISO2_TABLE1) else _NAN


def corrected_sd(errors, subjects) -> dict:
    """ISO 81060-3 formulas (5), (6) and (9) to (12), pp. 12-15, in their
    general unequal-count form: the between- and within-subject mean
    squares, the Bland-Altman factor, the corrected SD, the intra-class
    correlation and the effective number of independent readings.
    ``{n, k, s_corr, icc, n_ind}``; the last three NaN with fewer than two
    subjects or no repeats."""
    frame = pd.DataFrame({"x": np.asarray(errors, dtype=np.float64),
                          "s": np.asarray(subjects).astype(str)}).dropna()
    n, k = len(frame), int(frame["s"].nunique())
    out = {"n": n, "k": k, "s_corr": _NAN, "icc": _NAN, "n_ind": _NAN}
    if k < 2 or n <= k:
        return out
    mean = frame["x"].mean()
    per = frame.groupby("s")["x"].agg(["count", "mean", "var"])
    m = per["count"].to_numpy(dtype=np.float64)
    f_ba = (n ** 2 - (m ** 2).sum()) / ((k - 1) * n)                      # (10)
    mu_sb = float((m * (per["mean"].to_numpy() - mean) ** 2).sum() / (k - 1))   # (11)
    mu_sw = float(((m - 1) * per["var"].fillna(0.0).to_numpy()).sum() / (n - k))  # (12)
    between = max(0.0, (mu_sb - mu_sw) / f_ba)
    total = between + mu_sw                                              # (9) squared
    icc = between / total if total > 0 else _NAN                         # (5)
    r = n / k
    out.update({"s_corr": float(np.sqrt(total)), "icc": float(icc),
                "n_ind": float(k * (1 + (1 - icc) * (r - 1))) if np.isfinite(icc) else _NAN})  # (6)
    return out


def type_t_errors(part: pd.DataFrame, column: str) -> pd.Series:
    """ISO 81060-3 formulas (14) and (15), pp. 18-19: the errors with each
    participant's offset removed, the offset being their mean error over
    their first recording and camera (their first analysis period)."""
    offsets = {}
    for participant, group in part.groupby("participant"):
        first = group.sort_values(["recording", "perspective", "reading"]).iloc[0]
        same = ((group["recording"] == first["recording"])
                & (group["perspective"] == first["perspective"]))
        offsets[participant] = group.loc[same, column].mean()
    return part[column] - part["participant"].map(offsets)


def change_events(readings: pd.DataFrame, change_seconds: float) -> pd.DataFrame:
    """ISO 81060-3 clause 5.3.4, pp. 21-22: within each recording and camera,
    every pair of ABP readings at most ``change_seconds`` apart whose
    reference or predicted change reaches the level's threshold, with the
    unsigned relative error of the two changes (formula (18)) in percent."""
    rows = []
    abp = readings[readings["signal"] == BP_SIGNAL]
    for tags, group in abp.groupby(list(TAGS), sort=False):
        group = group.sort_values("t_start")
        t = group["t_start"].to_numpy(dtype=np.float64)
        for s, threshold in CHANGE_THRESHOLDS.items():
            ref = group[f"ref_{s}_mean"].to_numpy(dtype=np.float64)
            pred = group[f"pred_{s}_mean"].to_numpy(dtype=np.float64)
            for i in range(len(t)):
                for j in range(i + 1, len(t)):
                    if t[j] - t[i] > change_seconds:
                        break
                    d_ref, d_pred = ref[j] - ref[i], pred[j] - pred[i]
                    if not (np.isfinite(d_ref) and np.isfinite(d_pred)):
                        continue
                    largest = max(abs(d_ref), abs(d_pred))
                    if largest < threshold:
                        continue
                    rows.append({**dict(zip(TAGS, tags)), "signal": BP_SIGNAL,
                                 "measurand": s, "t_start": t[i], "t_end": t[j],
                                 "delta_ref": d_ref, "delta_pred": d_pred,
                                 "e_percent": abs(d_pred - d_ref) / largest * 100})
    return pd.DataFrame(rows, columns=list(CHANGE_COLUMNS))


def ieee_grade(mad: float, bias: float) -> str:
    """IEEE 1708 Table 6, p. 30: A for MAD <= 5; B for MAD in (5, 6] and
    |MD| <= 5; C for MAD in (6, 7] and |MD| <= 5; D otherwise. Empty when
    there is nothing to grade."""
    if not np.isfinite(mad):
        return ""
    if mad <= 5:
        return "A"
    if mad > 7 or not (np.isfinite(bias) and abs(bias) <= 5):
        return "D"
    return "B" if mad <= 6 else "C"


def ieee_waveform_grade(mad: float, r: float) -> str:
    """IEEE 1708 Table 7, p. 30, for a continuous waveform device: the worse
    of the MAD letter (<= 5 A, <= 6 B, <= 7 C) and the r letter (>= 0.9 A,
    >= 0.8 B, >= 0.7 C); D beyond either."""
    if not (np.isfinite(mad) and np.isfinite(r)):
        return ""
    if mad > 7 or r < 0.7:
        return "D"
    by_mad = "A" if mad <= 5 else "B" if mad <= 6 else "C"
    by_r = "A" if r >= 0.9 else "B" if r >= 0.8 else "C"
    return max(by_mad, by_r)


def criteria_table(readings: pd.DataFrame, categories: dict,
                   change_seconds: float) -> tuple:
    """``(criteria, change events)``: every standard's statistics on the ABP
    readings, one row per metric, with the standard's limit and pass mark
    where it fixes one."""
    rows = []

    def add(standard, clause, measurand, metric, value, limit=None, passed=None,
            note="", group_by="all", group="all"):
        rows.append({"standard": standard, "clause": clause, "signal": BP_SIGNAL,
                     "measurand": measurand, "group_by": group_by, "group": group,
                     "metric": metric, "value": value, "limit": limit,
                     "pass": passed, "note": note})

    def within(value, limit):
        return bool(abs(value) <= limit) if np.isfinite(value) else None

    def at_most(value, limit):
        return bool(value <= limit) if np.isfinite(value) else None

    def at_least(value, limit):
        return bool(value >= limit) if np.isfinite(value) else None

    changes = change_events(readings, change_seconds)
    abp = readings[readings["signal"] == BP_SIGNAL].copy()
    abp["participant"] = abp["participant"].astype(str)
    if abp.empty:
        return pd.DataFrame(columns=list(CRITERIA_COLUMNS)), changes
    grades = []
    for s in LEVELS:
        name = beat_labels(BP_SIGNAL)[s]
        ref, pred = abp[f"ref_{s}_mean"], abp[f"pred_{s}_mean"]
        err = abp[f"err_{s}"]
        agree = agreement(ref, pred)

        # ISO 81060-2 criterion 1: pooled bias and SD, plain and dead-band.
        for clause, column in (("5.2.4.1.2 a) criterion 1", f"err_{s}"),
                               ("6.2.6 criterion 1 with the 6.2.5 dead band", f"err_{s}_deadband")):
            e = abp[column].dropna()
            bias = float(e.mean()) if len(e) else _NAN
            sd = float(e.std(ddof=1)) if len(e) > 1 else _NAN
            add("ISO 81060-2", clause, name, "bias", bias, 5.0, within(bias, 5.0))
            add("ISO 81060-2", clause, name, "sd", sd, 8.0, at_most(sd, 8.0))

        # ISO 81060-2 criterion 2: SD of per-participant means about the pooled mean.
        pooled = float(err.mean()) if err.notna().any() else _NAN
        per = abp.dropna(subset=[f"err_{s}"]).groupby("participant")[f"err_{s}"].mean()
        sm = float(np.sqrt(((per - pooled) ** 2).sum() / (len(per) - 1))) if len(per) > 1 else _NAN
        limit = iso2_table1(pooled)
        add("ISO 81060-2", "5.2.4.1.2 b) criterion 2, Table 1", name,
            "sd_of_subject_means", sm, limit,
            at_most(sm, limit) if np.isfinite(limit) else None,
            note="cuff route only; the invasive route (6.2.6) is exempt")

        # ISO 81060-3 5.1.4, Type A; 5.2.4 b), Type T.
        c = corrected_sd(err, abp["participant"])
        add("ISO 81060-3", "5.1.4 Type A", name, "bias", pooled, 6.0, within(pooled, 6.0))
        add("ISO 81060-3", "5.1.4 Type A", name, "s_corr", c["s_corr"], 10.0, at_most(c["s_corr"], 10.0))
        add("ISO 81060-3", "5.1.4 Type A", name, "n_ind", c["n_ind"], 278, at_least(c["n_ind"], 278))
        add("ISO 81060-3", "5.1.4 Type A", name, "icc", c["icc"])
        add("ISO 81060-3", "4.5.1", name, "subjects", c["k"], 30, None,
            note="achieved; the standard needs at least 30")
        add("ISO 81060-3", "4.5.1", name, "readings", c["n"])
        ct = corrected_sd(type_t_errors(abp, f"err_{s}"), abp["participant"])
        add("ISO 81060-3", "5.2.4 b) Type T", name, "s_corr", ct["s_corr"], 6.0, at_most(ct["s_corr"], 6.0),
            note="per-participant offset from their first recording removed")
        add("ISO 81060-3", "5.2.4 b) Type T", name, "n_ind", ct["n_ind"], 278, at_least(ct["n_ind"], 278))

        # ISO 81060-3 5.3.5: change tracking.
        events = changes[changes["measurand"] == s]
        if not events.empty:
            per_subject = events.groupby("participant")["e_percent"]
            p50 = float(per_subject.quantile(0.5).mean())
            p85 = float(per_subject.quantile(0.85).mean())
            add("ISO 81060-3", "5.3.5", name, "p50_mean", p50, 25.0, at_most(p50, 25.0))
            add("ISO 81060-3", "5.3.5", name, "p85_mean", p85, 50.0, at_most(p85, 50.0))
            add("ISO 81060-3", "5.3.2 d)", name, "events", float(len(events)))
            add("ISO 81060-3", "5.3.2 d)", name, "min_events_per_subject",
                float(per_subject.size().min()), 50, None,
                note="achieved; the standard needs at least 50 per subject")

        # IEEE 1708 grades.
        grade = ieee_grade(agree["mad"], agree["bias"])
        grades.append(grade)
        add("IEEE 1708", "4.5.3.1 Table 6", name, "mad", agree["mad"], 7.0,
            (grade != "D") if grade else None, note=f"grade {grade}")
        add("IEEE 1708", "4.5.3.1 Table 6", name, "bias", agree["bias"], 5.0, within(agree["bias"], 5.0))
        for category, chunk in with_groups(abp, categories).groupby("bp_category"):
            if category in ("stage2", "unknown"):
                continue
            a = agreement(chunk[f"ref_{s}_mean"], chunk[f"pred_{s}_mean"])
            add("IEEE 1708", "4.5.3.4", name, "mad", a["mad"], 6.0, at_most(a["mad"], 6.0),
                group_by="bp_category", group=str(category))
    worst = max(g for g in grades if g) if any(grades) else ""
    add("IEEE 1708", "4.5.3.1 worst cell", "all", "grade", _NAN, None,
        (worst != "D") if worst else None, note=f"grade {worst}")
    mad, r = _finite_mean(abp["waveform_mad"]), _finite_mean(abp["waveform_r"])
    wave = ieee_waveform_grade(mad, r)
    add("IEEE 1708", "4.5.2 and Table 7 (waveform)", WAVEFORM, "mad", mad, 7.0,
        at_most(mad, 7.0), note=f"grade {wave}")
    add("IEEE 1708", "4.5.2 and Table 7 (waveform)", WAVEFORM, "r", r, 0.7,
        bool(r > 0.7) if np.isfinite(r) else None, note=f"grade {wave}")
    return pd.DataFrame(rows, columns=list(CRITERIA_COLUMNS)), changes


# ---------------------------------------------------------------------------
# Coverage of the reference range
# ---------------------------------------------------------------------------
#: (standard, clause, level, band label, predicate, required percent).
_BANDS = (
    ("ISO 81060-2", "6.1.5", "max", "<= 100", lambda v: v <= 100, 10),
    ("ISO 81060-2", "6.1.5", "max", ">= 160", lambda v: v >= 160, 10),
    ("ISO 81060-2", "6.1.5", "min", "<= 70", lambda v: v <= 70, 10),
    ("ISO 81060-2", "6.1.5", "min", ">= 85", lambda v: v >= 85, 10),
    ("ISO 81060-3", "4.3.3", "max", "<= 90", lambda v: v <= 90, 5),
    ("ISO 81060-3", "4.3.3", "max", "<= 110", lambda v: v <= 110, 20),
    ("ISO 81060-3", "4.3.3", "max", "110 < x < 140", lambda v: (v > 110) & (v < 140), 20),
    ("ISO 81060-3", "4.3.3", "max", ">= 140", lambda v: v >= 140, 20),
    ("ISO 81060-3", "4.3.3", "max", ">= 160", lambda v: v >= 160, 5),
    ("ISO 81060-3", "4.3.3", "min", "<= 50", lambda v: v <= 50, 5),
    ("ISO 81060-3", "4.3.3", "min", "<= 60", lambda v: v <= 60, 20),
    ("ISO 81060-3", "4.3.3", "min", "60 < x < 80", lambda v: (v > 60) & (v < 80), 20),
    ("ISO 81060-3", "4.3.3", "min", ">= 80", lambda v: v >= 80, 20),
    ("ISO 81060-3", "4.3.3", "min", ">= 90", lambda v: v >= 90, 5),
    ("ISO 81060-3", "4.3.3", "mean", "<= 65", lambda v: v <= 65, 5),
    ("ISO 81060-3", "4.3.3", "mean", "<= 75", lambda v: v <= 75, 20),
    ("ISO 81060-3", "4.3.3", "mean", "75 < x < 100", lambda v: (v > 75) & (v < 100), 20),
    ("ISO 81060-3", "4.3.3", "mean", ">= 100", lambda v: v >= 100, 20),
    ("ISO 81060-3", "4.3.3", "mean", ">= 115", lambda v: v >= 115, 5),
)


def coverage_table(readings: pd.DataFrame) -> pd.DataFrame:
    """The share of reference ABP readings in each band the standards want
    covered: ISO 81060-2 clause 6.1.5, pp. 18-19 (invasive route), ISO
    81060-3 clause 4.3.3, pp. 7-8, and IEEE 1708 Table 5, p. 27, the change
    from each participant's baseline reading."""
    rows = []
    abp = readings[readings["signal"] == BP_SIGNAL]
    if abp.empty:
        return pd.DataFrame(columns=list(COVERAGE_COLUMNS))
    labels = beat_labels(BP_SIGNAL)
    for standard, clause, s, band, predicate, required in _BANDS:
        values = abp[f"ref_{s}_mean"].dropna().to_numpy(dtype=np.float64)
        share = float(predicate(values).mean() * 100) if values.size else _NAN
        rows.append({"standard": standard, "clause": clause, "signal": BP_SIGNAL,
                     "measurand": labels[s], "band": band, "share": share,
                     "required": required,
                     "pass": bool(share >= required) if np.isfinite(share) else None})
    for s in ("max", "min"):
        change = baseline_change(abp, s).dropna().to_numpy(dtype=np.float64)
        for low, high, required in IEEE_CHANGE_BINS:
            share = float(((change >= low) & (change < high)).mean() * 100) if change.size else _NAN
            rows.append({"standard": "IEEE 1708", "clause": "Table 5 (change from baseline)",
                         "signal": BP_SIGNAL, "measurand": labels[s],
                         "band": f"[{low}, {high})", "share": share, "required": required,
                         "pass": (bool(share >= required) if np.isfinite(share) and required
                                  else None)})
    return pd.DataFrame(rows, columns=list(COVERAGE_COLUMNS))
```

- [ ] **Step 2: Append the Table 1 test to `tests/test_evaluation.py`**

```python
from src.evaluation.aggregate import iso2_table1


def test_iso2_table1_matches_the_standards_example():
    assert iso2_table1(4.2) == 5.49 and iso2_table1(-4.2) == 5.49   # the standard's example
    assert iso2_table1(0.0) == 6.95 and iso2_table1(5.0) == 4.79
    assert np.isnan(iso2_table1(5.1)) and np.isnan(iso2_table1(float("nan")))
```

Put the import beside the other one at the top of the file.

- [ ] **Step 3: Run the unit tests and the pool over the real run**

```bash
uv run pytest -p no:faulthandler tests/test_evaluation.py -q
uv run python -c "
from src.evaluation.aggregate import *
dirs = find_records_dirs(['runs/BIGSMALL_PURE.01_202609091150'])
pool = load_pool(dirs, 30.0)
print(len(pool.folders), 'folders;', len(pool.readings), 'reading rows;', len(pool.beats), 'beats')
cats = participant_categories(pool.readings)
summary = summary_table(pool.readings, pool.rates, cats)
print(summary[(summary.group_by == 'all')].to_string())
criteria, changes = criteria_table(pool.readings, cats, 60.0)
print(len(criteria), 'criteria rows;', len(changes), 'change events')
print(coverage_table(pool.readings).head())"
ls runs/BIGSMALL_PURE.01_202609091150/test_records/01-03/1/
```

Expected: 2 tests pass; 6 folders, 12 reading rows (two per recording),
a few hundred beats; the `all` summary has `PPG waveform` rows and `PPG hr`
rows with finite values; 0 criteria rows and 0 change events (no ABP on
PURE); an empty coverage frame; and every recording folder now holds the
three layer-one files. Run the pool line a second time and confirm it is
fast (no re-scoring: the files are fresh). Then run with `10.0` instead
of `30.0` and confirm the readings count becomes 36 (six per recording)
and the files were re-cut.

---

### Task 6: `plots.py`, every figure

**Files:**
- Rewrite: `src/evaluation/plots.py` (delete the current content entirely)

**Interfaces:**
- Consumes: Task 5's `GROUP_COLUMNS`, `BP_SIGNAL`, `Pool`, `baseline_change`, `with_groups`; Task 4's `read_trace`; Task 2's `LEVELS`; registry `beat_labels`, `is_absolute`, `is_cardiac`, `signal_unit`.
- Produces:
  - `evenly_spaced(items, count) -> list`
  - `aggregate_figures(pool, categories, changes) -> list[tuple[str, Figure]]`
  - `recording_figures(folder, meta, tags, pool) -> list[tuple[str, Figure]]`
  - the individual drawing functions `bland_altman`, `identity`, `histogram`, `scatter`, `ribbon`, `beat_overlay`, `beat_scatter`, `trend`, each returning a `matplotlib.figure.Figure`

- [ ] **Step 1: Write the module**

```python
"""Every figure the evaluation draws, once, for every signal and grouping.

Aggregate figures, per absolute signal and level: Bland-Altman (difference
against the mean of the two; ISO 81060-2 Amd 2 clause 5.1.4 i), p. 3) and
the identity scatter, coloured by participant and by recording; the
reference-reading histogram and the reference against time since the
recording start (ISO 81060-3 clause 5.1.3 f), p. 14); for ABP the error
against change from baseline and the change histogram (IEEE 1708 clause
4.6.3, p. 35, figures 5 and 6), and per change-event level the
reference-delta and relative-error histograms (ISO 81060-3 clause 5.3.4
c) and d), p. 22, figures A.6 to A.8). Plus one heart-rate Bland-Altman
per source.

Per-recording figures, for a sample of recordings: the ribbon (the
combined prediction ± its SD across overlapping windows over the label,
the repeatability plot), the beat overlay (both traces with the detected
beats, matched pairs joined, misses marked), the per-beat scatter, and
the trend of reference and predicted levels over time (ISO 81060-3 figure
A.3, p. 30, where lagged or under-reported changes show).

Seaborn on the Agg backend; every function returns a figure and never
saves, so the report can place it and the entry point can also save it.
"""

import matplotlib
matplotlib.use("Agg")            # write files; never open a window on a cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.evaluation.aggregate import BP_SIGNAL, GROUP_COLUMNS, baseline_change, with_groups
from src.evaluation.beats import LEVELS
from src.evaluation.recording import read_trace
from src.signal_transforms import beat_labels, is_absolute, is_cardiac, signal_unit

sns.set_theme(style="whitegrid", context="paper")
#: Colour by these groupings in the aggregate scatter plots.
HUE_GROUPINGS = ("participant", "recording")
#: A legend with more entries than this is noise; the colours still separate.
MAX_LEGEND = 12
#: Seconds of trace the per-recording ribbon and beat overlay show.
OVERLAY_SECONDS = 8.0
LINESTYLES = {"max": "-", "mean": "--", "min": ":"}


def evenly_spaced(items, count: int) -> list:
    """``count`` items spread across the list, not its head."""
    items = list(items)
    if len(items) <= count:
        return items
    picks = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in sorted(set(picks.tolist()))]


def _empty(title: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6, 3))
    axis.text(0.5, 0.5, "no finite data", ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()
    figure.suptitle(title, fontsize=10)
    return figure


def _pairs(ref, pred, hue=None) -> pd.DataFrame:
    frame = pd.DataFrame({"ref": np.asarray(ref, dtype=np.float64),
                          "pred": np.asarray(pred, dtype=np.float64)})
    frame["group"] = np.asarray(hue).astype(str) if hue is not None else "all"
    return frame[np.isfinite(frame["ref"]) & np.isfinite(frame["pred"])]


# ---------------------------------------------------------------------------
# Aggregate figures
# ---------------------------------------------------------------------------
def bland_altman(ref, pred, hue, title: str, unit: str) -> plt.Figure:
    """Difference against mean with bias and limits, coloured by ``hue``."""
    pairs = _pairs(ref, pred, hue)
    if pairs.empty:
        return _empty(title)
    pairs["mean"], pairs["diff"] = (pairs["ref"] + pairs["pred"]) / 2, pairs["pred"] - pairs["ref"]
    bias = float(pairs["diff"].mean())
    sd = float(pairs["diff"].std(ddof=1)) if len(pairs) > 1 else float("nan")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(pairs, x="mean", y="diff", hue="group", s=16, alpha=0.7,
                    edgecolor="none", ax=axis, legend=pairs["group"].nunique() <= MAX_LEGEND)
    axis.axhline(bias, linestyle="--", color="0.3", label=f"bias {bias:+.2f} {unit}")
    if np.isfinite(sd):
        for limit in (bias + 1.96 * sd, bias - 1.96 * sd):
            axis.axhline(limit, linestyle=":", color="0.5")
        title = f"{title}: bias {bias:+.2f}, limits ± {1.96 * sd:.2f} {unit} ({len(pairs)} readings)"
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel(f"mean of reference and predicted ({unit})")
    axis.set_ylabel(f"predicted - reference ({unit})")
    axis.set_title(title, fontsize=10)
    if axis.get_legend() is not None:
        axis.legend(fontsize=7, loc="upper right")
    figure.tight_layout()
    return figure


def identity(ref, pred, hue, title: str, unit: str) -> plt.Figure:
    """Predicted against reference with the identity line."""
    pairs = _pairs(ref, pred, hue)
    if pairs.empty:
        return _empty(title)
    figure, axis = plt.subplots(figsize=(5, 5))
    sns.scatterplot(pairs, x="ref", y="pred", hue="group", s=16, alpha=0.7,
                    edgecolor="none", ax=axis, legend=pairs["group"].nunique() <= MAX_LEGEND)
    low = min(pairs["ref"].min(), pairs["pred"].min())
    high = max(pairs["ref"].max(), pairs["pred"].max())
    axis.plot([low, high], [low, high], linestyle="--", linewidth=1, color="0.4")
    axis.set_xlabel(f"reference ({unit})")
    axis.set_ylabel(f"predicted ({unit})")
    axis.set_title(title, fontsize=10)
    if axis.get_legend() is not None:
        axis.legend(fontsize=7, loc="upper left")
    figure.tight_layout()
    return figure


def histogram(values, title: str, unit: str, bins: int = 20) -> plt.Figure:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return _empty(title)
    figure, axis = plt.subplots(figsize=(6, 3.5))
    sns.histplot(x=values, bins=bins, ax=axis)
    axis.set_xlabel(unit)
    axis.set_ylabel("readings")
    axis.set_title(f"{title} ({values.size})", fontsize=10)
    figure.tight_layout()
    return figure


def scatter(x, y, xlabel: str, ylabel: str, title: str) -> plt.Figure:
    pairs = _pairs(x, y)
    if pairs.empty:
        return _empty(title)
    figure, axis = plt.subplots(figsize=(6, 3.8))
    sns.scatterplot(pairs, x="ref", y="pred", s=16, alpha=0.7, edgecolor="none", ax=axis)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=10)
    figure.tight_layout()
    return figure


def aggregate_figures(pool, categories: dict, changes: pd.DataFrame) -> list:
    """``[(name, figure), ...]`` for the pooled readings, in report order."""
    readings = with_groups(pool.readings, categories)
    figures = []
    for sig in [s for s in readings["signal"].unique() if is_absolute(s)]:
        unit, labels = signal_unit(sig), beat_labels(sig)
        part = readings[readings["signal"] == sig]
        for s in LEVELS:
            ref, pred = part[f"ref_{s}_mean"], part[f"pred_{s}_mean"]
            for grouping in HUE_GROUPINGS:
                hue = part[GROUP_COLUMNS[grouping]]
                figures.append((f"{sig}_{s}_bland_altman_by_{grouping}",
                                bland_altman(ref, pred, hue, f"{sig} {labels[s]}", unit)))
                figures.append((f"{sig}_{s}_identity_by_{grouping}",
                                identity(ref, pred, hue, f"{sig} {labels[s]}", unit)))
            figures.append((f"{sig}_{s}_reference_histogram",
                            histogram(ref, f"{sig} {labels[s]}: reference readings", unit)))
            figures.append((f"{sig}_{s}_reference_vs_time",
                            scatter(part["t_start"], ref, "time since recording start (s)",
                                    f"reference {labels[s]} ({unit})",
                                    f"{sig} {labels[s]}: reference against time")))
            if sig == BP_SIGNAL:
                change = baseline_change(part, s)
                figures.append((f"{sig}_{s}_error_vs_change",
                                scatter(change, part[f"err_{s}"],
                                        f"reference change from baseline ({unit})",
                                        f"predicted - reference ({unit})",
                                        f"{sig} {labels[s]}: error against change from baseline")))
                figures.append((f"{sig}_{s}_change_histogram",
                                histogram(change, f"{sig} {labels[s]}: change from baseline", unit)))
    for s, events in changes.groupby("measurand", sort=False):
        label = beat_labels(BP_SIGNAL)[s]
        figures.append((f"{BP_SIGNAL}_{s}_delta_reference_histogram",
                        histogram(events["delta_ref"], f"{BP_SIGNAL} {label}: reference changes", "mmHg")))
        figures.append((f"{BP_SIGNAL}_{s}_change_error_histogram",
                        histogram(events["e_percent"], f"{BP_SIGNAL} {label}: change-tracking error", "%")))
    for source, part in pool.rates.groupby("source", sort=False):
        figures.append((f"HR_{source}_bland_altman",
                        bland_altman(part["ref_hr"], part["pred_hr"], None,
                                     f"heart rate from {source}", "bpm")))
    return figures


# ---------------------------------------------------------------------------
# Per-recording figures
# ---------------------------------------------------------------------------
def _window(trace: pd.DataFrame, t0: float, seconds: float) -> pd.DataFrame:
    return trace[(trace["t"] >= t0) & (trace["t"] < t0 + seconds)]


def ribbon(trace: pd.DataFrame, sig: str, t0: float, seconds: float, title: str) -> plt.Figure:
    """The combined prediction ± its SD across overlapping windows, over the label."""
    chunk = _window(trace, t0, seconds)
    if chunk.empty:
        return _empty(title)
    figure, axis = plt.subplots(figsize=(10, 3.2))
    t = chunk["t"].to_numpy()
    axis.plot(t, chunk["label"], label="label", linewidth=1.2, color="0.2")
    axis.plot(t, chunk["mean"], label="prediction (mean over windows)", linewidth=1.0)
    axis.fill_between(t, chunk["mean"] - chunk["std"], chunk["mean"] + chunk["std"],
                      alpha=0.25, label="± 1 SD across windows")
    axis.set_xlabel("time (s)")
    axis.set_ylabel(signal_unit(sig))
    axis.set_title(title, fontsize=10)
    axis.legend(fontsize=7, loc="upper right")
    figure.tight_layout()
    return figure


def beat_overlay(trace: pd.DataFrame, beats: pd.DataFrame, sig: str, t0: float,
                 seconds: float, title: str) -> plt.Figure:
    """Both traces with the detected beats: matched pairs joined, misses marked."""
    chunk = _window(trace, t0, seconds)
    if chunk.empty:
        return _empty(title)
    figure, axis = plt.subplots(figsize=(10, 3.2))
    t = chunk["t"].to_numpy()
    axis.plot(t, chunk["label"], label="label", linewidth=1.2, color="0.2")
    axis.plot(t, chunk["mean"], label="prediction", linewidth=1.0)
    here = beats[(beats["t_ref"] >= t0) & (beats["t_ref"] < t0 + seconds)]
    ok = np.isfinite(trace["label"].to_numpy(dtype=np.float64))
    okp = np.isfinite(trace["mean"].to_numpy(dtype=np.float64))
    if not (ok.any() and okp.any()):
        here = here.iloc[0:0]           # nothing to place a marker on
    label_at = lambda times: np.interp(times, trace["t"].to_numpy()[ok], trace["label"].to_numpy()[ok])  # noqa: E731
    pred_at = lambda times: np.interp(times, trace["t"].to_numpy()[okp], trace["mean"].to_numpy()[okp])  # noqa: E731
    matched, missed = here[here["t_pred"].notna()], here[here["t_pred"].isna()]
    if not matched.empty:
        axis.scatter(matched["t_ref"], label_at(matched["t_ref"]), s=22, color="0.2", zorder=3,
                     label=f"reference beat ({len(matched)} matched)")
        axis.scatter(matched["t_pred"], pred_at(matched["t_pred"]), s=22, color="C0", zorder=3,
                     label="predicted beat")
        for _, row in matched.iterrows():
            axis.plot([row["t_ref"], row["t_pred"]],
                      [label_at(row["t_ref"]), pred_at(row["t_pred"])],
                      color="0.6", linewidth=0.6)
    if not missed.empty:
        axis.scatter(missed["t_ref"], label_at(missed["t_ref"]), s=40, marker="x", color="C3",
                     zorder=4, label=f"missed ({len(missed)})")
    axis.set_xlabel("time (s)")
    axis.set_ylabel(signal_unit(sig))
    axis.set_title(title, fontsize=10)
    axis.legend(fontsize=7, loc="upper right")
    figure.tight_layout()
    return figure


def beat_scatter(beats: pd.DataFrame, sig: str, title: str) -> plt.Figure:
    """Predicted against reference per beat, one panel per level."""
    labels, unit = beat_labels(sig), signal_unit(sig)
    figure, axes = plt.subplots(1, len(LEVELS), figsize=(4 * len(LEVELS), 4))
    drew = False
    for axis, s in zip(axes, LEVELS):
        pairs = _pairs(beats[f"ref_{s}"], beats[f"pred_{s}"])
        axis.set_title(f"{labels[s]} ({len(pairs)} beats)", fontsize=9)
        axis.set_xlabel(f"reference ({unit})")
        axis.set_ylabel(f"predicted ({unit})")
        if pairs.empty:
            continue
        drew = True
        sns.scatterplot(pairs, x="ref", y="pred", s=12, alpha=0.6, edgecolor="none", ax=axis)
        low, high = min(pairs["ref"].min(), pairs["pred"].min()), max(pairs["ref"].max(), pairs["pred"].max())
        axis.plot([low, high], [low, high], linestyle="--", linewidth=1, color="0.4")
    if not drew:
        plt.close(figure)
        return _empty(title)
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    return figure


def trend(readings: pd.DataFrame, sig: str, title: str) -> plt.Figure:
    """Reference and predicted levels per reading over the recording."""
    readings = readings.sort_values("t_start")
    if readings.empty or readings[[f"ref_{s}_mean" for s in LEVELS]].isna().all().all():
        return _empty(title)
    labels, unit = beat_labels(sig), signal_unit(sig)
    figure, axis = plt.subplots(figsize=(10, 3.6))
    t = readings["t_start"].to_numpy()
    for s in LEVELS:
        axis.plot(t, readings[f"ref_{s}_mean"], color="0.2", linestyle=LINESTYLES[s],
                  marker="o", markersize=3, label=f"reference {labels[s]}")
        axis.plot(t, readings[f"pred_{s}_mean"], color="C0", linestyle=LINESTYLES[s],
                  marker="o", markersize=3, label=f"predicted {labels[s]}")
    axis.set_xlabel("reading start (s)")
    axis.set_ylabel(unit)
    axis.set_title(title, fontsize=10)
    axis.legend(fontsize=7, ncol=3, loc="upper right")
    figure.tight_layout()
    return figure


def _rows_of(frame: pd.DataFrame, tags: dict) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for key, value in tags.items():
        mask &= frame[key].astype(str).to_numpy() == str(value)
    return frame[mask]


def recording_figures(folder, meta: dict, tags: dict, pool) -> list:
    """``[(name, figure), ...]`` for one recording and camera."""
    beats_here, readings_here = _rows_of(pool.beats, tags), _rows_of(pool.readings, tags)
    where = f"{tags['participant']} {tags['recording']} camera {tags['perspective']}"
    figures = []
    for sig in [str(s) for s in meta["traces"]]:
        trace = read_trace(folder, sig)
        covered = trace["t"][trace["n"] > 0]
        t0 = float(covered.iloc[0]) if not covered.empty else 0.0
        prefix = f"{tags['participant']}_{tags['recording']}_{tags['perspective']}_{sig}"
        figures.append((f"{prefix}_ribbon",
                        ribbon(trace, sig, t0, OVERLAY_SECONDS, f"{sig} {where}: repeatability across windows")))
        if is_cardiac(sig):
            beats = beats_here[beats_here["signal"] == sig]
            figures.append((f"{prefix}_beats",
                            beat_overlay(trace, beats, sig, t0, OVERLAY_SECONDS, f"{sig} {where}: beats")))
            if is_absolute(sig):
                figures.append((f"{prefix}_beat_scatter",
                                beat_scatter(beats, sig, f"{sig} {where}: per-beat levels")))
        if is_absolute(sig):
            figures.append((f"{prefix}_trend",
                            trend(readings_here[readings_here["signal"] == sig], sig,
                                  f"{sig} {where}: readings over time")))
    return figures
```

- [ ] **Step 2: Verify on the real run**

```bash
uv run python -c "
from src.evaluation.aggregate import *
from src.evaluation.plots import aggregate_figures, recording_figures, evenly_spaced
import matplotlib.pyplot as plt
pool = load_pool(find_records_dirs(['runs/BIGSMALL_PURE.01_202609091150']), 30.0)
cats = participant_categories(pool.readings)
_, changes = criteria_table(pool.readings, cats, 60.0)
figs = aggregate_figures(pool, cats, changes)
print([n for n, _ in figs])
folder, meta, tags = evenly_spaced(pool.folders, 4)[0]
recs = recording_figures(folder, meta, tags, pool)
print([n for n, _ in recs])
for name, fig in recs: fig.savefig(f'{name}.png', dpi=100); plt.close(fig)
for _, fig in figs: plt.close(fig)"
```

Expected: aggregate names are `['HR_PPG_bland_altman']` only (PPG is
shape-class, PURE has no ABP); recording names are the ribbon and beats
figures for PPG. Open the two PNGs written in the repo root: the ribbon
shows the label and the prediction with a shaded band, the beats figure
shows dots on both traces with grey joins and any red crosses. Delete the
PNGs afterwards:

```bash
rm ./*_PPG_ribbon.png ./*_PPG_beats.png
```

---

### Task 7: `report.py`, the PDF and the digest

**Files:**
- Create: `src/evaluation/report.py`

**Interfaces:**
- Consumes: Task 5's `Pool`, `HR`, `WAVEFORM`.
- Produces: `REPORT_NAME = "report.pdf"`, `DIGEST_NAME = "digest.txt"`, `digest(pool, criteria, coverage, summary, reading_seconds) -> str`, `write_report(out_dir, pool, criteria, coverage, summary, aggregate_figs, recording_figs, reading_seconds) -> Path` which also saves each aggregate figure as `<out_dir>/<name>.pdf`, writes the digest, prints it, and closes every figure.

- [ ] **Step 1: Write the module**

```python
"""Layer three of the evaluation: the PdfPages report and its text digest.

The report is one PDF: a title page naming what was evaluated, the
criteria and coverage tables, the pooled summary, then every aggregate
figure and the per-recording figures of the sampled recordings. The
digest is the text of the first pages, printed to the console and written
beside the report, so a cluster log carries the numbers without the PDF.
Tables are rendered as monospace text pages: robust, greppable, no layout
engine.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from src.evaluation.aggregate import HR, WAVEFORM

REPORT_NAME, DIGEST_NAME = "report.pdf", "digest.txt"
LINES_PER_PAGE = 60
PAGE = (11.69, 8.27)          # A4 landscape, inches
_FLOAT = "{:.3g}".format


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(none)"
    return frame.to_string(index=False, float_format=_FLOAT, na_rep="")


def digest(pool, criteria: pd.DataFrame, coverage: pd.DataFrame,
           summary: pd.DataFrame, reading_seconds: float) -> str:
    """The readable summary: what was evaluated, the criteria, the coverage,
    and the pooled summary per signal."""
    readings = pool.readings
    lines = ["=== Evaluation ==="]
    lines.append(f"records directories: {len(pool.metas)}; run directories: "
                 + ", ".join(sorted({str(m.get('run_dir', '?')) for m in pool.metas})))
    lines.append(f"datasets: {', '.join(sorted(readings['dataset'].astype(str).unique()))}; "
                 f"participants: {readings['participant'].nunique()}; "
                 f"recordings: {len(pool.folders)}; "
                 f"readings of {reading_seconds:g} s: {readings['reading'].count()} rows over "
                 f"{', '.join(readings['signal'].unique())}")
    git = sorted({str(m.get('git', {}).get('commit', '?'))[:10] for m in pool.metas})
    lines.append(f"inferred at git {', '.join(git)}")
    lines.append("")
    lines.append("=== Criteria (ABP; prediction minus reference; blank pass = no fixed limit) ===")
    lines.append(_table(criteria.drop(columns=["signal"])))
    lines.append("")
    lines.append("=== Coverage of the reference range (percent of ABP readings) ===")
    lines.append(_table(coverage.drop(columns=["signal"])))
    lines.append("")
    pooled = summary[summary["group_by"] == "all"].drop(columns=["group_by", "group"])
    levels = pooled[~pooled["statistic"].isin([HR, WAVEFORM])]
    lines.append("=== Levels per reading, pooled ===")
    lines.append(_table(levels.pivot_table(index=["signal", "statistic"], columns="metric",
                                           values="value", sort=False).reset_index()
                        if not levels.empty else levels))
    lines.append("")
    lines.append("=== Waveform agreement per reading, pooled (mean over readings) ===")
    wave = pooled[pooled["statistic"] == WAVEFORM]
    lines.append(_table(wave.pivot_table(index="signal", columns="metric", values="value",
                                         sort=False).reset_index() if not wave.empty else wave))
    lines.append("")
    lines.append("=== Heart rate per reading, pooled (bpm; SNR and MACC on the combined "
                 "trace, not comparable with per-window toolbox numbers) ===")
    hr = pooled[pooled["statistic"] == HR]
    lines.append(_table(hr.pivot_table(index="signal", columns="metric", values="value",
                                       sort=False).reset_index() if not hr.empty else hr))
    return "\n".join(lines)


def _text_pages(pdf: PdfPages, title: str, text: str) -> None:
    lines = text.splitlines() or [""]
    for start in range(0, len(lines), LINES_PER_PAGE):
        figure = plt.figure(figsize=PAGE)
        heading = title if start == 0 else f"{title} (continued)"
        figure.text(0.03, 0.96, heading, fontsize=12, weight="bold", va="top")
        figure.text(0.03, 0.92, "\n".join(lines[start:start + LINES_PER_PAGE]),
                    family="monospace", fontsize=6.5, va="top")
        pdf.savefig(figure)
        plt.close(figure)


def write_report(out_dir, pool, criteria: pd.DataFrame, coverage: pd.DataFrame,
                 summary: pd.DataFrame, aggregate_figs: list, recording_figs: list,
                 reading_seconds: float) -> Path:
    """Write ``report.pdf`` and ``digest.txt`` into ``out_dir``, save every
    aggregate figure as its own PDF beside them, print the digest, close
    every figure; returns the report path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = digest(pool, criteria, coverage, summary, reading_seconds)
    (out_dir / DIGEST_NAME).write_text(text + "\n", encoding="utf-8")
    print(text)
    report = out_dir / REPORT_NAME
    with PdfPages(report) as pdf:
        _text_pages(pdf, "Evaluation", text)
        for name, figure in aggregate_figs:
            figure.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
        for _, figure in recording_figs:
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    return report
```

- [ ] **Step 2: Verify on the real run**

```bash
uv run python -c "
from src.evaluation.aggregate import *
from src.evaluation.plots import aggregate_figures, recording_figures, evenly_spaced
from src.evaluation.report import write_report
pool = load_pool(find_records_dirs(['runs/BIGSMALL_PURE.01_202609091150']), 30.0)
cats = participant_categories(pool.readings)
summary = summary_table(pool.readings, pool.rates, cats)
criteria, changes = criteria_table(pool.readings, cats, 60.0)
figs = aggregate_figures(pool, cats, changes)
recs = [f for folder, meta, tags in evenly_spaced(pool.folders, 2) for f in recording_figures(folder, meta, tags, pool)]
print(write_report('scratch_eval', pool, criteria, coverage_table(pool.readings), summary, figs, recs, 30.0))"
ls scratch_eval
```

Expected: the digest prints with an `=== Evaluation ===` header, `(none)`
under criteria and coverage, PPG rows under waveform and heart rate;
`scratch_eval/` holds `report.pdf`, `digest.txt` and
`HR_PPG_bland_altman.pdf`. Open `report.pdf`: text pages, then the
heart-rate Bland-Altman, then four per-recording pages. Then
`rm -r scratch_eval`.

---

### Task 8: `scripts/eval.py`, deletions, the smoke test

**Files:**
- Rewrite: `scripts/eval.py`
- Rewrite: `src/evaluation/__init__.py` (docstring only)
- Delete: `src/evaluation/evaluate.py`, `src/evaluation/records.py`, `tests/test_evaluate.py`
- Create: `tests/test_to_physical.py`
- Modify: `tests/test_scripts.py` (import at line 17; the end of `test_train_writes_the_run_and_infer_rebuilds_it`)

**Interfaces:**
- Consumes: Tasks 5, 6, 7.
- Produces: `scripts.eval.main(argv=None) -> Path` returning the evaluation directory; `EVALUATION_DIR = "evaluation"`.

- [ ] **Step 1: Write `scripts/eval.py`**

```python
"""Score a run's records against the clinical blood-pressure standards and
write the report.

    uv run python scripts/eval.py runs/PHYSNET_PURE.01_202609091000
    uv run python scripts/eval.py runs/A/test_records runs/B/test_records --out sweeps/loso

Each positional path is a records directory ``scripts/infer.py`` wrote, or a
run directory as shorthand for its ``test_records/``. No config, interface,
checkpoint or torch is needed: everything comes from the CSVs and
``meta.json``. Layer one (``src/evaluation/recording.py``) writes
``beats.csv``, ``readings.csv`` and ``rates.csv`` beside each recording's
trace tables where they are missing or cut at another length; layer two
(``aggregate.py``) pools them into ``readings.csv``, ``beats.csv``,
``rates.csv``, ``summary.csv``, ``criteria.csv``, ``coverage.csv`` and
``changes.csv``; layer three (``plots.py``, ``report.py``) writes the
figures, ``report.pdf`` and ``digest.txt``. They land in
``<records dir>/evaluation/`` or ``--out``, which is required when pooling
several directories, as a LOSO sweep does. ``docs/evaluation.md`` lists
every column and the clause it comes from.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.aggregate import (                           # noqa: E402
    coverage_table, criteria_table, find_records_dirs, load_pool,
    participant_categories, summary_table,
)
from src.evaluation.plots import (                               # noqa: E402
    aggregate_figures, evenly_spaced, recording_figures,
)
from src.evaluation.report import write_report                   # noqa: E402
from src.outputs import FLOAT_FORMAT                             # noqa: E402

EVALUATION_DIR = "evaluation"
DEFAULT_READING_SECONDS = 30.0
DEFAULT_CHANGE_SECONDS = 60.0
DEFAULT_REPORT_RECORDINGS = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score one or more records directories against the clinical "
                    "blood-pressure standards and write the report.")
    parser.add_argument(
        "dirs", nargs="+", metavar="DIR",
        help="records directories (or run directories holding test_records/)")
    parser.add_argument(
        "--out", metavar="DIR",
        help=f"where the evaluation lands (default: DIR/{EVALUATION_DIR}; "
             "required when pooling several directories)")
    parser.add_argument(
        "--reading-seconds", type=float, default=DEFAULT_READING_SECONDS, metavar="S",
        help="length of one blood-pressure reading: 30 is the ISO 81060-2 invasive "
             "reference interval, about 10 the ISO 81060-3 device segment (default: 30)")
    parser.add_argument(
        "--change-seconds", type=float, default=DEFAULT_CHANGE_SECONDS, metavar="S",
        help="the ISO 81060-3 change evaluation interval: two readings at most this "
             "far apart form a change event (default: 60)")
    parser.add_argument(
        "--report-recordings", type=int, default=DEFAULT_REPORT_RECORDINGS, metavar="N",
        help="how many recordings, evenly spaced, get per-recording pages in the "
             "report (default: 4)")
    return parser


def main(argv=None) -> Path:
    """Evaluate; returns the evaluation directory."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        records_dirs = find_records_dirs(args.dirs)
    except FileNotFoundError as err:
        parser.error(str(err))
    if args.out is None and len(records_dirs) > 1:
        parser.error("--out is required when pooling several directories")
    out_dir = Path(args.out) if args.out else records_dirs[0] / EVALUATION_DIR
    try:
        pool = load_pool(records_dirs, args.reading_seconds)
    except ValueError as err:
        parser.error(str(err))

    categories = participant_categories(pool.readings)
    summary = summary_table(pool.readings, pool.rates, categories)
    criteria, changes = criteria_table(pool.readings, categories, args.change_seconds)
    coverage = coverage_table(pool.readings)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("readings.csv", pool.readings), ("beats.csv", pool.beats),
                        ("rates.csv", pool.rates), ("summary.csv", summary),
                        ("criteria.csv", criteria), ("coverage.csv", coverage),
                        ("changes.csv", changes)):
        frame.to_csv(out_dir / name, index=False, float_format=FLOAT_FORMAT)

    figures = aggregate_figures(pool, categories, changes)
    per_recording = [figure for folder, meta, tags in evenly_spaced(pool.folders, args.report_recordings)
                     for figure in recording_figures(folder, meta, tags, pool)]
    write_report(out_dir, pool, criteria, coverage, summary, figures, per_recording,
                 args.reading_seconds)
    print(f"evaluation written to {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Replace `src/evaluation/__init__.py`**

```python
"""The evaluation of a run's records, with no config and no torch.

Three layers, each reading only what the one before it wrote:
``recording.py`` (with ``beats.py`` and ``rate.py``) scores one recording
and camera into beats, readings and heart rates; ``aggregate.py`` pools
any number of records directories into the summary, the standards'
criteria and the coverage tables; ``plots.py`` and ``report.py`` draw the
figures and write the PDF report and the digest. ``post_process.py`` is
the upstream toolbox's detrend / bandpass / SNR / MACC helpers the
estimator uses. ``scripts/eval.py`` is the entry point;
``docs/evaluation.md`` the reference.
"""
```

- [ ] **Step 3: Delete the legacy files and move the `to_physical` test**

```bash
git rm -q src/evaluation/evaluate.py src/evaluation/records.py tests/test_evaluate.py
```

(`git rm` because they are tracked; the deletion is staged like the rest
of the user's work. If `git rm` refuses because a file has staged
changes, use `git rm -f`.)

Create `tests/test_to_physical.py`:

```python
"""``to_physical`` inverts the label normalisation exactly and leaves raw alone."""
import numpy as np
import torch

from src.trainer import to_physical


def _record():
    t = torch.arange(180, dtype=torch.float32) / 30
    abp = 100 + 20 * torch.sin(2 * np.pi * 1.2 * t)
    ecg = 5 * torch.sin(2 * np.pi * 1.2 * t)

    def stats(x):
        return {"mean": x.mean(), "std": x.std(), "min": x.amin(), "max": x.amax()}

    return {"predictions": {"ABP": abp, "ECG": ecg}, "labels": {"ABP": abp, "ECG": ecg},
            "label_stats": {"ABP": stats(abp), "ECG": stats(ecg)}}


def test_to_physical_inverts_zscore_and_leaves_raw():
    record = _record()
    stats = record["label_stats"]["ECG"]
    z = (record["labels"]["ECG"] - stats["mean"]) / stats["std"]
    normed = {**record, "labels": {**record["labels"], "ECG": z}}
    physical = to_physical(normed, {"ABP": "raw", "ECG": "zscore"})
    torch.testing.assert_close(physical["labels"]["ECG"], record["labels"]["ECG"],
                               rtol=1e-4, atol=1e-3)
    assert physical["labels"]["ABP"] is record["labels"]["ABP"]
    assert physical["label_stats"] is record["label_stats"]
```

- [ ] **Step 4: Add the eval step to the chain test**

In `tests/test_scripts.py`, add after `from scripts.infer import main as infer`:

```python
from scripts.eval import EVALUATION_DIR, main as evaluate
```

and append to the end of `test_train_writes_the_run_and_infer_rebuilds_it`
(after the `assert not (out / CONFIG_NAME).exists() ...` line):

```python
    # The records alone: layer one beside each recording, the evaluation
    # beside the records. 32 covered frames at 30 fps cut into 0.5 s
    # readings gives two; the two-frame remainder is dropped.
    out = evaluate([str(run_dir), "--reading-seconds", "0.5"])
    assert out == run_dir / RECORDS_DIR / EVALUATION_DIR
    for name in ("readings.csv", "beats.csv", "rates.csv", "summary.csv", "criteria.csv",
                 "coverage.csv", "changes.csv", "report.pdf", "digest.txt"):
        assert (out / name).is_file(), name
    readings = pd.read_csv(out / "readings.csv", dtype={"participant": str})
    assert set(readings["signal"]) == {"ABP", "CVP"}
    assert sorted(readings["reading"].unique()) == [0, 1]
    assert set(readings["participant"]) == {"003"} and readings["waveform_mad"].notna().all()
    assert (run_dir / RECORDS_DIR / "P003_S01_R1_0_D" / "1" / "readings.csv").is_file()
    assert "=== Criteria" in (out / "digest.txt").read_text(encoding="utf-8")
```

Update the module docstring's first line to "The train -> infer -> eval
chain, end to end over a synthetic zarr cache on the CPU." and add to its
last sentence "and ``eval`` has to score the records from the directory
alone."

- [ ] **Step 5: Run the tests that cover the touched files**

```bash
uv run pytest -p no:faulthandler tests/test_evaluation.py tests/test_to_physical.py tests/test_scripts.py tests/test_post_process.py -q
uv run pytest --collect-only -q -p no:faulthandler 2>&1 | tail -3
grep -rn "torch" src/evaluation/ ; echo "(torch grep above must be empty)"
grep -rn "src.evaluation.evaluate\|evaluation.records\|test_records\.pt\|window_rates" src scripts tests tools --include=*.py
```

Expected: all pass; collection reports no errors; the torch grep prints
nothing; the last grep prints nothing.

- [ ] **Step 6: Run the real thing**

```bash
uv run python scripts/eval.py runs/BIGSMALL_PURE.01_202609091150
uv run python scripts/eval.py runs/BIGSMALL_PURE.01_202609091150 runs/PHYSNET_PURE.01_202609091224 --out scratch_pool --reading-seconds 10
ls runs/BIGSMALL_PURE.01_202609091150/test_records/evaluation scratch_pool
rm -r scratch_pool
```

Expected: the digest prints, the evaluation directory holds the seven
CSVs, `report.pdf`, `digest.txt` and `HR_PPG_bland_altman.pdf`; the pooled
run reports 12 recordings and 36 readings per signal at 10 s. Leave the
first evaluation directory in place; it is under `runs/`, which git
ignores.

---

### Task 9: Docs

**Files:**
- Modify: `README.md` lines 84-100 (steps 2 and 3, the stub sentence) and lines 142-152 (the output paragraph)
- Modify: `docs/adding_a_model.md` line 404 (the third command), lines 428-430 (step 3), and the two table rows for `test_records/` and the evaluation outputs
- Modify: `docs/plans/2026-09-08-model-migrations.md` line 97
- Create: `docs/evaluation.md`

- [ ] **Step 1: README**

Replace the step 3 line and comment in the three-command block with:

```bash
# 3. score the records against the clinical standards; writes test_records/evaluation/ (tables, figures, report.pdf, digest.txt)
uv run python scripts/eval.py runs/PHYSNET_PURE.01_<YYYYMMDDHHMM>
```

Delete the two sentences "`scripts/eval.py` is a stub for now; step 3 is
the evaluation package's own entry point until the script is shaped."
Keep "Add `--limit-windows 8` to steps 1 and 2 for a wiring check."

Replace the paragraph starting "The evaluation scores every absolute
signal on its level" with:

```markdown
The evaluation (`scripts/eval.py`) cuts each recording's combined trace
into readings (30 s by default), detects the beats of every cardiac trace
on both the label and the prediction, and scores per reading the systolic
/ MAP / diastolic levels of the absolute signals, the per-sample waveform
agreement, and a heart rate from every cardiac trace, their fused spectrum
and their median. `test_records/evaluation/` then holds the pooled tables,
`summary.csv` per grouping (all, dataset, participant, recording, blood-
pressure category), `criteria.csv` with the ISO 81060-2, ISO 81060-3 and
IEEE 1708 statistics against their limits, `coverage.csv`, the figures and
`report.pdf`; `digest.txt` is the readable summary. Several records
directories pool into one evaluation with `--out`, which is how a LOSO
sweep is reported. [docs/evaluation.md](docs/evaluation.md) lists every
column and the clause it comes from.
```

- [ ] **Step 2: `docs/adding_a_model.md`**

Line 404: replace `uv run python -m src.evaluation.evaluate runs/MYNET_NECKFLIX.1_<YYYYMMDDHHMM>`
with `uv run python scripts/eval.py runs/MYNET_NECKFLIX.1_<YYYYMMDDHHMM> --reading-seconds 0.5`
and add the sentence after the block: "With `--limit-windows 8` the
covered trace is a few seconds long, so shrink the reading for the wiring
check; drop both flags for a real fold."

Step 3 in "What happens, in order": replace with

```markdown
3. `scripts/eval.py`: reads `test_records/` alone, writes `beats.csv`,
   `readings.csv` and `rates.csv` beside each recording's trace tables,
   then the pooled tables, the standards' criteria, the figures and
   `report.pdf` under `test_records/evaluation/` (`docs/evaluation.md`).
```

In the outputs table replace the last row (`windows.csv`, `rates.csv`, ...)
with:

```markdown
| `test_records/<recording>/<camera>/{beats,readings,rates}.csv` | eval, layer one | per reference beat its matched predicted beat and both beats' levels; per reading the beat counts, the level means / SDs / errors and the waveform agreement; per reading a heart rate per source |
| `test_records/evaluation/` | eval | the pooled tables, `summary.csv`, `criteria.csv`, `coverage.csv`, `changes.csv`, the figures, `report.pdf`, `digest.txt` |
```

- [ ] **Step 3: `docs/plans/2026-09-08-model-migrations.md` line 97**

Replace `` `config.yaml`, `losses.csv`, `test_records.pt`, `rates.csv`, `summary.csv` ``
with `` `config.yaml`, `losses.csv`, `test_records/` and its `evaluation/` ``.

- [ ] **Step 4: Write `docs/evaluation.md`**

```markdown
# Evaluation

`scripts/eval.py DIR [DIR ...] [--out DIR] [--reading-seconds 30] [--change-seconds 60] [--report-recordings 4]`

Each `DIR` is a records directory `scripts/infer.py` wrote, or a run
directory as shorthand for its `test_records/`. The evaluation reads
`meta.json` and the per-recording trace tables and nothing else: no
config, no checkpoint, no torch, so it runs on a laptop over CSVs copied
from the cluster. Output goes to `DIR/evaluation/`, or `--out`, which is
required when pooling several directories. Clause and page numbers below
are the printed ones in `standards/`.

## The prediction being scored

The trace table's `mean` column: the average of every strided window
covering the frame. Its `std` is the repeatability across windows and
`label` the reference in physical units. Averaging overlapping windows
smooths the prediction; the per-window columns stay in the trace tables
for anyone who wants the unsmoothed one. Errors are prediction minus
reference everywhere, the sign every standard uses.

## Layer one: per recording and camera

Written beside the trace tables, remade when missing or when the reading
length changes.

**Readings.** Non-overlapping stretches of `--reading-seconds` from the
first covered frame; a trailing remainder shorter than half a reading is
dropped. 30 s is the ISO 81060-2 invasive reference interval (clause 6.2.4
b), p. 21); about 10 s is the ISO 81060-3 device segment (clause 5.1.3,
p. 14, and A.2, p. 27); IEEE 1708 records 60 s (clause 4.4.2, p. 24).

**Beats** (`src/evaluation/beats.py`). One detector for every cardiac
trace: detrend and bandpass to 0.6 to 3.3 Hz, `find_peaks` with the
minimum beat distance from the top of the band and the prominence a
registry fraction of the cleaned range (`beat` in
`src/signal_transforms.py`), each candidate moved to the raw extremum
within a quarter of the median beat interval. A beat spans trough to
trough: `max` is its peak, `min` the lower trough, `mean` the area under
the curve over the duration, the MAP definition of ISO 81060-2 clause
6.2.4 e), p. 22. Predicted beats match the nearest reference beat within
40 percent of the median reference interval, closest first, one to one.

`beats.csv`: `signal`, `beat`, `t_ref`, `t_pred` (blank on a miss),
`ref_max`, `ref_mean`, `ref_min`, `pred_max`, `pred_mean`, `pred_min`
(blank for shape-class signals).

`readings.csv`, per reading and signal: `t_start`, `t_end`;
`n_ref_beats`, `n_pred_beats`, `n_matched` (recall is matched over
reference, precision matched over predicted); for absolute signals
`ref_<s>_mean`, `ref_<s>_sd`, `pred_<s>_mean`, `pred_<s>_sd` over the
beats for `s` in `max`, `mean`, `min` (systolic, MAP, diastolic for ABP;
peak, mean, trough for CVP; for a non-cardiac absolute signal the sample
mean, SD, max and min), `err_<s>` and `err_<s>_deadband` (ISO 81060-2
clause 6.2.5, p. 22: zero inside the reference mean ± SD, else the
distance to the nearer limit); `waveform_mad`, `waveform_rmse`,
`waveform_r`, `waveform_ccc` over the reading's samples (IEEE 1708
equations (3) and (4), p. 28).

`rates.csv`, per reading and source: `ref_hr`, `pred_hr`, `err_hr`,
`snr`, `macc`; sources are each cardiac trace, `FUSED` (geometric mean of
the normalised spectra) and `MEDIAN` when a reading carries two or more.
SNR and MACC are on the combined trace and are not comparable with the
upstream toolbox's per-window numbers.

## Layer two: the pool

`readings.csv`, `beats.csv`, `rates.csv` are the layer-one tables tagged
with `dataset`, `participant`, `recording`, `perspective`.

`summary.csv`: `group_by` (all, dataset, participant, recording,
bp_category), `group`, `signal`, `statistic` (`max` / `mean` / `min` for
absolute signals, `waveform`, `hr`), `metric`, `value`, `n`. Level and
heart-rate metrics: `bias`, `sd`, `loa_low`, `loa_high`, `mad`, `mapd`,
`rmse`, `cp5`, `cp10`, `cp15`, `pearson`, `ccc` (IEEE 1708 clause 4.6.2,
p. 32, names MAD, MAPD, MD, SD and CP_L); heart rate adds `snr`, `macc`;
waveform rows are the mean over readings of the four waveform columns.
The blood-pressure category is the participant's, from their mean
reference systolic and diastolic, by IEEE 1708 Table 3, p. 22.

`criteria.csv` (ABP only): `standard`, `clause`, `measurand`, `group_by`,
`group`, `metric`, `value`, `limit`, `pass`, `note`. A blank `pass` means
the standard fixes a count we report but cannot meet by design.

| Standard, clause | Metric | Limit |
| --- | --- | --- |
| ISO 81060-2 5.2.4.1.2 a) criterion 1, and 6.2.6 with the 6.2.5 dead band | pooled `bias`, `sd` | ±5.0, 8.0 mmHg |
| ISO 81060-2 5.2.4.1.2 b) criterion 2, Table 1 | `sd_of_subject_means` about the pooled mean | Table 1 by the pooled mean (cuff route only) |
| ISO 81060-3 5.1.4 Type A | `bias`, `s_corr`, `n_ind`; `icc`, `subjects`, `readings` reported | ±6.0, 10.0 mmHg, 278; at least 30 subjects |
| ISO 81060-3 5.2.4 b) Type T | `s_corr`, `n_ind` after removing each participant's offset from their first recording | 6.0 mmHg, 278 |
| ISO 81060-3 5.3.5 | `p50_mean`, `p85_mean` of per-participant change-tracking error; `events`, `min_events_per_subject` reported | 25 %, 50 %; at least 50 events per subject |
| IEEE 1708 4.5.3.1 Table 6 | `mad` with `bias`, grade A/B/C/D in `note`; worst cell overall | A ≤ 5; B ≤ 6 and bias ≤ 5; C ≤ 7 and bias ≤ 5; D beyond |
| IEEE 1708 4.5.3.4 | `mad` per blood-pressure category (not stage 2) | 6.0 mmHg |
| IEEE 1708 4.5.2 and Table 7 | waveform `mad`, `r`, grade in `note` | 7.0 mmHg, 0.7 |

`s_corr`, `icc` and `n_ind` are ISO 81060-3 formulas (5), (6), (9) to
(12), pp. 12-15, in their unequal-count form. Change events (`changes.csv`)
are pairs of readings in one recording at most `--change-seconds` apart
where the reference or predicted change reaches 15 mmHg systolic, 12 MAP
or 10 diastolic (clause 5.3.2, p. 20), scored by formula (18), p. 22.

`coverage.csv`: the share of reference ABP readings in each band against
ISO 81060-2 clause 6.1.5, pp. 18-19, ISO 81060-3 clause 4.3.3, pp. 7-8,
and the IEEE 1708 Table 5, p. 27, change-from-baseline bins, the baseline
being each participant's first reading (calibration-free, clause 4.4.2,
p. 24).

## Layer three: figures and report

Aggregate figures, also saved individually: Bland-Altman and identity
scatter per level coloured by participant and by recording; the reference
histogram and reference against time since the recording start (ISO
81060-3 clause 5.1.3 f), p. 14); for ABP the error against change from
baseline and the change histogram (IEEE 1708 clause 4.6.3, p. 35), the
reference-change and change-error histograms (ISO 81060-3 clause 5.3.4,
p. 22); a heart-rate Bland-Altman per source. Per sampled recording
(`--report-recordings`): the ribbon (prediction ± SD across windows over
the label), the beat overlay (matched pairs joined, misses marked), the
per-beat scatter, and the trend of readings over time (ISO 81060-3 figure
A.3, p. 30). `report.pdf` is the digest's pages followed by every figure;
`digest.txt` is printed and written.

## Not covered

ESH 2023 and ISO 81060-1. Calibration and time-since-initialisation
beyond the baseline-first-reading convention. CVP beats are detected with
the shared detector and are expected to be unreliable; the recall and
precision columns say whether they are.
```

- [ ] **Step 5: Verify the docs reference nothing stale**

```bash
grep -rn "src.evaluation.evaluate\|test_records.pt\|is a stub" README.md docs/ scripts/ src/ tests/
```

Expected: nothing.

---

### Task 10: Final verification

**Files:** none new.

- [ ] **Step 1: The full picture**

```bash
uv run pytest -p no:faulthandler tests/test_evaluation.py tests/test_to_physical.py tests/test_scripts.py tests/test_signal_transforms.py tests/test_post_process.py -q
uv run pytest --collect-only -q -p no:faulthandler 2>&1 | tail -3
uv run python scripts/eval.py runs/BIGSMALL_PURE.01_202609091150 --reading-seconds 30
grep -rn "torch" src/evaluation/
git status --short | grep -v "^M  \|^D  \|^A  \|^R  " 
```

Expected: tests pass, collection clean, the digest prints, the torch grep
is empty, and `git status` shows exactly the files this plan named as
unstaged changes (`??` for the new files, ` M` for edits): `pyproject.toml`,
`uv.lock`, `src/signal_transforms.py`, `src/evaluation/__init__.py`,
`src/evaluation/rate.py`, `src/evaluation/plots.py`, `scripts/eval.py`,
`tests/test_scripts.py`, `README.md`, `docs/adding_a_model.md`,
`docs/plans/2026-09-08-model-migrations.md`, plus the new
`src/evaluation/beats.py`, `recording.py`, `aggregate.py`, `report.py`,
`tests/test_evaluation.py`, `tests/test_to_physical.py`,
`docs/evaluation.md`, and the three `git rm` deletions.

- [ ] **Step 2: Report to the user**

List: the files changed, the two commands run in Step 1 with their
output tails, the digest of the BigSmall run, and that nothing was
committed. Name any test that failed elsewhere in the suite and was
deleted as stale under the `CLAUDE.md` rule, so a human can veto.
