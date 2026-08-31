# Evaluation Rebuild (Phase 7) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the HR-shaped evaluation path with a layered, per-signal
package that scores pressure waveforms at beat level and reports ISO 81060-3
and IEEE 1708 agreement.

**Architecture:** Window records flow through one library —
`records → beats → levels → metrics → report` — producing a single tidy
DataFrame (`level × unit × signal × metric × value × se`) that the digest, the
CSV, the JSON and every plot are views over. The trainer and the sweep
reporter are thin callers of the same functions, differing only in which
levels of the hierarchy they can populate.

**Tech Stack:** Python 3.13, PyTorch, NumPy, pandas, SciPy, einops, matplotlib,
pytest, `uv`.

**Spec:** [docs/plans/2026-08-31-evaluation-clinical-metrics.md](2026-08-31-evaluation-clinical-metrics.md)

## Global Constraints

- **Dependencies go through `uv add`, never pip.** This plan adds none.
- **All tensor reshaping uses einops** (`rearrange` / `reduce` / `einsum`),
  never `view` / `permute` / `reshape`. This applies to code lifted from the
  prototype notebooks — port it to einops as you lift it.
- **Testing stays minimal.** Four new tests total, replacing
  `tests/test_metrics_report.py`. Do not add tests opportunistically.
- **Legacy code is deleted, not adapted.** No compatibility shims.
- **Do not touch `evaluation/metrics.py` or
  `evaluation/bigsmall_multitask_metrics.py`.** They are imported by the seven
  legacy trainers and pinned by `tests/test_legacy_contract.py`; they die in
  Phase 6.
- **Do not touch `evaluation/post_process.py`.** It is the DSP layer and
  `unsupervised_methods/` depends on it.
- Commits are conventional-commit style (`feat:`, `refactor:`, `docs:`).
- Run the full suite with `uv run pytest -q` before each commit; it is green at
  289 tests today.

---

### Task 1: Carry the store's root attrs through the batch contract

The hierarchy groups by participant, posture and session. Per-sample metadata
carries only `recording_id` / `camera_id` / `start_frame`, so the prototypes
recovered the rest by string-splitting `P015_S01_R3_0_D`. Add one generic
`attrs` map instead. This edits the shared batch contract, so it lands alone
and first.

**Files:**
- Modify: `neural_methods/batch.py:14-43` (docstring + key constants)
- Modify: `dataset/data_loader/zarr_dataset.py:706-720` (`__getitem__` return)
- Test: `tests/test_batch_contract.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `neural_methods.batch.ATTRS = "attrs"`; every sample's
  `metadata["attrs"]` is a `dict[str, str]` of the store's scalar root attrs.
  After `default_collate` it is `dict[str, list[str]]`, and `iter_samples`
  already indexes that correctly via `_index_sample`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_batch_contract.py`:

```python
def test_metadata_carries_store_attrs_as_strings():
    """Grouping attributes ride in metadata, not in the recording id's spelling."""
    from neural_methods.batch import ATTRS, METADATA, iter_samples

    sample = {
        "frames": {"G": torch.zeros(1, 4, 2, 2)},
        "labels": {"ABP": torch.zeros(4)},
        "label_stats": {"ABP": {"mean": torch.zeros(())}},
        "channel_mask": {"G": torch.tensor(True)},
        "label_mask": {"ABP": torch.tensor(True)},
        METADATA: {
            "recording_id": "P015_S01_R3_0_D",
            "camera_id": "1",
            "start_frame": 0,
            ATTRS: {"participant": "015", "posture": "0", "light": "D"},
        },
    }
    batch = torch.utils.data.default_collate([sample, sample])
    first = next(iter_samples(batch))
    assert first[METADATA][ATTRS] == {"participant": "015", "posture": "0",
                                      "light": "D"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_batch_contract.py::test_metadata_carries_store_attrs_as_strings -v`
Expected: FAIL with `ImportError: cannot import name 'ATTRS'`.

- [ ] **Step 3: Add the key and emit it**

In `neural_methods/batch.py`, beside the other metadata sub-keys:

```python
RECORDING_ID = "recording_id"
CAMERA_ID = "camera_id"
START_FRAME = "start_frame"
#: The store's own scalar root attrs, stringified. Dataset-agnostic: whatever
#: attrs a store carries, not a fixed Neckflix key list. This is what the
#: evaluation hierarchy groups by (participant, posture, session, ...).
ATTRS = "attrs"
```

Update the module docstring's per-sample block so the contract stays the
documentation:

```python
     "metadata":     {"recording_id": str, "camera_id": str, "start_frame": int,
                      "attrs": {str: str}}}
```

In `dataset/data_loader/zarr_dataset.py`, in `__getitem__`, replace the
returned `metadata` dict:

```python
        recording_id = root.attrs.get("recording", rec_name)
        # Scalars only, stringified: default_collate turns a dict of strings
        # into a dict of lists, which iter_samples already indexes. Nested or
        # array-valued attrs have no meaning as a grouping key.
        store_attrs = {
            str(key): str(value)
            for key, value in root.attrs.items()
            if isinstance(value, (str, int, float, bool))
        }

        return {
            "frames": frames,
            "labels": labels,
            "label_stats": label_stats,
            "channel_mask": channel_mask,
            "label_mask": label_mask,
            "metadata": {
                "recording_id": recording_id,
                "camera_id": camera_id,
                "start_frame": start,
                "attrs": store_attrs,
            },
        }
```

Update that method's docstring line to end `... start_frame / attrs.`

- [ ] **Step 4: Run the test and the suite**

Run: `uv run pytest tests/test_batch_contract.py -v && uv run pytest -q`
Expected: the new test PASSES; the suite stays green (289 + 1).

- [ ] **Step 5: Commit**

```bash
git add neural_methods/batch.py dataset/data_loader/zarr_dataset.py tests/test_batch_contract.py
git commit -m "feat(batch): carry the store's root attrs in per-sample metadata"
```

---

### Task 2: `evaluation/records.py` — one loader, one physical-unit inversion

The inversion from normalised space to physical units is currently written
three times. Collapse it here, and give both record sources (in-memory and
pickle) one shape.

**Files:**
- Create: `evaluation/records.py`
- Test: `tests/test_evaluation_records.py`

**Interfaces:**
- Consumes: `neural_methods.batch.ATTRS` (Task 1);
  `dataset.data_loader.label_transforms.INVERSES`.
- Produces:
  - `WindowRecord(signal, recording_id, camera_id, start_frame, prediction, label, attrs)`
    — a frozen dataclass; `prediction` and `label` are `np.ndarray` float64 in
    **physical units**.
  - `RunRecords(windows, fs, traces, label_norms)` — frozen dataclass with
    `signals() -> list[str]`.
  - `to_physical(values, stats, mode) -> np.ndarray`
  - `from_saved(raw_windows, *, fs, traces, label_norms) -> RunRecords`
  - `load(target) -> RunRecords` where `target` is a pickle path, a directory
    to search, or an iterable of either.

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluation_records.py`:

```python
"""The records layer: one physical-unit inversion, two equivalent sources."""
import pickle

import numpy as np

from evaluation.records import RunRecords, from_saved, load, to_physical


def _raw_window(signal="ABP", start=0):
    return {
        "signal": signal,
        "recording_id": "P015_S01_R3_0_D",
        "camera_id": "1",
        "start_frame": start,
        "prediction": np.linspace(80.0, 120.0, 16, dtype=np.float32),
        "label": np.linspace(78.0, 118.0, 16, dtype=np.float32),
        "label_stats": {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0},
        "attrs": {"participant": "015", "posture": "0"},
    }


def test_raw_mode_inverts_by_identity():
    values = np.array([80.0, 120.0])
    stats = {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0}
    assert np.allclose(to_physical(values, stats, "raw"), values)


def test_zscore_mode_uses_the_window_stats():
    values = np.array([-1.0, 0.0, 1.0])
    stats = {"mean": 90.0, "std": 10.0, "min": 80.0, "max": 100.0}
    assert np.allclose(to_physical(values, stats, "zscore"), [80.0, 90.0, 100.0])


def test_memory_and_pickle_sources_agree(tmp_path):
    """The trainer's in-memory records and its own pickle must load identically."""
    raw = [_raw_window(start=0), _raw_window(start=16)]
    meta = dict(fs=30.0, traces=("ABP",), label_norms={"ABP": "raw"})
    in_memory = from_saved(raw, **meta)

    path = tmp_path / "run_outputs.pickle"
    with open(path, "wb") as handle:
        pickle.dump({"windows": raw, "channels": ["G"], **meta,
                     "traces": ["ABP"]}, handle)
    from_disk = load(tmp_path)

    assert isinstance(in_memory, RunRecords) and isinstance(from_disk, RunRecords)
    assert len(in_memory.windows) == len(from_disk.windows) == 2
    assert in_memory.signals() == from_disk.signals() == ["ABP"]
    for a, b in zip(in_memory.windows, from_disk.windows):
        assert np.allclose(a.prediction, b.prediction)
        assert a.attrs == b.attrs == {"participant": "015", "posture": "0"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluation_records.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.records'`.

- [ ] **Step 3: Write the implementation**

Create `evaluation/records.py`:

```python
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
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_evaluation_records.py -v`
Expected: all three PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/records.py tests/test_evaluation_records.py
git commit -m "feat(eval): records layer with the single physical-unit inversion"
```

---

### Task 3: `evaluation/beats.py` — the PhysHydra peak picker and the beat table

Lift the detector the PhysHydra-era analysis used, port it to einops, fix one
latent bug in it, and build the reference-anchored beat table on top.

**Files:**
- Create: `evaluation/beats.py`
- Modify: `neural_methods/signals.py` (add `beat_labels` to the registry)
- Test: `tests/test_evaluation_beats.py`

**Interfaces:**
- Consumes: `evaluation.records.WindowRecord` (Task 2).
- Produces:
  - `find_peaks(trace, kind="max", width=31, clip_ends=False) -> (idx, values)`
  - `BEAT_STATS = ("max", "mean", "min")`
  - `beat_intervals(reference, fs) -> list[tuple[int, int]]`
  - `beat_stats(trace, intervals) -> dict[str, np.ndarray]` keyed by `BEAT_STATS`
  - `neural_methods.signals.beat_labels(signal) -> dict[str, str]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluation_beats.py`:

```python
"""The beat layer, against a synthetic arterial waveform of known shape."""
import numpy as np
import pytest

from evaluation.beats import BEAT_STATS, beat_intervals, beat_stats, find_peaks
from neural_methods.signals import beat_labels

FS = 30.0
BEATS_PER_SECOND = 1.2
SECONDS = 10.0
SYSTOLIC, DIASTOLIC = 120.0, 80.0


def synthetic_abp():
    """A clean pulsatile trace peaking at 120 and troughing at 80 mmHg."""
    t = np.arange(int(FS * SECONDS)) / FS
    wave = -np.cos(2 * np.pi * BEATS_PER_SECOND * t)      # starts at a trough
    midpoint, amplitude = (SYSTOLIC + DIASTOLIC) / 2, (SYSTOLIC - DIASTOLIC) / 2
    return midpoint + amplitude * wave


def test_find_peaks_recovers_the_pulse_rate():
    trace = synthetic_abp()
    idx, values = find_peaks(trace, kind="max", width=int(FS * 2 / 3))
    assert len(idx) == pytest.approx(BEATS_PER_SECOND * SECONDS, abs=1)
    assert values.max() == pytest.approx(SYSTOLIC, abs=1.0)


def test_even_width_is_made_odd_before_pooling():
    """The prototype incremented width after using it, so the fix must bite."""
    trace = synthetic_abp()
    assert np.array_equal(find_peaks(trace, width=20)[0],
                          find_peaks(trace, width=21)[0])


def test_beat_stats_recover_systolic_and_diastolic():
    trace = synthetic_abp()
    intervals = beat_intervals(trace, FS)
    assert len(intervals) >= 8
    stats = beat_stats(trace, intervals)
    assert set(stats) == set(BEAT_STATS)
    assert stats["max"].mean() == pytest.approx(SYSTOLIC, abs=2.0)
    assert stats["min"].mean() == pytest.approx(DIASTOLIC, abs=2.0)
    assert stats["mean"].mean() == pytest.approx((SYSTOLIC + DIASTOLIC) / 2, abs=2.0)


def test_beat_labels_are_per_signal():
    assert beat_labels("ABP")["max"] == "systolic"
    assert beat_labels("CVP")["max"] == "peak"
    assert beat_labels("CVP")["mean"] == "mean"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluation_beats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.beats'`.

- [ ] **Step 3: Add `beat_labels` to the signal registry**

In `neural_methods/signals.py`, add a `beat_labels` entry to each absolute
signal's dict and a reader beside `signal_unit`:

```python
    'ABP':  {'norm': (0.0, 200.0),
             'class': ABSOLUTE, 'unit': 'mmHg', 'prior': 90.0, 'scale': 20.0,
             'beat_labels': {'max': 'systolic', 'mean': 'MAP', 'min': 'diastolic'}},
    'CVP':  {'norm': (-20.0, 30.0),
             'class': ABSOLUTE, 'unit': 'mmHg', 'prior': 8.0,  'scale': 5.0,
             # CVP has no systole: its waveform is a/c/v waves, and the
             # quantity that matters clinically is the mean. The machinery is
             # shared with ABP; only the wording differs.
             'beat_labels': {'max': 'peak', 'mean': 'mean', 'min': 'trough'}},
    'SPO2': {'norm': (0.0, 100.0),
             'class': ABSOLUTE, 'unit': '%',    'prior': 97.0, 'scale': 3.0,
             'beat_labels': {'max': 'max', 'mean': 'mean', 'min': 'min'}},
```

```python
def beat_labels(sig) -> dict:
    """How this signal's per-beat max/mean/min are named in a report.

    Shape-class signals get no beat treatment, so they fall back to the plain
    words rather than borrowing arterial vocabulary.
    """
    entry = SIGNALS[canonical_signal(sig)]
    return dict(entry.get('beat_labels',
                          {'max': 'max', 'mean': 'mean', 'min': 'min'}))
```

- [ ] **Step 4: Write `evaluation/beats.py`**

```python
"""Reference-anchored beats: the clock, the intervals, the per-beat statistics.

The detector is the one the PhysHydra-era analysis used (the mature copy in
``evaluation/prototypes/neckflix_metrics.ipynb``, the one carrying
``clip_ends``): non-maximum suppression over a sliding window, which enforces
a minimum beat separation directly. Two changes on the way in — the reshapes
are einops per the repo rule, and the odd-width normalisation now happens
*before* pooling rather than after, where it had no effect.
"""

from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F

#: What is read off each beat. Uniform across absolute signals — only the
#: display wording is per-signal (``signals.beat_labels``).
BEAT_STATS = ("max", "mean", "min")

#: Sliding-window width as a fraction of the sampling rate: 2/3 s, the setting
#: the PhysHydra analysis ran at. Wide enough to suppress a dicrotic notch,
#: narrow enough to keep every beat up to ~90 bpm.
WIDTH_SECONDS = 2 / 3


def find_peaks(trace, kind="max", width=31, clip_ends=False):
    """Local extrema that survive non-maximum suppression of ``width`` samples.

    Returns ``(indices, values)``. A point is kept when it is the extremum of
    its own neighbourhood, which makes ``width`` a minimum-separation
    constraint rather than a smoothing parameter.
    """
    if kind not in ("max", "min"):
        raise ValueError(f"kind must be 'max' or 'min', got {kind!r}")
    if width % 2 == 0:
        width += 1          # symmetric padding needs an odd kernel
    tensor = torch.as_tensor(np.asarray(trace), dtype=torch.float32)
    work = tensor if kind == "max" else -tensor
    _, indices = F.max_pool1d_with_indices(
        rearrange(work, "t -> 1 1 t"), kernel_size=width, stride=1,
        padding=width // 2)
    indices = rearrange(indices, "1 1 t -> t")
    candidates = indices.unique()
    kept = candidates[indices[candidates] == candidates]
    idx = np.atleast_1d(kept.numpy().astype(int))
    values = np.atleast_1d(tensor.numpy()[idx].astype(np.float64))
    if clip_ends and len(values) > 3:
        interior_mean = float(values[1:-1].mean())
        interior_std = float(values[1:-1].std())
        if abs(values[0] - interior_mean) > 2 * interior_std:
            idx, values = idx[1:], values[1:]
        if abs(values[-1] - interior_mean) > 2 * interior_std:
            idx, values = idx[:-1], values[:-1]
    return idx, values


def beat_intervals(reference, fs) -> list:
    """Foot-to-foot beat boundaries, detected on the reference trace only.

    Anchoring on the reference means every reference beat yields exactly one
    comparison, so agreement statistics carry no selection bias from a
    prediction whose beats are hard to find.
    """
    width = max(3, int(fs * WIDTH_SECONDS))
    feet, _ = find_peaks(reference, kind="min", width=width, clip_ends=True)
    return [(int(start), int(end)) for start, end in zip(feet[:-1], feet[1:])
            if end - start >= 3]


def beat_stats(trace, intervals) -> dict:
    """``max`` / ``mean`` / ``min`` of ``trace`` inside each beat interval."""
    values = np.asarray(trace, dtype=np.float64)
    beats = [values[start:end] for start, end in intervals]
    if not beats:
        return {name: np.array([], dtype=np.float64) for name in BEAT_STATS}
    return {
        "max": np.array([beat.max() for beat in beats]),
        "mean": np.array([beat.mean() for beat in beats]),
        "min": np.array([beat.min() for beat in beats]),
    }
```

- [ ] **Step 5: Run the tests and commit**

Run: `uv run pytest tests/test_evaluation_beats.py -v && uv run pytest -q`
Expected: all four PASS; suite green.

```bash
git add evaluation/beats.py neural_methods/signals.py tests/test_evaluation_beats.py
git commit -m "feat(eval): reference-anchored beat layer on the PhysHydra peak picker"
```

---

### Task 4: Beat detection quality, with the amplitude gate

The detector has no amplitude gate, so on a flat or noise-only trace it returns
one "peak" per window rather than none. Harmless when anchored on a real
arterial line; wrong here, where "did the model produce beats at all" is the
question.

**Files:**
- Modify: `evaluation/beats.py`
- Test: `tests/test_evaluation_beats.py`

**Interfaces:**
- Consumes: `find_peaks`, `beat_intervals` (Task 3).
- Produces: `detection_quality(prediction, reference, fs, tolerance_seconds=0.15)
  -> {"sensitivity": float, "ppv": float, "ibi_error": float, "n_reference": int,
  "n_detected": int}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evaluation_beats.py`:

```python
def test_detection_quality_is_perfect_on_the_reference_itself():
    trace = synthetic_abp()
    quality = detection_quality(trace, trace, FS)
    assert quality["sensitivity"] == pytest.approx(1.0)
    assert quality["ppv"] == pytest.approx(1.0)
    assert quality["ibi_error"] == pytest.approx(0.0, abs=1e-9)


def test_noise_scores_no_beats_rather_than_one_per_window():
    """The amplitude gate is the whole point: NMS alone always finds peaks."""
    rng = np.random.default_rng(0)
    noise = 100.0 + 1e-3 * rng.standard_normal(int(FS * SECONDS))
    quality = detection_quality(noise, synthetic_abp(), FS)
    assert quality["n_detected"] == 0
    assert quality["sensitivity"] == pytest.approx(0.0)
```

Add `detection_quality` to the import at the top of the file.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluation_beats.py -k detection -v`
Expected: FAIL with `ImportError: cannot import name 'detection_quality'`.

- [ ] **Step 3: Implement the gate and the pass**

Append to `evaluation/beats.py`:

```python
#: A trace must swing at least this many times its own high-frequency noise
#: floor before its extrema count as beats. Non-maximum suppression alone
#: always returns one extremum per window, so without this a model emitting
#: noise would score a full complement of beats.
AMPLITUDE_GATE = 5.0


def _has_pulsatility(trace) -> bool:
    """True when the trace's beat-scale swing stands clear of its noise floor."""
    values = np.asarray(trace, dtype=np.float64)
    if values.size < 4:
        return False
    noise = np.median(np.abs(np.diff(values)))
    swing = float(values.max() - values.min())
    if noise <= 0:
        return swing > 0
    return swing > AMPLITUDE_GATE * noise


def detection_quality(prediction, reference, fs, tolerance_seconds=0.15) -> dict:
    """How well the prediction's *own* beats line up with the reference's.

    Kept apart from the agreement statistics on purpose. Agreement is
    reference-anchored, so it says nothing about whether the prediction has
    recognisable beats; this does, and folding the two together would let a
    model that finds few beats look accurate on the ones it did find.
    """
    width = max(3, int(fs * WIDTH_SECONDS))
    reference_feet, _ = find_peaks(reference, kind="min", width=width,
                                   clip_ends=True)
    if _has_pulsatility(prediction):
        predicted_feet, _ = find_peaks(prediction, kind="min", width=width,
                                       clip_ends=True)
    else:
        predicted_feet = np.array([], dtype=int)

    tolerance = tolerance_seconds * fs
    unmatched = list(map(int, reference_feet))
    matched = 0
    for foot in predicted_feet:
        if not unmatched:
            break
        nearest = min(unmatched, key=lambda candidate: abs(candidate - foot))
        if abs(nearest - foot) <= tolerance:
            unmatched.remove(nearest)
            matched += 1

    n_reference, n_detected = len(reference_feet), len(predicted_feet)
    reference_ibi = np.diff(reference_feet) / fs
    predicted_ibi = np.diff(predicted_feet) / fs
    ibi_error = (abs(float(predicted_ibi.mean()) - float(reference_ibi.mean()))
                 if predicted_ibi.size and reference_ibi.size else float("nan"))
    return {
        "sensitivity": matched / n_reference if n_reference else float("nan"),
        "ppv": matched / n_detected if n_detected else 0.0,
        "ibi_error": ibi_error,
        "n_reference": n_reference,
        "n_detected": n_detected,
    }
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_evaluation_beats.py -v`
Expected: all six PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/beats.py tests/test_evaluation_beats.py
git commit -m "feat(eval): beat detection quality with an amplitude gate"
```

---

### Task 5: `evaluation/uncertainty.py` — HAC standard errors and the block bootstrap

Lift the mature estimators from `neckflix_metrics.ipynb` (naive/HAC switch,
Andrews automatic bandwidth), not the earlier fixed-lag copy in `metrics.ipynb`.

**Files:**
- Create: `evaluation/uncertainty.py`
- Test: `tests/test_evaluation_uncertainty.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `mean_se(signal, method="HAC", kernel="QS", bandwidth="auto", fs=30) -> (mean, sd, se)`
  - `moving_block_bootstrap(stat_fn, pred, label, resamples=500, seed=0) -> float`

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluation_uncertainty.py`:

```python
"""Autocorrelation-aware uncertainty: the naive SE is the one that lies."""
import numpy as np
import pytest

from evaluation.uncertainty import mean_se, moving_block_bootstrap


def ar1(rho=0.9, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    series, value = np.empty(n), 0.0
    for i in range(n):
        value = rho * value + rng.standard_normal()
        series[i] = value
    return series


def test_hac_se_exceeds_the_naive_se_on_autocorrelated_data():
    series = ar1()
    _, _, naive = mean_se(series, method="naive")
    _, _, hac = mean_se(series, method="HAC")
    assert hac > 2 * naive


def test_naive_and_hac_agree_on_white_noise():
    series = np.random.default_rng(1).standard_normal(2000)
    _, _, naive = mean_se(series, method="naive")
    _, _, hac = mean_se(series, method="HAC")
    assert hac == pytest.approx(naive, rel=0.5)


def test_bootstrap_is_deterministic_under_a_seed():
    pred, label = ar1(seed=2), ar1(seed=3)
    def pearson(a, b):
        return float(np.corrcoef(a, b)[0, 1])
    first = moving_block_bootstrap(pearson, pred, label, resamples=64, seed=7)
    second = moving_block_bootstrap(pearson, pred, label, resamples=64, seed=7)
    assert first == second and first > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluation_uncertainty.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.uncertainty'`.

- [ ] **Step 3: Write the implementation**

Create `evaluation/uncertainty.py`:

```python
"""Standard errors that respect autocorrelation.

Consecutive samples of a physiological signal are nowhere near independent, so
``std / sqrt(n)`` over a 150-sample window is optimistic by a large factor.
Lifted from the PhysHydra-era analysis: a HAC (Newey-West) estimator with
Andrews (1991) automatic bandwidth selection, and a moving-block bootstrap for
the statistics with no usable closed form.

Applies to the DL-side metrics only. Clinical numbers use the standards' own
prescribed aggregation — see ``evaluation/scoring/standards.py``.
"""

import numpy as np

#: Andrews (1991) optimal-bandwidth constants and exponents, per kernel.
_ANDREWS = {
    "Bartlett": (1.1447, 1 / 3),
    "QS": (1.3221, 1 / 5),
    "Parzen": (2.6614, 1 / 5),
}


def _andrews_bandwidth(centred, n_samples, kernel) -> int:
    """Optimal maximum lag from a fitted AR(1) coefficient."""
    lag0 = float(centred @ centred) / n_samples
    lag1 = float(centred[1:] @ centred[:-1]) / n_samples
    rho = lag1 / lag0 if lag0 > 0 else 0.0
    rho = float(np.clip(rho, -0.97, 0.97))     # keep the formulas finite
    # Exactly the forms the PhysHydra analysis used — do not "simplify" them.
    if kernel == "Bartlett":
        alpha = 4 * rho ** 2 / ((1 - rho ** 2) ** 2 * (1 + rho ** 2))
    elif kernel == "QS":
        alpha = 4 * rho ** 2 / (1 - rho ** 2) ** 2
    elif kernel == "Parzen":
        alpha = 4 * rho ** 2 / (1 - rho ** 2)
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}")
    constant, exponent = _ANDREWS[kernel]
    return int(min(max(1, constant * (alpha * n_samples) ** exponent),
                   n_samples - 1))


def _kernel_weight(kernel, lag, max_lag) -> float:
    x = lag / (max_lag + 1.0)
    if kernel == "Bartlett":
        return 1.0 - x
    if kernel == "Parzen":
        return 2.0 * (1.0 - x) ** 3 if x > 0.5 else 1.0 - 6 * x ** 2 + 6 * x ** 3
    if kernel == "QS":
        z = 6.0 * np.pi * x / 5.0
        return 25.0 / (12.0 * np.pi ** 2 * x ** 2) * (np.sin(z) / z - np.cos(z))
    raise ValueError(f"Unknown kernel: {kernel!r}")


def mean_se(signal, method="HAC", kernel="QS", bandwidth="auto", fs=30):
    """``(mean, sd, se)`` for a 1-D series.

    ``method='naive'`` assumes i.i.d. samples and is right for a series whose
    elements are already independent units (one value per subject, say).
    ``method='HAC'`` is right within a recording.
    """
    values = np.asarray(signal, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    n_samples = values.size
    if n_samples == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if n_samples == 1:
        return mean, float("nan"), float("nan")
    sd = float(values.std(ddof=1))
    if method == "naive":
        return mean, sd, sd / np.sqrt(n_samples)
    if method != "HAC":
        raise ValueError(f"Unknown method: {method!r}. Use 'naive' or 'HAC'")

    centred = values - mean
    max_lag = (_andrews_bandwidth(centred, n_samples, kernel)
               if bandwidth == "auto"
               else int(min(bandwidth * fs, n_samples - 1)))
    variance = float(centred @ centred) / n_samples
    for lag in range(1, max_lag + 1):
        gamma = float(centred[lag:] @ centred[:-lag]) / n_samples
        variance += 2.0 * _kernel_weight(kernel, lag, max_lag) * gamma
    variance = max(variance, 0.0)               # kernels can undershoot
    return mean, sd, float(np.sqrt(variance / n_samples))


def moving_block_bootstrap(stat_fn, pred, label, resamples=500, seed=0) -> float:
    """Standard error of ``stat_fn(pred, label)`` under block resampling.

    Blocks of length ``n ** (1/3)`` keep the local autocorrelation intact,
    which an i.i.d. bootstrap would destroy. Seeded, so a report is
    reproducible.
    """
    pred = np.asarray(pred, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    n_samples = pred.size
    if n_samples < 4:
        return float("nan")
    rng = np.random.default_rng(seed)
    block = max(2, int(round(n_samples ** (1 / 3))))
    n_blocks = int(np.ceil(n_samples / block))
    starts = rng.integers(0, n_samples - block + 1, size=(resamples, n_blocks))
    offsets = np.arange(block)
    values = np.empty(resamples, dtype=np.float64)
    for i, row in enumerate(starts):
        index = (row[:, None] + offsets[None, :]).ravel()[:n_samples]
        values[i] = stat_fn(pred[index], label[index])
    values = values[np.isfinite(values)]
    return float(values.std(ddof=1)) if values.size > 1 else float("nan")
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_evaluation_uncertainty.py -v`
Expected: all three PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/uncertainty.py tests/test_evaluation_uncertainty.py
git commit -m "feat(eval): HAC standard errors and moving-block bootstrap"
```

---

### Task 6: `evaluation/levels.py` — the hierarchy, and sections from contiguity

**Files:**
- Create: `evaluation/levels.py`
- Test: covered by Task 12's tidy-frame contract test; no test of its own.

**Interfaces:**
- Consumes: `evaluation.records.WindowRecord` (Task 2).
- Produces:
  - `LEVELS = ("beat", "window", "section", "recording", "participant", "cohort")`
  - `GROUPING: dict[str, tuple[str, ...]]`
  - `sections(windows) -> list[Section]` where
    `Section(recording_id, camera_id, index, windows, prediction, label, attrs)`
    carries the stitched arrays.
  - `unit_id(level, **fields) -> str`

- [ ] **Step 1: Write the implementation**

Create `evaluation/levels.py`:

```python
"""The aggregation hierarchy, and the contiguity rule that defines a section.

The standards specify different levels — ISO 81060-3 wants a per-subject mean
and SD, IEEE 1708 a per-subject MAE — so a report that hardcodes one level
loses the others. Each level here names its grouping keys and nothing else.
"""

from dataclasses import dataclass

import numpy as np

#: Coarsening order. A run populates every level it can reach; a LOSO fold
#: simply has one participant.
LEVELS = ("beat", "window", "section", "recording", "participant", "cohort")

#: What identifies one unit at each level.
GROUPING = {
    "beat": ("recording_id", "camera_id", "section_index", "beat_index"),
    "window": ("recording_id", "camera_id", "start_frame"),
    "section": ("recording_id", "camera_id", "section_index"),
    "recording": ("recording_id", "camera_id"),
    "participant": ("participant",),
    "cohort": (),
}


@dataclass(frozen=True)
class Section:
    """A maximal contiguous run of windows, stitched back into one trace."""

    recording_id: str
    camera_id: str
    index: int
    windows: list
    prediction: np.ndarray
    label: np.ndarray
    attrs: dict


def unit_id(level, **fields) -> str:
    """Stable identity for one unit, from that level's grouping keys."""
    keys = GROUPING[level]
    return "|".join(str(fields[key]) for key in keys) if keys else "all"


def sections(windows) -> list:
    """Split windows into maximal contiguous runs per (recording, camera).

    Contiguity is ``next.start_frame == previous.start_frame + len(previous)``.
    Overlapping windows (a non-zero ``STRIDE_SECONDS`` under the window length)
    therefore never stitch — each becomes its own single-window section, which
    is the safe degradation: stitching overlapping windows would double-count
    the shared samples.

    Only signals loaded ``raw`` reconstruct correctly, so callers pass
    absolute-class windows. A per-window z-scored signal would step at every
    seam.
    """
    by_camera = {}
    for window in windows:
        by_camera.setdefault((window.recording_id, window.camera_id), []).append(window)

    result, index = [], 0
    for (recording_id, camera_id), group in sorted(by_camera.items()):
        group.sort(key=lambda w: w.start_frame)
        run = [group[0]]
        for previous, window in zip(group, group[1:]):
            if window.start_frame == previous.start_frame + len(previous.label):
                run.append(window)
            else:
                result.append(_stitch(recording_id, camera_id, index, run))
                index += 1
                run = [window]
        result.append(_stitch(recording_id, camera_id, index, run))
        index += 1
    return result


def _stitch(recording_id, camera_id, index, run) -> Section:
    return Section(
        recording_id=recording_id,
        camera_id=camera_id,
        index=index,
        windows=list(run),
        prediction=np.concatenate([w.prediction for w in run]),
        label=np.concatenate([w.label for w in run]),
        attrs=dict(run[0].attrs),
    )
```

- [ ] **Step 2: Verify it imports and stitches**

Run:
```bash
uv run python -c "
from evaluation.levels import sections, unit_id
from evaluation.records import WindowRecord
import numpy as np
w = lambda s: WindowRecord('ABP','R','1',s,np.zeros(4),np.zeros(4),{})
print([len(x.windows) for x in sections([w(0), w(4), w(8), w(100)])])
print(unit_id('recording', recording_id='R', camera_id='1'))
"
```
Expected: `[3, 1]` and `R|1`.

- [ ] **Step 3: Run the suite and commit**

Run: `uv run pytest -q`

```bash
git add evaluation/levels.py
git commit -m "feat(eval): aggregation hierarchy with sections from window contiguity"
```

---

### Task 7: `evaluation/scoring/waveform.py`

**Files:**
- Create: `evaluation/scoring/__init__.py`
- Create: `evaluation/scoring/waveform.py`
- Test: covered by Task 12.

**Interfaces:**
- Consumes: `evaluation.uncertainty.mean_se`, `moving_block_bootstrap` (Task 5);
  `evaluation.post_process._compute_macc`.
- Produces: `waveform_metrics(prediction, label, *, fs, bootstrap=0) -> dict[str, tuple[float, float]]`
  mapping metric name to `(value, se)`. Keys: `mae`, `rmse`, `pearson`, `ccc`, `macc`.

- [ ] **Step 1: Create the package and the module**

Create `evaluation/scoring/__init__.py`:

```python
"""Metric families. Which apply to a signal follows from its class, never from
a config key — a metric can then never go missing because a YAML forgot to
ask for it."""

from neural_methods.signals import is_absolute

#: Signal class -> the families computed for it.
FAMILIES = {
    "absolute": ("waveform", "rate", "clinical"),
    "shape": ("waveform", "rate"),
}


def families_for(signal) -> tuple:
    return FAMILIES["absolute" if is_absolute(signal) else "shape"]
```

Create `evaluation/scoring/waveform.py`:

```python
"""Shape and error agreement for one pair of traces, in physical units."""

import numpy as np

from evaluation.post_process import _compute_macc
from evaluation.uncertainty import mean_se, moving_block_bootstrap

_NAN = (float("nan"), float("nan"))


def _pearson(a, b) -> float:
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _ccc(a, b) -> float:
    """Lin's concordance: correlation penalised by disagreement in level."""
    if a.size < 2:
        return float("nan")
    r = _pearson(a, b)
    if not np.isfinite(r):
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    denominator = va + vb + (a.mean() - b.mean()) ** 2
    return float(2 * r * np.sqrt(va * vb) / denominator) if denominator > 0 else float("nan")


def waveform_metrics(prediction, label, *, fs, bootstrap=0) -> dict:
    """``{metric: (value, se)}`` for one prediction/label pair.

    Standard errors are HAC where the samples are autocorrelated, and bootstrap
    for Pearson and CCC, which have no usable closed form here. The bootstrap
    is the one expensive computation in the report, so it is opt-in.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(label, dtype=np.float64)
    if pred.size == 0 or pred.size != ref.size:
        return {name: _NAN for name in ("mae", "rmse", "pearson", "ccc", "macc")}

    error = pred - ref
    mae, _, mae_se = mean_se(np.abs(error), method="HAC", fs=fs)
    mean_square, _, square_se = mean_se(error ** 2, method="HAC", fs=fs)
    rmse = float(np.sqrt(mean_square))
    # Delta method: se(sqrt(x)) = se(x) / (2 sqrt(x)).
    rmse_se = square_se / (2 * rmse) if rmse > 0 else float("nan")

    r, ccc = _pearson(pred, ref), _ccc(pred, ref)
    if bootstrap:
        r_se = moving_block_bootstrap(_pearson, pred, ref, resamples=bootstrap)
        ccc_se = moving_block_bootstrap(_ccc, pred, ref, resamples=bootstrap)
    else:
        r_se = ccc_se = float("nan")

    return {
        "mae": (float(mae), float(mae_se)),
        "rmse": (rmse, float(rmse_se)),
        "pearson": (r, r_se),
        "ccc": (ccc, ccc_se),
        "macc": (float(_compute_macc(pred, ref)), float("nan")),
    }
```

- [ ] **Step 2: Verify it runs**

Run:
```bash
uv run python -c "
import numpy as np
from evaluation.scoring.waveform import waveform_metrics
t = np.linspace(0, 10, 300)
print(waveform_metrics(np.sin(t) + 0.05, np.sin(t), fs=30))
"
```
Expected: `pearson` ≈ 1.0, `mae` ≈ 0.05, no exception.

- [ ] **Step 3: Run the suite and commit**

Run: `uv run pytest -q`

```bash
git add evaluation/scoring/__init__.py evaluation/scoring/waveform.py
git commit -m "feat(eval): waveform metric family"
```

---

### Task 8: `evaluation/scoring/rate.py` — the HR family absorbs `report_hr_metrics`

**Files:**
- Create: `evaluation/scoring/rate.py`
- Test: covered by Task 12.

**Interfaces:**
- Consumes: `evaluation.post_process.calculate_metric_per_video`;
  `evaluation.uncertainty.mean_se` (Task 5).
- Produces:
  - `MIN_HR_WINDOW = 9`
  - `rate_metrics(prediction, label, *, fs, hr_method="FFT") -> dict[str, tuple[float, float]]`
    with keys `gt_hr`, `pred_hr`, `hr_error`, `snr`, `macc`; `{}` when the
    window is shorter than `MIN_HR_WINDOW`.
  - `aggregate_rate(rows) -> dict[str, tuple[float, float]]` reducing many
    windows' `rate_metrics` to `mae`, `rmse`, `mape`, `pearson`, `snr`, `macc`.

- [ ] **Step 1: Write the implementation**

Create `evaluation/scoring/rate.py`:

```python
"""Heart-rate agreement — what the upstream toolbox called *the* evaluation.

Now one family among several. The engine is unchanged
(``post_process.calculate_metric_per_video``); what changed is that it is
computed per signal and lands in the same tidy frame as everything else,
rather than being the whole report.
"""

import numpy as np

from evaluation.post_process import calculate_metric_per_video
from evaluation.uncertainty import mean_se

#: Shortest window ``filtfilt`` can pad. Below this the rate family declines to
#: report and the other families carry on.
MIN_HR_WINDOW = 9


def rate_metrics(prediction, label, *, fs, hr_method="FFT") -> dict:
    """One window's rate agreement, or ``{}`` when the window is too short."""
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(label, dtype=np.float64)
    if pred.size < MIN_HR_WINDOW:
        return {}
    gt_hr, pred_hr, snr, macc = calculate_metric_per_video(
        pred, ref, diff_flag=False, fs=fs, hr_method=hr_method)
    nan = float("nan")
    return {
        "gt_hr": (float(gt_hr), nan),
        "pred_hr": (float(pred_hr), nan),
        "hr_error": (float(pred_hr - gt_hr), nan),
        "snr": (float(snr), nan),
        "macc": (float(macc), nan),
    }


def aggregate_rate(rows) -> dict:
    """Reduce many windows' ``rate_metrics`` to the reported summary.

    Windows are treated as independent units here — they are separate
    measurement occasions, not consecutive samples of one series — so the
    standard errors are naive rather than HAC.
    """
    if not rows:
        return {}
    gt = np.array([row["gt_hr"][0] for row in rows])
    pred = np.array([row["pred_hr"][0] for row in rows])
    snr = np.array([row["snr"][0] for row in rows])
    macc = np.array([row["macc"][0] for row in rows])
    error = pred - gt

    mae, _, mae_se = mean_se(np.abs(error), method="naive")
    mean_square, _, square_se = mean_se(error ** 2, method="naive")
    rmse = float(np.sqrt(mean_square))
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.abs(error / gt)
    relative = relative[np.isfinite(relative)]
    mape, _, mape_se = mean_se(relative * 100, method="naive")

    # Undefined for fewer than three points or a constant series — which a
    # short split, or a model predicting one rate, produces. Say nan rather
    # than raising.
    if pred.size >= 3 and pred.std() > 0 and gt.std() > 0:
        r = float(np.corrcoef(pred, gt)[0, 1])
        r_se = float(np.sqrt(max(1 - r ** 2, 0.0) / (pred.size - 2)))
    else:
        r = r_se = float("nan")

    snr_mean, _, snr_se = mean_se(snr, method="naive")
    macc_mean, _, macc_se = mean_se(macc, method="naive")
    return {
        "mae": (float(mae), float(mae_se)),
        "rmse": (rmse, float(square_se / (2 * rmse)) if rmse > 0 else float("nan")),
        "mape": (float(mape), float(mape_se)),
        "pearson": (r, r_se),
        "snr": (float(snr_mean), float(snr_se)),
        "macc": (float(macc_mean), float(macc_se)),
    }
```

- [ ] **Step 2: Verify it runs**

Run:
```bash
uv run python -c "
import numpy as np
from evaluation.scoring.rate import rate_metrics, aggregate_rate
t = np.arange(300) / 30
w = np.sin(2 * np.pi * 1.2 * t)
rows = [rate_metrics(w, w, fs=30) for _ in range(4)]
print(rows[0]['gt_hr'], aggregate_rate(rows)['mae'])
"
```
Expected: a gt_hr near 72 bpm and an `mae` of 0.0.

- [ ] **Step 3: Run the suite and commit**

Run: `uv run pytest -q`

```bash
git add evaluation/scoring/rate.py
git commit -m "feat(eval): rate metric family, absorbing report_hr_metrics"
```

---

### Task 9: `evaluation/scoring/standards.py` — thresholds as named, sourced constants

> **These values are written from general knowledge of the standards, not from
> their text.** They must be checked against the purchased ISO 81060-3:2022 and
> IEEE 1708-2014 / 1708a-2019 documents before any report is used to make a
> clinical claim. That is why every constant carries a source comment and why
> the report prints a provenance line. Do not delete the `UNVERIFIED` marker as
> part of this task — removing it is the deliverable of the verification step,
> not of the implementation.

**Files:**
- Create: `evaluation/scoring/standards.py`
- Test: covered by Task 10.

**Interfaces:**
- Consumes: nothing.
- Produces: `IEEE_1708_GRADE_BANDS`, `ISO_81060_3_MEAN_ERROR_LIMIT`,
  `ISO_81060_3_SD_LIMIT`, `PROVENANCE: list[str]`,
  `STUDY_DESIGN_REQUIREMENTS: list[str]`, `CRITERIA: tuple[Criterion, ...]`.

- [ ] **Step 1: Write the implementation**

Create `evaluation/scoring/standards.py`:

```python
"""Clinical acceptance criteria, expressed as data.

Every threshold is a named constant with a source, and every report prints
where its numbers came from. Adding ESH 2023 or ISO 81060-2 later is rows in
``CRITERIA``, not new code.
"""

from dataclasses import dataclass

#: Set to True only once each constant below has been checked line by line
#: against the purchased standard. Until then every report says so out loud.
VERIFIED_AGAINST_STANDARD_TEXT = False

# --- IEEE 1708 ------------------------------------------------------------
#: Mean absolute error in mmHg, and the grade it earns. Anything above the
#: last band is grade D.
#: Source: IEEE 1708-2014, amended by 1708a-2019 — UNVERIFIED.
IEEE_1708_GRADE_BANDS = ((5.0, "A"), (6.0, "B"), (7.0, "C"))
IEEE_1708_FALLBACK_GRADE = "D"
IEEE_1708_SOURCE = "IEEE 1708-2014 / 1708a-2019 (UNVERIFIED)"

# --- ISO 81060-3 ----------------------------------------------------------
#: Continuous non-invasive BP against an invasive reference. Acceptance is on
#: the mean error and its standard deviation, pooled across subjects.
#: Source: ISO 81060-3:2022 — UNVERIFIED.
ISO_81060_3_MEAN_ERROR_LIMIT = 5.0     # mmHg
ISO_81060_3_SD_LIMIT = 8.0             # mmHg
ISO_81060_3_SOURCE = "ISO 81060-3:2022 (UNVERIFIED)"

#: Minimum subjects before a pooled verdict means anything. A LOSO fold has
#: one, so the report declines rather than grading.
MIN_SUBJECTS_FOR_VERDICT = 2

PROVENANCE = [
    f"IEEE 1708 grade bands from {IEEE_1708_SOURCE}",
    f"ISO 81060-3 limits from {ISO_81060_3_SOURCE}",
]

#: Requirements a metrics report can note but never satisfy: they constrain the
#: study, not the arithmetic. Printed with the numbers we do have, so nobody
#: reads a printed grade as a validation result.
STUDY_DESIGN_REQUIREMENTS = [
    "subject count and recruitment (a research cohort is not a validation study)",
    "prescribed distribution of reference pressures across the cohort",
    "reference-device protocol and its calibration record",
    "cuff procedure and observer training",
    "arm-circumference distribution",
]


@dataclass(frozen=True)
class Criterion:
    """One computable acceptance test."""

    name: str
    level: str          # the hierarchy level it consumes
    statistic: str      # which of BEAT_STATS it applies to
    source: str


CRITERIA = tuple(
    Criterion(name=name, level="participant", statistic=statistic, source=source)
    for name, source in (("ieee1708", IEEE_1708_SOURCE),
                         ("iso81060_3", ISO_81060_3_SOURCE))
    for statistic in ("max", "mean", "min")
)


def provenance_lines() -> list:
    """What a report prints above its clinical numbers."""
    lines = list(PROVENANCE)
    if not VERIFIED_AGAINST_STANDARD_TEXT:
        lines.append(
            "WARNING: thresholds have NOT been verified against the standards' "
            "text; treat grades as indicative only")
    lines.append("Not satisfiable by this dataset: "
                 + "; ".join(STUDY_DESIGN_REQUIREMENTS))
    return lines
```

- [ ] **Step 2: Verify the provenance prints the warning**

Run: `uv run python -c "from evaluation.scoring.standards import provenance_lines; print('\n'.join(provenance_lines()))"`
Expected: two source lines, the WARNING line, and the study-design line.

- [ ] **Step 3: Commit**

```bash
git add evaluation/scoring/standards.py
git commit -m "feat(eval): clinical criteria as sourced constants with provenance"
```

---

### Task 10: `evaluation/scoring/clinical.py` — per-beat agreement and grading

**Files:**
- Create: `evaluation/scoring/clinical.py`
- Test: `tests/test_evaluation_clinical.py`

**Interfaces:**
- Consumes: `evaluation.beats.BEAT_STATS`, `beat_intervals`, `beat_stats`
  (Task 3); `evaluation.scoring.standards` (Task 9);
  `evaluation.uncertainty.mean_se` (Task 5).
- Produces:
  - `beat_errors(prediction, label, fs) -> dict[str, np.ndarray]` keyed by `BEAT_STATS`
  - `grade_ieee1708(mae) -> str`
  - `iso81060_3_verdict(per_subject_errors) -> dict`

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluation_clinical.py`:

```python
"""Clinical criteria: a known error distribution must earn a known grade."""
import numpy as np
import pytest

from evaluation.scoring.clinical import (
    beat_errors, grade_ieee1708, iso81060_3_verdict)


def test_grades_follow_the_bands():
    assert grade_ieee1708(4.9) == "A"
    assert grade_ieee1708(5.5) == "B"
    assert grade_ieee1708(6.5) == "C"
    assert grade_ieee1708(9.0) == "D"
    assert grade_ieee1708(float("nan")) == "ungraded"


def test_iso_verdict_passes_a_tight_distribution_and_fails_a_wide_one():
    tight = iso81060_3_verdict(np.full(20, 1.0))
    assert tight["passes"] and tight["mean_error"] == pytest.approx(1.0)

    wide = iso81060_3_verdict(np.linspace(-40.0, 40.0, 20))
    assert not wide["passes"]


def test_a_single_subject_is_declined_rather_than_graded():
    verdict = iso81060_3_verdict(np.array([1.0]))
    assert verdict["passes"] is None
    assert "n = 1" in verdict["note"]


def test_beat_errors_are_zero_against_the_reference_itself():
    t = np.arange(300) / 30.0
    trace = 100.0 + 20.0 * -np.cos(2 * np.pi * 1.2 * t)
    errors = beat_errors(trace, trace, 30.0)
    assert errors["max"].size >= 8
    assert np.allclose(errors["mean"], 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluation_clinical.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.scoring.clinical'`.

- [ ] **Step 3: Write the implementation**

Create `evaluation/scoring/clinical.py`:

```python
"""Per-beat pressure agreement, and the acceptance criteria over it.

Beats are reference-anchored (``evaluation.beats``), so each reference beat
contributes exactly one comparison of each statistic and the error
distribution is not filtered by how detectable the prediction's own beats are.
"""

import numpy as np

from evaluation.beats import BEAT_STATS, beat_intervals, beat_stats
from evaluation.scoring import standards
from evaluation.uncertainty import mean_se


def beat_errors(prediction, label, fs) -> dict:
    """``prediction - label`` per beat, for each of ``BEAT_STATS``.

    The intervals come from the label, and both traces are read inside them.
    """
    intervals = beat_intervals(label, fs)
    if not intervals:
        return {name: np.array([], dtype=np.float64) for name in BEAT_STATS}
    predicted = beat_stats(prediction, intervals)
    reference = beat_stats(label, intervals)
    return {name: predicted[name] - reference[name] for name in BEAT_STATS}


def grade_ieee1708(mae) -> str:
    """The A-D band a mean absolute error falls in."""
    if mae is None or not np.isfinite(mae):
        return "ungraded"
    for limit, grade in standards.IEEE_1708_GRADE_BANDS:
        if mae <= limit:
            return grade
    return standards.IEEE_1708_FALLBACK_GRADE


def iso81060_3_verdict(per_subject_errors) -> dict:
    """Pooled mean error and SD against the acceptance limits.

    ``per_subject_errors`` is one mean error per subject, so the samples are
    independent and the standard error is naive — the standard prescribes this
    aggregation, and layering a HAC estimate on top would depart from it.

    ``passes`` is ``None``, never ``False``, when there are too few subjects
    for the verdict to mean anything — which is every LOSO fold.
    """
    errors = np.asarray(per_subject_errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    n_subjects = errors.size
    mean_error, sd, se = mean_se(errors, method="naive")
    result = {
        "n_subjects": int(n_subjects),
        "mean_error": float(mean_error),
        "sd": float(sd),
        "se": float(se),
        "mean_error_limit": standards.ISO_81060_3_MEAN_ERROR_LIMIT,
        "sd_limit": standards.ISO_81060_3_SD_LIMIT,
        "source": standards.ISO_81060_3_SOURCE,
        "note": "",
    }
    if n_subjects < standards.MIN_SUBJECTS_FOR_VERDICT:
        result["passes"] = None
        result["note"] = (f"not computable at n = {n_subjects}: the pooled SD "
                          f"needs at least "
                          f"{standards.MIN_SUBJECTS_FOR_VERDICT} subjects")
        return result
    result["passes"] = bool(
        abs(mean_error) <= standards.ISO_81060_3_MEAN_ERROR_LIMIT
        and sd <= standards.ISO_81060_3_SD_LIMIT)
    return result
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_evaluation_clinical.py -v`
Expected: all four PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/scoring/clinical.py tests/test_evaluation_clinical.py
git commit -m "feat(eval): per-beat clinical agreement and criteria evaluation"
```

---

### Task 11: `evaluation/report.py` — assemble the tidy frame

**Files:**
- Create: `evaluation/report.py`
- Test: `tests/test_evaluation_report.py`

**Interfaces:**
- Consumes: everything from Tasks 2-10.
- Produces:
  - `FRAME_COLUMNS = ("level", "unit_id", "signal", "metric", "statistic", "value", "se", "n")`
  - `build_frame(run, *, bootstrap=0, hr_method="FFT") -> pandas.DataFrame`
  - `digest(frame, run) -> str`
  - `write(frame, digest_text, output_dir, filename_id) -> tuple[Path, Path]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluation_report.py`:

```python
"""The tidy frame is the interface between layers; its schema is the contract."""
import numpy as np
import pytest

from evaluation.records import from_saved
from evaluation.report import FRAME_COLUMNS, build_frame, digest

FS = 30.0


def _run(n_windows=6, signal="ABP"):
    t = np.arange(120) / FS
    wave = 100.0 + 20.0 * -np.cos(2 * np.pi * 1.2 * t)
    raw = [{
        "signal": signal,
        "recording_id": "P015_S01_R3_0_D",
        "camera_id": "1",
        "start_frame": i * 120,
        "prediction": (wave + 2.0).astype(np.float32),
        "label": wave.astype(np.float32),
        "label_stats": {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0},
        "attrs": {"participant": "015", "posture": "0"},
    } for i in range(n_windows)]
    return from_saved(raw, fs=FS, traces=(signal,), label_norms={signal: "raw"})


def test_frame_has_the_contract_columns_and_level_vocabulary():
    frame = build_frame(_run())
    assert tuple(frame.columns[:len(FRAME_COLUMNS)]) == FRAME_COLUMNS
    assert set(frame["level"]) <= {"beat", "window", "section", "recording",
                                   "participant", "cohort"}
    assert frame["value"].dtype == np.float64
    assert "participant" in frame.columns


def test_a_constant_offset_shows_up_as_beat_level_bias():
    frame = build_frame(_run())
    beats = frame[(frame["level"] == "beat") & (frame["metric"] == "beat_error")]
    assert not beats.empty
    assert beats["value"].mean() == pytest.approx(2.0, abs=0.5)


def test_single_participant_is_declined_not_graded():
    frame = build_frame(_run())
    verdict = frame[frame["metric"] == "iso81060_3_passes"]
    assert verdict.empty or verdict["value"].isna().all()


def test_digest_names_the_threshold_provenance():
    text = digest(build_frame(_run()), _run())
    assert "UNVERIFIED" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluation_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.report'`.

- [ ] **Step 3: Write the implementation**

Create `evaluation/report.py`:

```python
"""Assemble every family at every level into one tidy frame, then say it.

The frame — ``level x unit x signal x metric x value x se`` — is what the
digest, the CSV, the JSON and every plot are views over. It is also why the
trainer and the sweep reporter are the same code: they call this with
different records and get whichever levels those records can support.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.beats import detection_quality
from evaluation.levels import sections, unit_id
from evaluation.scoring import families_for, standards
from evaluation.scoring.clinical import beat_errors, grade_ieee1708, iso81060_3_verdict
from evaluation.scoring.rate import aggregate_rate, rate_metrics
from evaluation.scoring.waveform import waveform_metrics
from neural_methods.signals import beat_labels, is_absolute, signal_unit

FRAME_COLUMNS = ("level", "unit_id", "signal", "metric", "statistic",
                 "value", "se", "n")


def _row(level, unit, signal, metric, value, se=float("nan"), statistic="",
         n=1, **attrs):
    return {"level": level, "unit_id": unit, "signal": signal, "metric": metric,
            "statistic": statistic, "value": float(value), "se": float(se),
            "n": int(n), **attrs}


def _participant(window) -> str:
    """The store's own normalised participant id, with a legible fallback."""
    return window.attrs.get("participant") or window.recording_id.split("_")[0]


def build_frame(run, *, bootstrap=0, hr_method="FFT") -> pd.DataFrame:
    """Every metric the records can support, as one long frame."""
    rows = []
    for signal in run.signals():
        windows = [w for w in run.windows if w.signal == signal]
        families = families_for(signal)
        rows.extend(_window_rows(windows, signal, run.fs, families, bootstrap,
                                 hr_method))
        if "clinical" in families and run.label_norms.get(signal) == "raw":
            rows.extend(_clinical_rows(windows, signal, run.fs))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=list(FRAME_COLUMNS))
    ordered = list(FRAME_COLUMNS) + [c for c in frame.columns
                                     if c not in FRAME_COLUMNS]
    return frame[ordered]


def _window_rows(windows, signal, fs, families, bootstrap, hr_method) -> list:
    rows, rate_rows = [], []
    for window in windows:
        unit = unit_id("window", recording_id=window.recording_id,
                       camera_id=window.camera_id, start_frame=window.start_frame)
        # Store attrs first: a store that already carries `participant` must
        # not collide with the key we derive. dict(**attrs, participant=...)
        # would raise TypeError on exactly the stores we care about.
        common = {**window.attrs,
                  "recording_id": window.recording_id,
                  "camera_id": window.camera_id,
                  "participant": _participant(window)}
        if "waveform" in families:
            for metric, (value, se) in waveform_metrics(
                    window.prediction, window.label, fs=fs,
                    bootstrap=bootstrap).items():
                rows.append(_row("window", unit, signal, metric, value, se,
                                 **common))
        if "rate" in families:
            measured = rate_metrics(window.prediction, window.label, fs=fs,
                                    hr_method=hr_method)
            if measured:
                rate_rows.append(measured)
                for metric, (value, se) in measured.items():
                    rows.append(_row("window", unit, signal, f"rate_{metric}",
                                     value, se, **common))
    for metric, (value, se) in aggregate_rate(rate_rows).items():
        rows.append(_row("cohort", "all", signal, f"rate_{metric}", value, se,
                         n=len(rate_rows)))
    return rows


def _clinical_rows(windows, signal, fs) -> list:
    """Beat, section, recording, participant and cohort rows for one signal."""
    rows, per_subject = [], {}
    for section in sections(windows):
        section_unit = unit_id("section", recording_id=section.recording_id,
                               camera_id=section.camera_id,
                               section_index=section.index)
        participant = section.attrs.get("participant") or \
            section.recording_id.split("_")[0]
        common = {**section.attrs,
                  "recording_id": section.recording_id,
                  "camera_id": section.camera_id,
                  "participant": participant}

        errors = beat_errors(section.prediction, section.label, fs)
        for statistic, values in errors.items():
            for index, value in enumerate(values):
                beat_unit = unit_id(
                    "beat", recording_id=section.recording_id,
                    camera_id=section.camera_id, section_index=section.index,
                    beat_index=index)
                rows.append(_row("beat", beat_unit, signal, "beat_error", value,
                                 statistic=statistic, **common))
            if values.size:
                rows.append(_row("section", section_unit, signal, "bias",
                                 float(values.mean()), statistic=statistic,
                                 n=values.size, **common))
                per_subject.setdefault(participant, {}).setdefault(
                    statistic, []).append(float(values.mean()))

        quality = detection_quality(section.prediction, section.label, fs)
        for metric, value in quality.items():
            rows.append(_row("section", section_unit, signal,
                             f"detection_{metric}", value, **common))

    for statistic in ("max", "mean", "min"):
        subject_means = {name: float(np.mean(stats[statistic]))
                         for name, stats in per_subject.items()
                         if stats.get(statistic)}
        for participant, value in subject_means.items():
            rows.append(_row("participant", unit_id("participant",
                                                    participant=participant),
                             signal, "bias", value, statistic=statistic,
                             participant=participant))
        if not subject_means:
            continue
        errors = np.array(list(subject_means.values()))
        verdict = iso81060_3_verdict(errors)
        passes = verdict["passes"]
        rows.append(_row("cohort", "all", signal, "iso81060_3_passes",
                         float("nan") if passes is None else float(passes),
                         statistic=statistic, n=verdict["n_subjects"]))
        rows.append(_row("cohort", "all", signal, "mean_error",
                         verdict["mean_error"], verdict["se"],
                         statistic=statistic, n=verdict["n_subjects"]))
        rows.append(_row("cohort", "all", signal, "sd", verdict["sd"],
                         statistic=statistic, n=verdict["n_subjects"]))
        grade = grade_ieee1708(float(np.abs(errors).mean()))
        rows.append(_row("cohort", "all", signal, f"ieee1708_grade_{grade}",
                         float(np.abs(errors).mean()), statistic=statistic,
                         n=verdict["n_subjects"]))
    return rows


def digest(frame, run) -> str:
    """The fixed, readable summary. The CSV is where everything else lives."""
    lines = ["=== Evaluation report ==="]
    lines += [f"  {line}" for line in standards.provenance_lines()]
    for signal in run.signals():
        unit = signal_unit(signal)
        rows = frame[frame["signal"] == signal]
        if rows.empty:
            lines.append(f"[{signal}] no windows carried this label — skipped")
            continue
        windows = rows[rows["level"] == "window"]
        lines.append(f"--- {signal} ({unit}) ---")
        for metric in ("mae", "rmse", "pearson", "ccc"):
            values = windows[windows["metric"] == metric]["value"]
            if not values.empty:
                lines.append(f"  window {metric}: {values.mean():.4f} "
                             f"over {len(values)} windows")
        if is_absolute(signal):
            labels = beat_labels(signal)
            for statistic in ("max", "mean", "min"):
                bias = rows[(rows["level"] == "cohort") &
                            (rows["metric"] == "mean_error") &
                            (rows["statistic"] == statistic)]
                if bias.empty:
                    continue
                row = bias.iloc[0]
                lines.append(f"  {labels[statistic]}: bias {row['value']:+.2f} "
                             f"{unit} over {int(row['n'])} subjects")
    return "\n".join(lines)


def write(frame, digest_text, output_dir, filename_id):
    """Write the frame and the digest; return both paths."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = Path(output_dir) / f"{filename_id}_metrics.csv"
    json_path = Path(output_dir) / f"{filename_id}_report.json"
    frame.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"digest": digest_text,
                   "provenance": standards.provenance_lines(),
                   "study_design_unmet": standards.STUDY_DESIGN_REQUIREMENTS},
                  handle, indent=2)
    print(f"Saved metrics to {csv_path} and {json_path}")
    return csv_path, json_path
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_evaluation_report.py -v`
Expected: all four PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/report.py tests/test_evaluation_report.py
git commit -m "feat(eval): tidy-frame report assembly, digest and outputs"
```

---

### Task 12: `evaluation/plots.py`, replacing `BlandAltmanPy`

**Files:**
- Create: `evaluation/plots.py`
- Test: none (figures are checked by eye; the smoke run exercises the code path).

**Interfaces:**
- Consumes: `evaluation.records.RunRecords` (Task 2);
  `neural_methods.signals.beat_labels` (Task 3).
- Produces: `STANDARD_PLOTS = ("waveforms", "agreement", "clinical")`;
  `draw(frame, run, *, output_dir, filename_id, plots=STANDARD_PLOTS) -> None`.

- [ ] **Step 1: Write the implementation**

Create `evaluation/plots.py`. Port `plot_waveform_overlays` and
`plot_absolute_agreement` from `evaluation/metrics_report.py` unchanged in
substance, with two edits: they take `WindowRecord`s (already physical, so the
inversion inside them goes), and the agreement panel titles come from
`beat_labels` rather than the hardcoded "systolic (max)" / "diastolic (min)".

```python
"""Every figure the evaluation produces, drawn once and never per model.

Cross-model comparability is the point of a standard plot set: a plot written
inside a model's own trainer stops being comparable the moment a second model
draws its own version.
"""

import os

import matplotlib
matplotlib.use("Agg")            # write files; never open a window on a cluster
import matplotlib.pyplot as plt
import numpy as np

from neural_methods.signals import beat_labels, is_absolute, signal_unit

STANDARD_PLOTS = ("waveforms", "agreement", "clinical")

#: Windows overlaid per signal. Enough to see whether a prediction tracks at
#: all across different recordings, few enough to read.
OVERLAYS_PER_SIGNAL = 4


def _save(figure, output_dir, file_name):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, file_name)
    figure.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(figure)
    print(f"Saved {file_name} to {output_dir}.")


def _evenly_spaced(items, count):
    """``count`` items spread across the list, not the first ``count``.

    A test split is ordered by recording, so the first N windows all come from
    one recording and one posture — the least informative sample available.
    """
    if len(items) <= count:
        return items
    picks = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in sorted(set(picks.tolist()))]


def draw(frame, run, *, output_dir, filename_id, plots=STANDARD_PLOTS) -> None:
    """Draw the requested figures from the records and the tidy frame."""
    for signal in run.signals():
        windows = [w for w in run.windows if w.signal == signal]
        if "waveforms" in plots:
            _waveforms(windows, signal, run.fs, output_dir, filename_id)
        if "agreement" in plots and is_absolute(signal):
            _agreement(windows, signal, output_dir, filename_id)
        if "clinical" in plots and is_absolute(signal):
            _bland_altman(frame, signal, output_dir, filename_id)


def _waveforms(windows, signal, fs, output_dir, filename_id):
    """Prediction vs label for a few windows, in the signal's own units."""
    chosen = _evenly_spaced(windows, OVERLAYS_PER_SIGNAL)
    if not chosen:
        return
    unit = signal_unit(signal)
    figure, axes = plt.subplots(len(chosen), 1, figsize=(9, 2.2 * len(chosen)),
                                squeeze=False)
    for axis, window in zip(axes[:, 0], chosen):
        time = np.arange(len(window.label)) / fs
        axis.plot(time, window.label, label="label", linewidth=1.2)
        axis.plot(time, window.prediction, label="prediction", linewidth=1.0,
                  alpha=0.85)
        axis.set_title(f"{window.recording_id} cam{window.camera_id} "
                       f"@ frame {window.start_frame}", fontsize=8)
        axis.set_ylabel(unit, fontsize=8)
        axis.tick_params(labelsize=7)
    axes[-1, 0].set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    figure.suptitle(f"{filename_id} — {signal} waveforms", fontsize=10)
    figure.tight_layout()
    _save(figure, output_dir, f"{filename_id}_{signal}_waveforms.pdf")


def _agreement(windows, signal, output_dir, filename_id):
    """Predicted vs true window statistics, named for the signal at hand."""
    labels = beat_labels(signal)
    unit = signal_unit(signal)
    panels = (("mean", np.mean), ("max", np.max), ("min", np.min))
    figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for axis, (statistic, reduce_fn) in zip(np.atleast_1d(axes), panels):
        predicted = np.array([reduce_fn(w.prediction) for w in windows])
        truth = np.array([reduce_fn(w.label) for w in windows])
        axis.scatter(truth, predicted, s=12, alpha=0.5, edgecolors="none")
        limits = [min(truth.min(), predicted.min()),
                  max(truth.max(), predicted.max())]
        axis.plot(limits, limits, linestyle="--", linewidth=1, color="0.4")
        bias = float(np.mean(predicted - truth))
        error = float(np.mean(np.abs(predicted - truth)))
        axis.set_title(f"{labels[statistic]}\nbias {bias:+.1f} · "
                       f"MAE {error:.1f} {unit}", fontsize=9)
        axis.set_xlabel(f"reference ({unit})", fontsize=8)
        axis.set_ylabel(f"predicted ({unit})", fontsize=8)
        axis.tick_params(labelsize=7)
    figure.suptitle(f"{filename_id} — {signal} agreement "
                    f"({len(windows)} windows)", fontsize=10)
    figure.tight_layout()
    _save(figure, output_dir, f"{filename_id}_{signal}_agreement.pdf")


def _bland_altman(frame, signal, output_dir, filename_id):
    """Per-subject bias with the 95% limits of agreement, one panel per statistic."""
    rows = frame[(frame["signal"] == signal) & (frame["level"] == "participant")
                 & (frame["metric"] == "bias")]
    if rows.empty:
        return
    labels = beat_labels(signal)
    unit = signal_unit(signal)
    statistics = [s for s in ("max", "mean", "min")
                  if not rows[rows["statistic"] == s].empty]
    figure, axes = plt.subplots(1, len(statistics),
                                figsize=(4 * len(statistics), 4), squeeze=False)
    for axis, statistic in zip(axes[0], statistics):
        bias = rows[rows["statistic"] == statistic]["value"].to_numpy()
        mean_bias = float(bias.mean())
        sd = float(bias.std(ddof=1)) if bias.size > 1 else float("nan")
        axis.scatter(np.arange(bias.size), bias, s=18, alpha=0.7)
        axis.axhline(mean_bias, linestyle="--", color="0.3", label="mean bias")
        if np.isfinite(sd):
            for limit in (mean_bias + 1.96 * sd, mean_bias - 1.96 * sd):
                axis.axhline(limit, linestyle=":", color="0.5")
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_title(f"{labels[statistic]} — bias {mean_bias:+.1f} {unit}",
                       fontsize=9)
        axis.set_xlabel("subject", fontsize=8)
        axis.set_ylabel(f"predicted − reference ({unit})", fontsize=8)
        axis.tick_params(labelsize=7)
    figure.suptitle(f"{filename_id} — {signal} per-subject agreement", fontsize=10)
    figure.tight_layout()
    _save(figure, output_dir, f"{filename_id}_{signal}_bland_altman.pdf")
```

- [ ] **Step 2: Leave `BlandAltmanPy.py` on disk, and stop using it**

Do **not** delete it, despite it being superseded. `evaluation/metrics.py` and
`evaluation/bigsmall_multitask_metrics.py` both do
`from evaluation.BlandAltmanPy import BlandAltman`, and those two modules are
Phase 6's to delete, not this plan's. So `BlandAltmanPy.py` joins them as
untouched legacy: no new code imports it, and it dies with the last legacy
trainer.

Confirm that is the whole remaining surface:

Run: `grep -rln "BlandAltman" --include=*.py .`
Expected: exactly `evaluation/BlandAltmanPy.py`, `evaluation/metrics.py`,
`evaluation/bigsmall_multitask_metrics.py`. If any other file appears, it has
not been migrated yet — migrate it before continuing.

- [ ] **Step 3: Confirm the new module imports cleanly**

Run: `uv run python -c "import evaluation.plots; print(evaluation.plots.STANDARD_PLOTS)"`
Expected: `('waveforms', 'agreement', 'clinical')`.

- [ ] **Step 4: Run the suite and commit**

Run: `uv run pytest -q`

```bash
git add evaluation/plots.py
git commit -m "feat(eval): plots module with per-signal beat labels"
```

---

### Task 13: Config — `TEST.REPORT` replaces `TEST.METRICS`

**Files:**
- Modify: `config.py:43` (delete `DEFAULT_METRICS`), `config.py:143-151` (`TestConfig`)
- Modify: `configs/neckflix/NECKFLIX_PHYSMAMBA.yaml:64` (delete the stale comment)
- Test: existing `tests/test_config*.py` must stay green.

**Interfaces:**
- Consumes: nothing.
- Produces: `ReportConfig(BOOTSTRAP: int = 0, PLOTS: list = STANDARD_PLOTS)`
  on `TestConfig.REPORT`.

- [ ] **Step 1: Edit the schema**

In `config.py`, delete the `DEFAULT_METRICS` constant, and replace
`TestConfig.METRICS` with a nested report block:

```python
@dataclass
class ReportConfig:
    """What the evaluation report costs, not what it contains.

    Which metrics apply to a signal follows from that signal's class, so there
    is deliberately no per-metric switch: a number can never go missing because
    a config forgot to ask for it. These two keys gate only the parts that cost
    something.
    """

    BOOTSTRAP: int = 0                  # resamples for Pearson/CCC SEs; 0 = skip
    PLOTS: list = field(default_factory=lambda: list(DEFAULT_PLOTS))


@dataclass
class TestConfig:
    """How predictions are scored — shared by every mode."""

    BATCH_SIZE: int = 4
    USE_LAST_EPOCH: bool = True
    EVALUATION_METHOD: str = "FFT"      # 'FFT' or 'peak detection'
    EVALUATION_WINDOW_SECONDS: float = 0.0  # 0 = score each window whole
    MODEL_PATH: str = ""                # only_test: the checkpoint to load
    REPORT: ReportConfig = field(default_factory=ReportConfig)
```

Define the default list in `config.py` rather than importing it — importing
`evaluation.plots` would drag matplotlib into every config load, including
`tools/list_neckflix_folds.py`, which is meant to be safe and fast on a login
node:

```python
#: Mirrors evaluation.plots.STANDARD_PLOTS, which is the source of truth.
#: Duplicated deliberately: importing it here would pull matplotlib into every
#: config load, including the metadata-only tools.
DEFAULT_PLOTS = ("waveforms", "agreement", "clinical")
```

and have `ReportConfig.PLOTS` default from `DEFAULT_PLOTS`.

- [ ] **Step 2: Delete the stale comment**

Remove line 64 of `configs/neckflix/NECKFLIX_PHYSMAMBA.yaml`:
`# METRICS omitted: the standard set (MAE RMSE MAPE MACC Pearson SNR BA).`

- [ ] **Step 3: Verify old configs are refused with a useful message**

Run:
```bash
uv run python -c "
from config import load_config
import tempfile, pathlib
p = pathlib.Path(tempfile.mkdtemp()) / 'bad.yaml'
p.write_text('MODE: only_test\nTEST:\n  METRICS: [MAE]\n')
try:
    load_config(str(p))
except Exception as exc:
    print(type(exc).__name__, exc)
"
```
Expected: a `ConfigError` naming the full path `TEST.METRICS` — the schema
already refuses unknown keys, so no new code is needed for this.

- [ ] **Step 4: Run the suite and commit**

Run: `uv run pytest -q`

```bash
git add config.py configs/neckflix/NECKFLIX_PHYSMAMBA.yaml
git commit -m "refactor(config): TEST.REPORT replaces the HR-only TEST.METRICS list"
```

---

### Task 14: Wire the trainer and the unsupervised predictor; delete `metrics_report.py`

**Files:**
- Modify: `neural_methods/trainer/MultiSignalTrainer.py:36-39, 523-570, 706-743`
- Modify: `unsupervised_methods/unsupervised_predictor.py:20, 171-184`
- Delete: `evaluation/metrics_report.py`, `tests/test_metrics_report.py`
- Test: existing smoke tests must stay green.

**Interfaces:**
- Consumes: `evaluation.records.from_saved`, `evaluation.report.build_frame`,
  `digest`, `write`, `evaluation.plots.draw`, `evaluation.scoring.rate.aggregate_rate`.
- Produces: `MultiSignalTrainer.test()` returns the tidy `DataFrame`.

- [ ] **Step 1: Rewrite the trainer's reporting tail**

In `MultiSignalTrainer.py`, replace the `evaluation.metrics_report` import:

```python
from evaluation.plots import draw as draw_plots
from evaluation.records import from_saved
from evaluation.report import build_frame, digest, write as write_report
```

Replace the reporting block at the end of `test()` (everything after the
inference loop, from `print('')` to `return report`) with:

```python
        print('')
        run = from_saved(windows, fs=self.frame_rate, traces=self.traces,
                         label_norms=self.label_norms)
        hr_method = ('Peak' if self.config.TEST.EVALUATION_METHOD == "peak detection"
                     else 'FFT')
        frame = build_frame(run, bootstrap=self.config.TEST.REPORT.BOOTSTRAP,
                            hr_method=hr_method)
        summary = digest(frame, run)
        print(summary)
        draw_plots(frame, run, output_dir=self._plot_dir(),
                   filename_id=self._filename_id(),
                   plots=self.config.TEST.REPORT.PLOTS)
        if self.config.RUN.output_dir:
            write_report(frame, summary, self.config.RUN.output_dir,
                         self._filename_id())
            self.save_dict_outputs(windows)
        return frame
```

Delete `plot_test_windows` and `_to_physical` from the trainer — `plots.draw`
and `records.to_physical` replace them. In `_score_sample`, drop the metric
accumulation into `stats` (the frame computes all of it now) and keep only the
record building, adding the new attrs:

```python
            records.append({
                'signal': signal,
                'recording_id': metadata['recording_id'],
                'camera_id': metadata['camera_id'],
                'start_frame': int(metadata['start_frame']),
                'attrs': dict(metadata.get(ATTRS) or {}),
                'prediction': prediction,
                'label': label,
                'label_stats': {k: float(v) for k, v in sample[LABEL_STATS][signal].items()},
            })
```

Import `ATTRS` from `neural_methods.batch`, and add `'attrs'` handling to
`save_dict_outputs`'s payload (the records already carry it).

- [ ] **Step 2: Wire the unsupervised predictor**

In `unsupervised_methods/unsupervised_predictor.py`, replace the
`report_hr_metrics` import with `from evaluation.scoring.rate import aggregate_rate`
and replace the body of `_report`'s loop:

```python
    report = {}
    for signal in sorted(signal_groups):
        group = signal_groups[signal]
        rows = [{"gt_hr": (gt, float("nan")), "pred_hr": (pred, float("nan")),
                 "snr": (snr, float("nan")), "macc": (macc, float("nan"))}
                for gt, pred, snr, macc in zip(group["gt"], group["pred"],
                                               group["snr"], group["macc"])]
        summary = aggregate_rate(rows)
        report[signal] = summary
        print(f"--- {signal}: {len(rows)} windows ---")
        for metric, (value, se) in summary.items():
            print(f"[{signal}] {metric}: {value} +/- {se}")
    return report
```

- [ ] **Step 3: Delete the old report module and its test**

```bash
git rm evaluation/metrics_report.py tests/test_metrics_report.py
```

- [ ] **Step 4: Run the suite and a real smoke run**

Run:
```bash
uv run pytest -q
uv run python main.py --limit_windows 8 --test_participants P015 \
  --config_file configs/neckflix/NECKFLIX_PHYSMAMBA_SMOKE.yaml
```
Expected: suite green; the smoke run prints the digest with the UNVERIFIED
provenance warning and writes `*_metrics.csv` and `*_report.json`.

- [ ] **Step 5: Commit**

```bash
git add -A neural_methods/trainer/MultiSignalTrainer.py unsupervised_methods/unsupervised_predictor.py evaluation tests
git commit -m "refactor(eval): move the trainer and unsupervised path onto the new report"
```

---

### Task 15: Grow `tools/summarise_neckflix_outputs.py` into the sweep reporter

This is where the participant and cohort levels become meaningful: a LOSO fold
has one test subject, so pooling folds is the only route to a cohort band.

**Files:**
- Modify: `tools/summarise_neckflix_outputs.py` (whole file)
- Test: none; exercised by the command in Step 2.

**Interfaces:**
- Consumes: `evaluation.records.load`, `evaluation.report.build_frame`,
  `digest`, `write`.
- Produces: the CLI `uv run python tools/summarise_neckflix_outputs.py <dir-or-pickle> [--csv PATH] [--bootstrap N]`.

- [ ] **Step 1: Rewrite the tool**

Replace the whole file with a thin CLI — every computation now lives in
`evaluation/`, so the tool's only jobs are argument parsing and pointing
`records.load` at a sweep:

```python
"""Pool one run or a whole LOSO sweep into one clinical report.

``MultiSignalTrainer`` writes one self-describing record per scored window, so
everything here is recomputed offline without touching the zarr cache. Point it
at a single fold's pickle for that fold's numbers, or at a sweep directory to
reach the participant and cohort levels — a fold has exactly one test subject,
so no single run can produce a pooled band.

    uv run python tools/summarise_neckflix_outputs.py runs/neckflix_physmamba
    uv run python tools/summarise_neckflix_outputs.py <pickle> --csv windows.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")

from evaluation.records import load                       # noqa: E402
from evaluation.report import build_frame, digest, write  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path,
                        help="an *_outputs.pickle, or a directory of them")
    parser.add_argument("--csv", type=Path, default=None,
                        help="also write the tidy frame here")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="resamples for Pearson/CCC standard errors")
    parser.add_argument("--hr-method", default="FFT",
                        choices=("FFT", "Peak"))
    args = parser.parse_args()

    run = load(args.target)
    print(f"Pooled {len(run.windows)} windows at {run.fs} Hz "
          f"across signals {run.signals()}")
    frame = build_frame(run, bootstrap=args.bootstrap, hr_method=args.hr_method)
    summary = digest(frame, run)
    print(summary)
    if args.csv:
        frame.to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}")
    else:
        directory = args.target if args.target.is_dir() else args.target.parent
        write(frame, summary, directory, "pooled")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against the smoke run's output**

Run: `uv run python tools/summarise_neckflix_outputs.py runs/`
Expected: the pooled line, the digest, and a `pooled_metrics.csv` /
`pooled_report.json` pair. With one fold the ISO verdict prints
"not computable at n = 1", which is the correct behaviour.

- [ ] **Step 3: Run the suite and commit**

Run: `uv run pytest -q`

```bash
git add tools/summarise_neckflix_outputs.py
git commit -m "feat(tools): pool a LOSO sweep into one cohort report"
```

---

### Task 16: Consume the prototypes; update the docs

**Files:**
- Delete: `evaluation/prototypes/` (whole directory)
- Modify: `CLAUDE.md` (the Codebase Map, Config Keys and Adding Metrics sections)
- Modify: `docs/project_status.md`
- Modify: `docs/plans/2026-08-31-overhaul-roadmap.md` (mark Phase 7 done)

- [ ] **Step 1: Confirm the prototypes are fully consumed**

Check each prototype capability has a home before deleting:
`find_peaks` → `evaluation/beats.py`; `mean_se` / `_moving_block_bootstrap` →
`evaluation/uncertainty.py`; `get_rmse` / `get_mae` / `get_pearson_r` /
`get_ccc` / `get_macc` → `evaluation/scoring/waveform.py`; `get_hr_fft` /
`get_snr` → `evaluation/scoring/rate.py`; `aggregate_data` →
`evaluation/levels.py` + `evaluation/report.py`; the Bland-Altman cells →
`evaluation/plots.py`.

Run: `grep -rn "prototypes" --include=*.py --include=*.md . | grep -v "^./docs/plans"`
Expected: no live references outside the plan and spec.

- [ ] **Step 2: Delete them**

```bash
git rm -r evaluation/prototypes
```

- [ ] **Step 3: Update the docs**

In `CLAUDE.md`:
- Codebase Map — replace the `evaluation/metrics_report.py, post_process.py —
  per-signal scoring` line with:
  `evaluation/ — records, beats, levels, uncertainty, metrics/ (waveform, rate,
  clinical, standards), report, plots: per-signal scoring from beat to cohort`.
- Config Keys — replace the `METRICS (omit for the standard set)` clause in the
  `TEST` bullet with
  `REPORT.BOOTSTRAP / REPORT.PLOTS (what applies to a signal follows from its
  class; these gate only cost)`.
- Adding Metrics — replace the section body with: extend the relevant family in
  `evaluation/scoring/`; a new clinical criterion is rows in
  `metrics/standards.py`, not new code; every number lands in the tidy frame.
- Delete the `evaluation/prototypes/` sentence.

In `docs/project_status.md`, move the "Pressure-specific metrics" To Do bullet
into Completed, describing what landed.

In `docs/plans/2026-08-31-overhaul-roadmap.md`, mark Phase 7's three steps done
and add a decision-log entry recording the design's key choices (hierarchy,
reference anchoring, uniform max/mean/min with per-signal labels, thresholds
pending verification).

- [ ] **Step 4: Run the full suite and one smoke run**

Run:
```bash
uv run pytest -q
uv run python main.py --limit_windows 8 --test_participants P015 \
  --config_file configs/neckflix/NECKFLIX_DEEPPHYS_SMOKE.yaml
```
Expected: green suite; a clean run end to end.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs: close Phase 7; delete the consumed evaluation prototypes"
```

---

## Verification steps (carried from the spec, §17)

These are **not** part of the tasks above. Do them before any report is used to
support a clinical claim.

1. **Check every constant in `evaluation/scoring/standards.py` against the
   purchased ISO 81060-3:2022 and IEEE 1708-2014 / 1708a-2019 texts.** Only
   then set `VERIFIED_AGAINST_STANDARD_TEXT = True` and update each `UNVERIFIED`
   source string with the clause it came from.
2. **Check the beat detector against real cached ABP**, not only the synthetic
   test waveform: run the beat layer over a full recording from `CACHED_PATH`
   and confirm the detected rate matches the ECG-derived rate, and that
   `WIDTH_SECONDS` and `AMPLITUDE_GATE` behave on the noisiest posture.
3. **Confirm the sweep path on a real LOSO sweep** once one exists — the
   participant and cohort levels are exercised by nothing smaller.
