# Contract v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Implement contract v2 — the cache validator (Part 1), the model
contract rework: in-model losses + style-C parallel models (Part 2), and the
v2 cache reader adoption (Part 3, **gated** on a regenerated cache).

**Architecture:** Part 1 makes the cache contract executable
(`tools/validate_cache.py`) against vocabulary tables added to
`neural_methods/signals.py`. Part 2 moves loss computation inside
`DictModel` (`batch["raw_losses"]`, model-written) with weighting in the
trainer (`batch["losses"]`), and adds `ParallelSignals` — S full copies of a
single-signal architecture — as the default `HEAD_STYLE: parallel`. Part 3
adapts `BaseZarrDataset` to the v2 store layout and collapses to one dataset
class.

**Tech Stack:** Python 3.13, PyTorch 2.12, zarr v3, einops, pytest, `uv`.

**Spec:** `docs/plans/2026-09-01-contract-v2-design.md` — read it first; the
plan argues from it. Roadmap context:
`docs/plans/2026-08-31-overhaul-roadmap.md` (Phases A–B, decision 17).

## Global Constraints

- Dependencies via `uv add` only, never pip. Run everything as
  `uv run python ...` / `uv run python -m pytest ...`.
- All tensor reshaping uses einops (`rearrange`/`reduce`/`einsum`), never
  `view`/`permute`/`reshape`.
- **Testing stays minimal**: extend the named existing test files; one new
  test file total (`tests/test_validate_cache.py`). Do not add tests
  opportunistically.
- Legacy code is deleted, not adapted; no compatibility shims. The
  `pre-overhaul` tag and git history are the archive.
- Dev box is Windows. The full suite (`uv run python -m pytest -q`) is 289
  tests / ~50 s; two tests (`test_main.py::test_unsupervised_mode_gets_its_own_output_dir`,
  `test_multisignal_trainer.py::test_one_training_step_moves_the_weights`)
  can hit a known Windows tmp-dir `PermissionError` flake in full runs —
  rerun them in isolation before concluding anything broke.
- The working tree may carry the user's own unstaged edits (`.gitignore`,
  `configs/neckflix/*.yaml`, `docs/project_status.md`, untracked `test.py`).
  **Never stage, commit, or revert those files.** Stage your own files by
  exact path (`git add <paths>`), never `git add -A`/`-u`/`.`.
- Commits: conventional-commit style (`feat:`, `fix:`, `test:`, `docs:`).
- Existing checkpoints predate contract v2 (`HEAD_STYLE: widened`
  state_dicts). They are not migrated — re-training is expected and
  acceptable; do not build a loading shim.
- Parts 1 and 2 are independent; tasks within a part are ordered. **Part 3
  must not start** until a regenerated v2 Neckflix cache exists and passes
  the validator (its location will be supplied then; the current
  `D:/neckflix_zarr/rgbid256` cache is v1).

---

## Part 1 — The cache validator (execute now)

### Task 1: Vocabulary tables in `neural_methods/signals.py`

**Files:**
- Modify: `neural_methods/signals.py` (CHANNELS is at line 9, `_ALIASES` ~line 50)
- Test: `tests/test_signals.py`

**Interfaces:**
- Consumes: existing `CHANNELS`, `SIGNALS`, `canonical_signal`.
- Produces (used by Tasks 3, 11): `MODALITY_CHANNELS: dict[str, tuple | None]`,
  `TRACE_KEYS: dict[str, str]` — both module-level constants in
  `neural_methods.signals`.

- [x] **Step 1: Write the failing test** — append to `tests/test_signals.py`:

```python
def test_modality_and_trace_vocabularies():
    from neural_methods.signals import (
        CHANNELS, MODALITY_CHANNELS, TRACE_KEYS, canonical_signal,
        validate_channels,
    )
    # Every pinned modality's channels are canonical channel names.
    for modality, channels in MODALITY_CHANNELS.items():
        if channels is not None:
            assert validate_channels(list(channels)) == list(channels)
    assert MODALITY_CHANNELS["rgb"] == ("R", "G", "B")
    assert MODALITY_CHANNELS["ev"] is None            # not yet pinned
    assert "Y" in CHANNELS and "T" in CHANNELS        # grayscale, thermal
    # Every cache trace key maps to a known canonical signal.
    for cache_key, signal in TRACE_KEYS.items():
        assert canonical_signal(signal) == signal
    assert TRACE_KEYS["rr"] == "RESP"
```

- [x] **Step 2: Run it, expect ImportError**

Run: `uv run python -m pytest tests/test_signals.py::test_modality_and_trace_vocabularies -v`
Expected: FAIL — `ImportError: cannot import name 'MODALITY_CHANNELS'`.

- [x] **Step 3: Implement** — in `neural_methods/signals.py`, extend
  `CHANNELS` and add the two tables directly below it (keep the existing
  five letters first so nothing that indexes them shifts):

```python
CHANNELS = ('R', 'G', 'B', 'I', 'D', 'Y', 'T')

#: Cache contract v2 (docs/plans/2026-09-01-contract-v2-design.md, Part 1):
#: modality group name -> the canonical channels its video planes carry, in
#: stacking order. THE global channel map — per-dataset channel_map
#: subclasses die against this table in the Part 3 reader adoption.
#: ``None`` = frame representation not yet pinned (validated loosely).
MODALITY_CHANNELS = {
    'gr':    ('Y',),
    'rgb':   ('R', 'G', 'B'),
    'ir':    ('I',),
    'depth': ('D',),
    't':     ('T',),
    'ev':    None,
}

#: Cache trace group name -> canonical signal name.
TRACE_KEYS = {'ecg': 'ECG', 'abp': 'ABP', 'cvp': 'CVP',
              'ppg': 'PPG', 'rr': 'RESP'}
```

- [x] **Step 4: Run the test file and the suite's fast neighbours**

Run: `uv run python -m pytest tests/test_signals.py tests/test_batch_contract.py -q`
Expected: all PASS (appending two letters to `CHANNELS` only widens
`validate_channels`; nothing indexes past position 4).

- [x] **Step 5: Commit**

```bash
git add neural_methods/signals.py tests/test_signals.py
git commit -m "feat(signals): contract-v2 modality and trace vocabularies"
```

### Task 2: v2 store fixture builder

**Files:**
- Modify: `tests/zarr_fixtures.py` (leave the existing v1 `make_store`
  untouched — the reader still speaks v1 until Part 3)

**Interfaces:**
- Produces (used by Tasks 3, 11):
  `make_v2_store(cache_dir, name="P030_S01_R1_0_D", *, attrs=None,
  perspectives=("1",), modalities=("rgb", "ir", "depth"),
  traces=("abp", "cvp"), num_frames=12, hw=(8, 8), fps=30.0,
  units=None, trace_lengths=None, first_frame_offsets_us=None) -> pathlib.Path`
  — returns the store path.

- [x] **Step 1: Implement** — append to `tests/zarr_fixtures.py`:

```python
V2_UNITS = {"abp": "mmHg", "cvp": "mmHg", "ecg": "arb",
            "ppg": "arb", "rr": "arb"}


def make_v2_store(cache_dir, name="P030_S01_R1_0_D", *, attrs=None,
                  perspectives=("1",), modalities=("rgb", "ir", "depth"),
                  traces=("abp", "cvp"), num_frames=12, hw=(8, 8), fps=30.0,
                  units=None, trace_lengths=None,
                  first_frame_offsets_us=None):
    """A contract-v2 store (docs/plans/2026-09-01-contract-v2-design.md).

    v2 layout: root attrs carry only ``participant`` (+ free attrs), each
    perspective carries ``fps``, each modality carries ``timestamps_us/data``
    and ``video/data``, each trace carries a ``units`` attr.

    ``trace_lengths`` / ``first_frame_offsets_us``: optional per-modality
    dicts to build deliberately inconsistent stores for validator tests.
    """
    channel_counts = {"gr": 1, "rgb": 3, "ir": 1, "depth": 1, "t": 1, "ev": 1}
    units = {**V2_UNITS, **(units or {})}
    height, width = hw
    path = cache_dir / f"{name}.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update({"participant": name.split("_")[0][1:],
                       "posture": name.split("_")[-2],
                       **(attrs or {})})
    for perspective in perspectives:
        cam = root.create_group(perspective)
        cam.attrs["fps"] = fps
        for modality in modalities:
            group = cam.create_group(modality)
            length = (trace_lengths or {}).get(modality, num_frames)
            offset = (first_frame_offsets_us or {}).get(modality, 0.0)
            step = 1e6 / fps
            stamps = offset + step * np.arange(length)
            group.create_group("timestamps_us")["data"] = stamps.astype(np.int64)
            channels = channel_counts[modality]
            video = np.zeros((channels, length, height, width), dtype=np.uint8)
            group.create_group("video")["data"] = video
            for trace in traces:
                trace_group = group.create_group(trace)
                trace_group["data"] = np.full(
                    length, TRACE_OFFSETS.get(trace, 1.0), dtype=np.float64)
                trace_group.attrs["units"] = units.get(trace, "arb")
    return path
```

- [x] **Step 2: Sanity-run it**

Run: `uv run python -c "import pathlib, tempfile; from tests.zarr_fixtures import make_v2_store; d = pathlib.Path(tempfile.mkdtemp()); print(make_v2_store(d))"`
Expected: prints the store path, no traceback.

- [x] **Step 3: Commit**

```bash
git add tests/zarr_fixtures.py
git commit -m "test: contract-v2 zarr store fixture"
```

### Task 3: `tools/validate_cache.py`

**Files:**
- Create: `tools/validate_cache.py`
- Create: `tests/test_validate_cache.py`

**Interfaces:**
- Consumes: Task 1's `MODALITY_CHANNELS`, `TRACE_KEYS`; Task 2's
  `make_v2_store`.
- Produces: `Violation(where: str, message: str)` (NamedTuple),
  `validate_store(path) -> list[Violation]`, CLI
  `uv run python tools/validate_cache.py <cache-dir | store.zarr ...>`
  (exit 0 = all pass). The external preprocessor repo will import
  `validate_store`; keep its signature stable.

- [x] **Step 1: Write the failing tests** — `tests/test_validate_cache.py`:

```python
"""Smoke test for the contract-v2 cache validator (one file, several cases)."""
import numpy as np
import zarr

from tests.zarr_fixtures import make_v2_store
from tools.validate_cache import validate_store


def test_conformant_store_passes(tmp_path):
    path = make_v2_store(tmp_path, traces=("abp", "cvp", "ecg"))
    assert validate_store(path) == []


def _messages(path):
    return " | ".join(v.message for v in validate_store(path))


def test_violations_are_itemised(tmp_path):
    # Missing participant.
    path = make_v2_store(tmp_path, name="P001_S01_R1_0_D")
    root = zarr.open_group(str(path), mode="a")
    del root.attrs["participant"]
    assert "participant" in _messages(path)

    # Unknown modality name.
    path = make_v2_store(tmp_path, name="P002_S01_R1_0_D")
    root = zarr.open_group(str(path), mode="a")
    bad = root["1"].create_group("sonar")
    bad.create_group("video")["data"] = np.zeros((1, 4, 2, 2), np.uint8)
    assert "sonar" in _messages(path)

    # Trace set differs between modalities.
    path = make_v2_store(tmp_path, name="P003_S01_R1_0_D")
    root = zarr.open_group(str(path), mode="a")
    del root["1"]["ir"]["cvp"]
    assert "trace" in _messages(path).lower()

    # Missing units attr.
    path = make_v2_store(tmp_path, name="P004_S01_R1_0_D")
    root = zarr.open_group(str(path), mode="a")
    del root["1"]["rgb"]["abp"].attrs["units"]
    assert "units" in _messages(path)

    # Trace length disagrees with the video within one modality.
    path = make_v2_store(tmp_path, name="P005_S01_R1_0_D")
    root = zarr.open_group(str(path), mode="a")
    root["1"]["rgb"]["abp"]["data"] = np.zeros(7, np.float64)
    assert "abp" in _messages(path)

    # First-frame misalignment beyond 1/fps (fps=30 -> 33_333 us budget).
    path = make_v2_store(tmp_path, name="P006_S01_R1_0_D",
                         first_frame_offsets_us={"ir": 50_000.0})
    assert "align" in _messages(path).lower()

    # Missing perspective fps.
    path = make_v2_store(tmp_path, name="P007_S01_R1_0_D")
    root = zarr.open_group(str(path), mode="a")
    del root["1"].attrs["fps"]
    assert "fps" in _messages(path)
```

- [x] **Step 2: Run, expect ModuleNotFoundError**

Run: `uv run python -m pytest tests/test_validate_cache.py -v`
Expected: FAIL — `tools.validate_cache` does not exist. (If `tools/` lacks
an `__init__.py` and the import fails for that reason, do NOT add one —
other `tools/` scripts are plain scripts; instead import in the test via
the same pattern any existing test uses for tools, or load by path with
`importlib.util.spec_from_file_location`. Check how `tests/test_pure_cache.py`
imports `tools/cache_pure.py` first and copy that pattern exactly, adjusting
the test file's import line to match.)

- [x] **Step 3: Implement `tools/validate_cache.py`**

```python
"""Contract-v2 cache validator — the admission mechanism, made executable.

The contract: docs/plans/2026-09-01-contract-v2-design.md (Part 1). Run this
after generating a cache; a store this passes is admissible, full stop —
there is no ``complete``/``tool_version`` gate any more.

    uv run python tools/validate_cache.py <cache-dir | store.zarr ...>
"""
import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import zarr

from neural_methods.signals import MODALITY_CHANNELS, TRACE_KEYS


class Violation(NamedTuple):
    where: str          # "store/perspective/modality" style path
    message: str

    def __str__(self):
        return f"{self.where}: {self.message}"


def _check_modality(out, where, group, fps):
    """Within-modality checks; returns the first timestamp, or None."""
    for required in ("timestamps_us", "video"):
        if required not in group or "data" not in group[required]:
            out.append(Violation(where, f"missing {required}/data"))
            return None
    video = group["video"]["data"]
    if video.ndim != 4:
        out.append(Violation(where, f"video/data is {video.ndim}-D, want (C, T, H, W)"))
        return None
    if video.dtype != np.uint8:
        out.append(Violation(where, f"video/data dtype {video.dtype}, want uint8"))
    modality = where.rsplit("/", 1)[-1]
    expected = MODALITY_CHANNELS[modality]
    if expected is not None and video.shape[0] != len(expected):
        out.append(Violation(
            where, f"video/data has C={video.shape[0]}, {modality} wants "
                   f"{len(expected)} ({', '.join(expected)})"))
    stamps = group["timestamps_us"]["data"][:]
    frames = video.shape[1]
    if stamps.shape != (frames,):
        out.append(Violation(
            where, f"timestamps_us length {stamps.shape} vs T={frames}"))
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
        if "data" not in group[key]:
            out.append(Violation(sub, "missing data array"))
            continue
        trace = group[key]["data"]
        if trace.shape != (frames,):
            out.append(Violation(
                sub, f"trace length {trace.shape} is not index-aligned to "
                     f"video T={frames}"))
        if not np.issubdtype(trace.dtype, np.floating):
            out.append(Violation(sub, f"trace dtype {trace.dtype}, want float"))
        if "units" not in group[key].attrs:
            out.append(Violation(sub, "missing required 'units' attr"))
    return float(stamps[0]) if stamps.size else None


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
    perspectives = list(root.group_keys())
    if not perspectives:
        out.append(Violation(path.name, "store has no perspective groups"))
    for perspective in perspectives:
        cam = root[perspective]
        where = f"{path.name}/{perspective}"
        fps = cam.attrs.get("fps")
        if not fps:
            out.append(Violation(where, "missing required perspective attr 'fps'"))
        modalities = list(cam.group_keys())
        trace_sets, first_stamps = {}, {}
        for modality in modalities:
            sub = f"{where}/{modality}"
            if modality not in MODALITY_CHANNELS:
                out.append(Violation(
                    sub, f"unknown modality; vocabulary: "
                         f"{sorted(MODALITY_CHANNELS)}"))
                continue
            first = _check_modality(out, sub, cam[modality], fps)
            trace_sets[modality] = frozenset(
                k for k in cam[modality].group_keys()
                if k not in ("timestamps_us", "video"))
            if first is not None:
                first_stamps[modality] = first
        if len(set(trace_sets.values())) > 1:
            listing = "; ".join(f"{m}: {sorted(s)}" for m, s in trace_sets.items())
            out.append(Violation(
                where, f"modalities carry different trace sets ({listing})"))
        if fps and len(first_stamps) > 1:
            budget_us = 1e6 / float(fps)
            spread = max(first_stamps.values()) - min(first_stamps.values())
            if spread >= budget_us:
                out.append(Violation(
                    where, f"first frames misaligned by {spread:.0f}us, over "
                           f"the 1/fps budget of {budget_us:.0f}us"))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+",
                        help="cache directories and/or individual .zarr stores")
    args = parser.parse_args(argv)
    stores = []
    for raw in args.paths:
        path = Path(raw)
        stores.extend(sorted(path.glob("*.zarr")) if path.is_dir() else [path])
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
    print(f"{len(stores) - failed}/{len(stores)} stores pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 4: Run the tests**

Run: `uv run python -m pytest tests/test_validate_cache.py -v`
Expected: both PASS. Iterate on message wording only until the asserted
substrings match.

- [x] **Step 5: Run the CLI against the current v1 cache** (a useful
  negative check — v1 stores must FAIL). They fail at the *first* clause, not
  the ones this step originally predicted: v1 keeps `timestamps_us` and
  `frames` as arrays under `<modality>/video/`, and `fps` on that video group,
  so every store reports `missing timestamps_us/data` and `missing required
  perspective attr 'fps'` and never reaches the `video/data`/`units` checks.
  Confirmed: 0/332 stores pass, exit code 1.

Run: `uv run python tools/validate_cache.py D:/neckflix_zarr/rgbid256` (skip
gracefully if the drive is absent; do not treat absence as failure)
Expected: FAIL lines for every store — v1 layout is not v2.

- [x] **Step 6: Commit**

```bash
git add tools/validate_cache.py tests/test_validate_cache.py
git commit -m "feat(tools): contract-v2 cache validator"
```

---

## Part 2 — Model contract v2 (execute now)

### Task 4: Batch keys for the two loss dicts

**Files:**
- Modify: `neural_methods/batch.py` (key constants at lines 30–48)

**Interfaces:**
- Produces (used by Tasks 5–7): `RAW_LOSSES = "raw_losses"`,
  `LOSSES = "losses"`, `LABEL_UNITS = "label_units"` (reserved now, wired in
  Part 3).

- [x] **Step 1: Add the constants** beside the existing ones in
  `neural_methods/batch.py`:

```python
RAW_LOSSES = "raw_losses"    # {module: {component: () tensor}} — model-written, unweighted
LOSSES = "losses"            # same structure, config-weighted — trainer-written
LABEL_UNITS = "label_units"  # {signal: str}, from the v2 cache's units attrs
```

- [x] **Step 2: Run the contract tests**

Run: `uv run python -m pytest tests/test_batch_contract.py -q`
Expected: PASS (additive change).

- [x] **Step 3: Commit**

```bash
git add neural_methods/batch.py
git commit -m "feat(batch): raw_losses/losses/label_units key names"
```

### Task 5: `PerSignalLoss` returns raw components; weighting becomes a free function

**Files:**
- Modify: `neural_methods/loss/PerSignalLoss.py` (class at line 250,
  `forward` at line 278)
- Test: `tests/test_per_signal_loss.py`

**Interfaces:**
- Consumes: existing component functions and `resolve_loss_specs` (both
  unchanged).
- Produces (used by Tasks 6–7):
  - `PerSignalLoss.forward(preds, labels, label_mask) -> dict[str, dict[str, Tensor]]`
    — **unweighted**, graph-attached `()` tensors, one module entry per
    signal, components exactly the spec'd weight keys, **no `'total'` entry**.
  - `weight_losses(raw: dict, weights: dict) -> tuple[Tensor, dict]` —
    module-level. `weights = {module: {component: float}}`; a component with
    no weight entry defaults to `1.0`. Returns `(total, weighted)`:
    `total` is the graph-attached scalar to backpropagate — **the mean over
    modules of each module's weighted component sum** (identical to today's
    mean-over-signals when the modules are exactly the signals);
    `weighted` mirrors `raw` with detached floats plus a `'total'` float
    per module.

- [x] **Step 1: Rewrite the existing forward tests' expectations.** In
  `tests/test_per_signal_loss.py`, tests currently unpack
  `total, breakdown = criterion(preds, labels, mask)`. Update each call
  site to the new two-step shape and keep the numeric assertions by going
  through `weight_losses`:

```python
from neural_methods.loss.PerSignalLoss import PerSignalLoss, weight_losses

raw = criterion(preds, labels, mask)
weights = {sig: spec["weights"] for sig, spec in criterion.specs.items()}
total, weighted = weight_losses(raw, weights)
```

  Also add one new test pinning the contract itself:

```python
def test_raw_is_unweighted_and_weighting_is_separate():
    import torch
    # Zero weights are filtered by resolve_loss_specs, so this spec leaves
    # exactly one component (ccc) — the arithmetic below relies on that.
    criterion = PerSignalLoss(["ABP"], specs={"ABP": {"WEIGHTS": {
        "CCC": 2.0, "MEAN": 0, "MAX": 0, "MIN": 0}}})
    preds = {"ABP": torch.randn(4, 32)}
    labels = {"ABP": torch.randn(4, 32)}
    mask = {"ABP": torch.ones(4, dtype=torch.bool)}
    raw = criterion(preds, labels, mask)
    assert set(raw) == {"ABP"} and "total" not in raw["ABP"]
    assert raw["ABP"]["ccc"].requires_grad
    total, weighted = weight_losses(raw, {"ABP": {"ccc": 2.0}})
    assert torch.isclose(total, 2.0 * raw["ABP"]["ccc"])
    assert weighted["ABP"]["total"] == float(total)
```

- [x] **Step 2: Run, expect failures**

Run: `uv run python -m pytest tests/test_per_signal_loss.py -q`
Expected: FAIL — old forward returns a 2-tuple, `weight_losses` undefined.

- [x] **Step 3: Implement.** Replace `PerSignalLoss.forward` (keep
  `__init__`, `_component`, `extra_repr`, and everything above the class
  untouched):

```python
    def forward(self, preds, labels, label_mask):
        """Unweighted masked components per signal — contract v2's raw_losses.

        Reads    : preds, labels, label_mask (all keyed by signal)
        Returns  : {signal: {component: () tensor}}, graph-attached.
        Weighting is the trainer's job — see :func:`weight_losses`.
        """
        raw = {}
        for signal in self.traces:
            pred, label = preds[signal], labels[signal]
            mask = label_mask[signal].to(pred.dtype)                  # (B,)
            # Clamped denominator: a signal absent from every window in the
            # batch contributes exactly 0 instead of 0/0.
            denominator = mask.sum().clamp(min=1.0)
            raw[signal] = {
                component: (self._component(component, pred, label) * mask).sum()
                           / denominator
                for component in self.specs[signal]['weights']
            }
        return raw
```

  and add, at module level (below the class):

```python
def weight_losses(raw, weights):
    """Apply config weights to a model's raw_losses dict.

    ``raw``: ``{module: {component: () tensor}}`` (unweighted, graph-attached).
    ``weights``: ``{module: {component: float}}``; a missing entry means 1.0.
    Returns ``(total, weighted)``: the scalar to backpropagate — the mean over
    modules of each module's weighted component sum — and a float mirror of
    ``raw`` with a ``'total'`` per module, for logging.
    """
    module_totals, weighted = [], {}
    for module, components in raw.items():
        if not components:            # a fully zero-weighted spec: contributes 0
            weighted[module] = {'total': 0.0}
            continue
        module_weights = weights.get(module, {})
        module_total = None
        entries = {}
        for component, value in components.items():
            term = module_weights.get(component, 1.0) * value
            entries[component] = float(term.detach())
            module_total = term if module_total is None else module_total + term
        entries['total'] = float(module_total.detach())
        weighted[module] = entries
        module_totals.append(module_total)
    return torch.stack(module_totals).mean(), weighted
```

- [x] **Step 4: Run the loss tests**

Run: `uv run python -m pytest tests/test_per_signal_loss.py -q`
Expected: PASS. (Trainer and model tests will fail until Tasks 6–7 — do not
run the full suite yet.)

- [x] **Step 5: Commit**

```bash
git add neural_methods/loss/PerSignalLoss.py tests/test_per_signal_loss.py
git commit -m "feat(loss): raw per-signal components; weighting split into weight_losses"
```

### Task 6: `DictModel` computes and carries `raw_losses`

**Files:**
- Modify: `neural_methods/model/DictModel.py` (forward at line 97)
- Modify: `neural_methods/trainer/MultiSignalTrainer.py` — only
  `build_model` (line ~247) in this task
- Test: `tests/test_batch_contract.py`

**Interfaces:**
- Consumes: Task 4's `RAW_LOSSES`; Task 5's `PerSignalLoss` raw forward.
- Produces (used by Task 7):
  - `DictModel.forward(batch)` dict branch now also writes
    `out[RAW_LOSSES] = {**self.loss(...), **self.stage_losses(out)}`.
  - `DictModel.loss_modules() -> tuple[str, ...]` — stage names beyond the
    signals; base returns `()`.
  - `DictModel.stage_losses(out) -> dict` — base returns `{}`; a composite
    model (PhysHydra, later) overrides both.
  - `DictModel.attach_loss(loss: PerSignalLoss) -> None` — replaces the
    default criterion; called by `build_model` so config `TRAIN.LOSS`
    overrides reach the model **before** any DDP wrap.

- [x] **Step 1: Write the failing test** — append to
  `tests/test_batch_contract.py` (reuse whatever tiny concrete model or
  batch-building helper that file already uses; the assertions are what
  matter):

```python
def test_forward_writes_raw_losses(tiny_model_and_batch):
    from neural_methods.batch import RAW_LOSSES
    model, batch = tiny_model_and_batch
    out = model(batch)
    assert RAW_LOSSES in out
    for signal in model.traces:
        assert signal in out[RAW_LOSSES]
        for value in out[RAW_LOSSES][signal].values():
            assert value.ndim == 0
    assert model.loss_modules() == ()
```

  If no such fixture exists, build the model/batch inline the same way the
  file's existing forward test does — copy that construction verbatim.

- [x] **Step 2: Run, expect KeyError/AttributeError**

Run: `uv run python -m pytest tests/test_batch_contract.py -q`

- [x] **Step 3: Implement in `DictModel`:**

In `__init__`, after the `_fs` buffer:

```python
        # The model's own criterion (contract v2: losses are computed inside
        # the model and ride the batch). Class defaults now; build_model
        # swaps in the config-resolved one via attach_loss. PerSignalLoss
        # holds no parameters, so this never touches the state_dict.
        self.loss = PerSignalLoss(self.traces, fs=float(fs) or None)
```

(import `PerSignalLoss` at the top of the module, and `LABELS`,
`LABEL_MASK`, `RAW_LOSSES` from `neural_methods.batch`.)

New methods:

```python
    def attach_loss(self, loss):
        """Swap in the config-resolved criterion (build_model calls this)."""
        self.loss = loss

    def loss_modules(self):
        """Stage-loss names beyond the per-signal entries. Base: none."""
        return ()

    def stage_losses(self, out):
        """Extra raw stage losses, keyed by loss_modules() names. Base: none.

        Reads    : whatever intermediate keys the model added to ``out``
        Returns  : {stage: {component: () tensor}}
        """
        return {}
```

Replace the dict branch of `forward`:

```python
        out = {**require_batch_dict(batch), PREDICTIONS: self.predict(batch)}
        out[RAW_LOSSES] = {
            **self.loss(out[PREDICTIONS], out[LABELS], out[LABEL_MASK]),
            **self.stage_losses(out),
        }
        return out
```

Also update the module docstring's contract summary (lines 1–19) to name
both added keys, and give `forward` a `Reads:/Modifies:` docstring — the
convention every module adopts from here on:

```python
        Reads    : batch["frames"], batch["labels"], batch["label_mask"]
        Modifies : batch["predictions"], batch["raw_losses"]
        Returns  : the same dict
```

In `MultiSignalTrainer.build_model` (line ~247), attach the resolved loss
after construction, before returning:

```python
    model.attach_loss(PerSignalLoss(
        spec.traces, specs=getattr(config.TRAIN, 'LOSS', None) or None,
        fs=spec.fs))
```

**Caveat:** `TRAIN.LOSS` may now legally contain *stage* keys (Task 7);
`resolve_loss_specs` raises on names outside `traces`. So in `build_model`
filter to signal keys first:

```python
    overrides = dict(getattr(config.TRAIN, 'LOSS', None) or {})
    stage_names = set(model.loss_modules())
    signal_overrides = {k: v for k, v in overrides.items()
                        if k not in stage_names} or None
    model.attach_loss(PerSignalLoss(spec.traces, specs=signal_overrides,
                                    fs=spec.fs))
```

- [x] **Step 4: Run the model-side tests**

Run: `uv run python -m pytest tests/test_batch_contract.py tests/test_deepphys_multisignal.py tests/test_physformer_multisignal.py tests/test_physmamba_dict.py -q`
Expected: PASS (models still return the batch; the extra key is additive).
Fix any test that asserted the exact output key set.

- [x] **Step 5: Commit**

```bash
git add neural_methods/model/DictModel.py neural_methods/trainer/MultiSignalTrainer.py tests/test_batch_contract.py
git commit -m "feat(model): DictModel computes raw_losses inside forward"
```

### Task 7: Trainer weights, sums, and logs the two loss dicts

**Files:**
- Modify: `neural_methods/trainer/MultiSignalTrainer.py` — criterion setup
  (lines 332–336), `_loss_for` (line 388), `_accumulate` (line 396) and the
  per-signal loss-curve plotting (lines ~620–660)
- Test: `tests/test_multisignal_trainer.py`

**Interfaces:**
- Consumes: Task 5's `weight_losses`; Task 6's model-side `raw_losses` +
  `loss_modules()`.
- Produces: `self.loss_weights: dict[str, dict[str, float]]` (signals +
  stages); `_loss_for(batch) -> (total, weighted, out)` where
  `out[LOSSES] = weighted`; training/validation logs now accumulate **both**
  raw and weighted components.

- [x] **Step 1: Update the trainer tests.** In
  `tests/test_multisignal_trainer.py`, any test reaching into
  `trainer.criterion` moves to `trainer.loss_weights`; add one assertion to
  the existing one-training-step test:

```python
    from neural_methods.batch import LOSSES, RAW_LOSSES
    # inside the existing step test, after a forward/step:
    assert RAW_LOSSES in out and LOSSES in out
    for module, entries in out[LOSSES].items():
        assert "total" in entries
```

- [x] **Step 2: Implement.** Replace the criterion block (lines 332–336):

```python
        # Contract v2: the model computes raw_losses; the trainer only
        # weights and sums. Signals resolve through the per-signal registry
        # (class defaults + TRAIN.LOSS overrides); stage names come from the
        # model, WEIGHTS only. Anything else in TRAIN.LOSS is a config error.
        overrides = dict(getattr(config.TRAIN, 'LOSS', None) or {})
        stage_names = tuple(self._unwrap_model().loss_modules())
        stage_overrides = {k: overrides.pop(k) for k in list(overrides)
                           if k in stage_names}
        signal_specs = resolve_loss_specs(self.traces, overrides or None)
        self.loss_weights = {s: dict(spec['weights'])
                             for s, spec in signal_specs.items()}
        for stage, spec in stage_overrides.items():
            if set(spec or {}) - {'WEIGHTS'}:
                raise ValueError(
                    f"TRAIN.LOSS[{stage}] is a model stage: WEIGHTS only "
                    f"(the model defines what its stage losses are); got "
                    f"{sorted(set(spec) - {'WEIGHTS'})}")
            self.loss_weights[stage] = {
                str(c).lower(): float(w)
                for c, w in dict((spec or {}).get('WEIGHTS') or {}).items()}
        if self.is_main:
            listing = "\n".join(
                f"{module}: " + ", ".join(f"{c}={w:g}" for c, w in ws.items())
                for module, ws in self.loss_weights.items())
            print(f"Loss weights per module:\n{listing}")
```

(import `resolve_loss_specs` and `weight_losses` from
`neural_methods.loss.PerSignalLoss`, and `LOSSES`, `RAW_LOSSES` from
`neural_methods.batch`; `resolve_loss_specs` already raises for a name that
is neither a trace nor — after the pop above — a stage.)

Replace `_loss_for`:

```python
    def _loss_for(self, batch):
        """Forward one batch; weight the model's raw losses into the total.

        Reads    : the model's out["raw_losses"]
        Modifies : out["losses"] (weighted mirror, floats, with totals)
        Returns  : (total, weighted, out)
        """
        out = self.model(batch)
        total, weighted = weight_losses(out[RAW_LOSSES], self.loss_weights)
        out[LOSSES] = weighted
        return total, weighted, out
```

Callers of `_loss_for` treated `breakdown` as `{signal: {component: float}}`
— `weighted` has exactly that shape (plus `'total'` entries, which the old
breakdown also had), so `_accumulate` and the loss-curve code keep working.
Extend logging to raw as well: accumulate a second totals dict from
`{m: {c: float(v.detach()) for c, v in comps.items()} for m, comps in out[RAW_LOSSES].items()}`,
and in the per-signal component plot draw the raw curve dashed on the same
axes (label `f"{component} (raw)"`). The weighted curves stay exactly as
they are — this is additive.

One more site: line ~639 titles subplots via `self.criterion.specs`; take
the type from `resolve_loss_specs`'s output instead — keep `signal_specs`
on `self` as `self.signal_specs` for that.

- [x] **Step 3: Run the trainer + main tests**

Run: `uv run python -m pytest tests/test_multisignal_trainer.py tests/test_main.py tests/test_legacy_contract.py -q`
Expected: PASS (rerun the two known Windows flakes in isolation if they
error in a full run).

- [x] **Step 4: Run the whole suite**

Run: `uv run python -m pytest -q`
Expected: 289+ passing.

- [x] **Step 5: Commit**

```bash
git add neural_methods/trainer/MultiSignalTrainer.py tests/test_multisignal_trainer.py
git commit -m "feat(trainer): weight and log the model's raw_losses; drop the owned criterion"
```

### Task 8: `ParallelSignals` — style C

**Files:**
- Create: `neural_methods/model/ParallelSignals.py`
- Test: `tests/test_batch_contract.py` (append one test; no new file)

**Interfaces:**
- Consumes: `DictModel` (Task 6 state), `SignalDictWrapper` (unchanged).
- Produces (used by Task 9):
  `ParallelSignals(make_copy, channels, traces, frame_transform=None, fs=0.0)`
  — `make_copy(trace: str) -> module` returns a **single-trace** object
  exposing `forward_video((B, C_in, T, H, W)) -> (B, 1, T)` and
  `output_layers()` (any `DictModel`, including a `SignalDictWrapper`,
  qualifies). The parent applies the frame transform once; a copy's own
  `frame_transform`/`prepare_frames` is never invoked.

- [x] **Step 1: Write the failing test** — append to
  `tests/test_batch_contract.py`:

```python
def test_parallel_signals_is_one_copy_per_trace():
    import torch
    from neural_methods.model.ParallelSignals import ParallelSignals
    from neural_methods.model.SignalDictWrapper import SignalDictWrapper

    def make_copy(trace):
        backbone = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 1))
        return SignalDictWrapper(backbone, channels=("R", "G", "B"),
                                 traces=[trace], input_mode='frames2d')

    model = ParallelSignals(make_copy, channels=("R", "G", "B"),
                            traces=("ABP", "CVP"))
    video = torch.randn(2, 3, 5, 4, 4)               # (B, C, T, H, W)
    out = model.forward_video(video)
    assert out.shape == (2, 2, 5)                     # (B, S, T)
    assert len(model.copies) == 2
    # Copies are independent: zeroing one branch only kills its signal.
    with torch.no_grad():
        for parameter in model.copies[0].parameters():
            parameter.zero_()
    out = model.forward_video(video)
    assert torch.all(out[:, 0] == 0) and not torch.all(out[:, 1] == 0)
```

- [x] **Step 2: Run, expect ModuleNotFoundError**

Run: `uv run python -m pytest tests/test_batch_contract.py::test_parallel_signals_is_one_copy_per_trace -v`

- [x] **Step 3: Implement `neural_methods/model/ParallelSignals.py`:**

```python
"""Style C: S complete copies of a single-signal architecture, in parallel.

Contract v2's default multi-signal form
(docs/plans/2026-09-01-contract-v2-design.md, Part 2): each predicted signal
gets its own untouched copy of the published architecture — input widened to
the demanded channels, everything downstream true to the paper. Cost is
parameters, by design.
"""
import torch
import torch.nn as nn
from einops import rearrange  # noqa: F401  (kept: einops is the reshape idiom)

from neural_methods.model.DictModel import DictModel


class ParallelSignals(DictModel):
    """One single-trace copy per signal, presented as one DictModel.

    ``make_copy(trace)`` builds the copy for one signal: any object with
    ``forward_video((B, C_in, T, H, W)) -> (B, 1, T)`` and
    ``output_layers()``. The parent owns the frame transform and applies it
    once; a copy's own transform is never invoked.
    """

    def __init__(self, make_copy, channels, traces, frame_transform=None,
                 fs=0.0):
        super().__init__(channels=channels, traces=traces,
                         frame_transform=frame_transform, fs=fs)
        self.copies = nn.ModuleList([make_copy(trace) for trace in self.traces])
        # Window constraints are per-architecture, so every copy agrees;
        # surface the first copy's so build-time checks see them.
        first = self.copies[0]
        self.temporal_divisor = getattr(first, 'temporal_divisor', 1)
        self.temporal_length = getattr(first, 'temporal_length', None)

    def output_layers(self):
        """One readout per copy, in traces order — the per-signal shape that
        init_output_bias and the weight-decay exemption already handle."""
        return [layer for copy in self.copies for layer in copy.output_layers()]

    def forward_video(self, video):
        """Reads: the transformed clip. Returns: (B, S, T), traces order."""
        return torch.cat([copy.forward_video(video) for copy in self.copies],
                         dim=1)

    def extra_repr(self):
        return f"{super().extra_repr()}, copies={len(self.copies)}"
```

- [x] **Step 4: Run the test**

Run: `uv run python -m pytest tests/test_batch_contract.py::test_parallel_signals_is_one_copy_per_trace -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add neural_methods/model/ParallelSignals.py tests/test_batch_contract.py
git commit -m "feat(model): ParallelSignals wrapper (style C)"
```

### Task 9: `parallel` becomes the default head style; the three builders learn it

**Files:**
- Modify: `config.py:126` (`HEAD_STYLE: str = "widened"` → `"parallel"`)
- Modify: `neural_methods/trainer/MultiSignalTrainer.py` — `model_spec`
  fallback (line 138: `'widened'` → `'parallel'`) and the three builders
  (lines 145–191)
- Delete: `neural_methods/trainer/PhysMambaTrainer.py`
- Test: existing `tests/test_deepphys_multisignal.py`,
  `tests/test_physformer_multisignal.py`, `tests/test_physmamba_dict.py`

**Interfaces:**
- Consumes: Task 8's `ParallelSignals`.
- Produces: `HEAD_STYLE` values `parallel` (default) / `widened` /
  `per_signal`; each builder honours all values it supports.

- [x] **Step 1: Add the parallel paths.** In `MultiSignalTrainer.py`,
  above the builders:

```python
def _parallel(spec, make_copy):
    from neural_methods.model.ParallelSignals import ParallelSignals
    return ParallelSignals(make_copy, channels=spec.channels,
                           traces=spec.traces, frame_transform=spec.transform,
                           fs=spec.fs)
```

  `_build_physmamba` becomes:

```python
def _build_physmamba(config, spec):
    from neural_methods.model.PhysMamba import PhysMamba
    if spec.head_style == 'parallel':
        return _parallel(spec, lambda trace: PhysMamba(
            channels=spec.channels, traces=[trace],
            frame_transform=spec.transform, fs=spec.fs))
    return PhysMamba(channels=spec.channels, traces=spec.traces,
                     frame_transform=spec.transform, fs=spec.fs)
```

  `_build_deepphys`: keep the existing square-frame and two-`DATA_TYPE`
  checks exactly as they are, then:

```python
    if spec.head_style == 'parallel':
        def make_copy(trace):
            copy = DeepPhys(in_channels=spec.camera_channels, out_signals=1,
                            img_size=height, head_style='widened')
            return SignalDictWrapper(copy, channels=spec.channels,
                                     traces=[trace], input_mode='frames2d',
                                     frame_transform=spec.transform, fs=spec.fs)
        return _parallel(spec, make_copy)
    backbone = DeepPhys(in_channels=spec.camera_channels, ...)   # unchanged path
```

  `_build_physformer`: change the guard — `parallel` and `widened` are
  legal, `per_signal` keeps the existing refusal message — then:

```python
    if spec.head_style == 'parallel':
        return _parallel(spec, lambda trace: PhysFormer(
            channels=spec.channels, traces=[trace],
            frame_transform=spec.transform, fs=spec.fs,
            image_size=(spec.window, height, width),
            patches=int(block.PATCH_SIZE), dim=int(block.DIM),
            ff_dim=int(block.FF_DIM), num_heads=int(block.NUM_HEADS),
            num_layers=int(block.NUM_LAYERS), theta=float(block.THETA),
            dropout_rate=float(config.MODEL.DROP_RATE)))
```

  (hoist `height, width = spec.img_size` and `block = config.MODEL.PHYSFORMER`
  above the guard so both paths share them.)

- [x] **Step 2: Flip the defaults** — `config.py:126` and
  `model_spec`'s `'widened'` fallback (line 138) both become `'parallel'`.

- [x] **Step 3: Run the model + trainer + config tests; fix expectations**

Run: `uv run python -m pytest tests/test_deepphys_multisignal.py tests/test_physformer_multisignal.py tests/test_physmamba_dict.py tests/test_multisignal_trainer.py tests/test_config_keys.py -q`

Tests that build via configs now get `ParallelSignals` models; tests that
assert `state_dict` key names or `output_layers()` counts for the widened
shape must either pin `HEAD_STYLE: widened` in their config fixture (when
the test is *about* the widened head) or update expectations to the
parallel shape (when the test is about the contract). Choose per test;
`init_output_bias` needs no change — S copies × 1 readout each hits its
existing per-signal branch.

- [x] **Step 4: Delete the dead legacy trainer**

```bash
git rm neural_methods/trainer/PhysMambaTrainer.py
```

Then `grep -rn "PhysMambaTrainer" --include=*.py .` — remove any import
(check `neural_methods/trainer/__init__.py`); expect none in reachable code.

- [x] **Step 5: Full suite**

Run: `uv run python -m pytest -q`
Expected: all pass (modulo the two known Windows flakes — rerun in
isolation).

- [x] **Step 6: Real-cache smoke run** (GPU box; the v1 cache still works —
  Part 3 hasn't landed):

Run: `uv run python main.py --limit_windows 8 --test_participants P015 --config_file configs/neckflix/NECKFLIX_DEEPPHYS_SMOKE.yaml`
Expected: trains, tests, writes the report + plots; the startup print shows
`Loss weights per module:` and the model repr shows `ParallelSignals`.
Repeat with `NECKFLIX_PHYSMAMBA_SMOKE.yaml`.

- [x] **Step 7: Commit**

```bash
git add config.py neural_methods/trainer/MultiSignalTrainer.py tests/
git commit -m "feat(model): style-C parallel copies as the default head style"
```

### Task 10: Migration-contract doc update

**Files:**
- Modify: `docs/plans/2026-08-31-model-migration-contract.md`

This must land **before** any Part-C migration agent is dispatched — they
implement from this document.

- [x] **Step 1: Update the contract doc.** Precise edits, keeping its
  structure:
  - The prediction contract section: `forward(batch) -> batch` now also
    writes `raw_losses` (base-computed; a migration writes no loss code)
    and the trainer writes `losses`; cite
    `docs/plans/2026-09-01-contract-v2-design.md` Part 2 as normative.
  - Head styles: **style C (`HEAD_STYLE: parallel`, the default)** — build
    a single-trace copy per signal via `ParallelSignals` and a
    `make_copy(trace)` in the builder, exactly as
    `_build_deepphys`/`_build_physmamba`/`_build_physformer` now do (point
    at them as the worked examples); styles A/B remain options where an
    architecture supports them.
  - The per-model recipe: step "delete the legacy trainer" unchanged; add
    "declare `loss_modules()`/`stage_losses()` only if the architecture has
    internal stages (rare; PhysHydra)".
  - Add the `Reads:/Modifies:` docstring convention to the checklist.

- [x] **Step 2: Self-check** — grep the contract doc for `criterion`,
  `MaskedMultiSignalLoss`, "trainer computes": no stale statements that the
  trainer owns the loss.

- [x] **Step 3: Commit**

```bash
git add docs/plans/2026-08-31-model-migration-contract.md
git commit -m "docs: migration contract updated to contract v2"
```

---

## Part 3 — Reader adoption (GATED — do not start until a regenerated v2 cache passes the validator)

Confirm the gate first: `uv run python tools/validate_cache.py <new-cache-path>`
prints all-PASS. The user supplies the path.

### Task 11: `BaseZarrDataset` reads v2; one dataset class

**Files:**
- Modify: `tests/zarr_fixtures.py` — rewrite `make_store` as a thin alias of
  `make_v2_store` (same keyword surface the tests use: `streams=` becomes
  `modalities=`; keep a `streams` alias parameter so call sites need not all
  change at once), then update call sites that poke v1 internals.
- Modify: `dataset/data_loader/zarr_dataset.py`:
  - delete the admission block (lines ~173–230: `complete`/`tool_version`)
    and its docstring claims;
  - `channel_map` stops being abstract: implement it on the base from
    `MODALITY_CHANNELS` —

```python
    @property
    def channel_map(self) -> dict:
        """Global contract-v2 table: channel name -> (modality, plane)."""
        from neural_methods.signals import MODALITY_CHANNELS
        return {channel: (modality, index)
                for modality, channels in MODALITY_CHANNELS.items()
                if channels is not None
                for index, channel in enumerate(channels)}
```

  - every `["video"]["frames"]` read becomes `["video"]["data"]`
    (lines 547, 640; grep for the rest);
  - `_native_fps` (line 372) reads the **perspective attr** `fps`
    (`cam.attrs["fps"]`) instead of per-stream `video.attrs["fps"]`; keep
    `same_nominal_rate` reconciliation against `target_fps` unchanged;
  - trace reads move to `TRACE_KEYS` naming (`rr` → RESP etc.);
  - length reconciliation: where a sample's modalities disagree on `T`,
    truncate every stream and its index-aligned traces to the shortest
    before windowing;
  - `__getitem__` adds `LABEL_UNITS`: `{signal: str}` from each trace
    group's `units` attr; refuse (raise `ValueError` naming both stores) if
    one signal's units differ across admitted stores;
  - make the class concrete: rename to `ZarrDataset` (keep the module),
    `ABC` import dropped.
- Delete: `dataset/data_loader/NeckflixLoader.py`,
  `dataset/data_loader/PURELoader.py` (`git rm`).
- Modify: `main.py` — construct `ZarrDataset` where it now names
  `NeckflixDataset`; delete `DATA.DATASET` from `config.py`'s schema and
  from the seven `configs/neckflix/*.yaml` (coordinate with the user's
  unstaged config edits — ask before touching the four modified YAMLs).
- Modify: `tests/test_neckflix_zarr.py`, `tests/test_main.py`,
  `tests/test_legacy_contract.py`, `tests/test_config_keys.py` — follow the
  renames; drop admission-rejection tests (delete, don't port — the
  validator owns admission now); add ONE test: a store with mismatched
  modality lengths yields truncated, equal-length streams and a
  `label_units` entry.

- [ ] Steps: update fixtures → run `tests/test_neckflix_zarr.py` (fails) →
  implement the reader changes → run
  `uv run python -m pytest tests/test_neckflix_zarr.py tests/test_main.py tests/test_config_keys.py tests/test_legacy_contract.py -q`
  → full suite → commit:

```bash
git add -A dataset/data_loader tests config.py main.py
git commit -m "feat(dataset): contract-v2 reader; one dataset class"
```

(`git add -A` scoped to those paths only; still never the user's unstaged
config YAMLs unless the DATASET-key removal was agreed — if it was, stage
them explicitly and say so in the commit body.)

### Task 12: `label_units` through evaluation

**Files:**
- Modify: `neural_methods/trainer/MultiSignalTrainer.py` (~line 567: the
  per-window record dict gains `'units': sample[LABEL_UNITS][signal]`)
- Modify: `evaluation/records.py`, `evaluation/report.py`,
  `evaluation/plots.py` — carry a per-signal `units` column into the tidy
  frame and use it for axis labels/digest lines wherever
  `neural_methods.signals.signal_unit` is used today (grep
  `signal_unit`); fall back to `signal_unit(sig)` when a record predates
  the column.
- Test: extend one existing case in `tests/test_evaluation_report.py` to
  assert the unit string flows through to the frame.

- [ ] Steps: failing test → implement → run
  `uv run python -m pytest tests/test_evaluation_report.py tests/test_evaluation_records.py -q`
  → commit `feat(evaluation): per-signal units from the cache`.

### Task 13: PURE bridge writes v2; cache specs updated

**Files:**
- Modify: `tools/cache_pure.py` — emit v2 stores: root `participant`
  (unprefixed), perspective `"1"` with `fps` attr, `rgb/` modality with
  real `timestamps_us` (the PURE image timestamps it already parses),
  `video/data`, `ppg/data` with `units: "arb"`. Drop
  `complete`/`tool_version` attrs.
- Modify: `dataset/data_loader/PURE.md` + the other eleven cache-spec
  `.md` files: one v2 layout section each (layout tree + attrs), replacing
  the v1 store description; PURE.md documents the implemented mapping.
- Test: `tests/test_pure_cache.py` — point its assertions at the v2 layout;
  finish with `validate_store` over the freshly written store:

```python
    from tools.validate_cache import validate_store
    assert validate_store(store_path) == []
```

- [ ] Steps: failing test → implement → run
  `uv run python -m pytest tests/test_pure_cache.py tests/test_validate_cache.py -q`
  → commit `feat(tools): PURE bridge writes contract-v2 stores`.

### Task 14: Part-3 gate verification

- [ ] Run the validator over the regenerated Neckflix cache: all PASS.
- [ ] Full suite: `uv run python -m pytest -q`.
- [ ] End-to-end smoke on the new cache (config `CACHED_PATH` re-pointed by
  the user): `uv run python main.py --limit_windows 8 --test_participants P015 --config_file configs/neckflix/NECKFLIX_PHYSMAMBA_SMOKE.yaml`
  — train, test, report, plots; digest shows real units.
- [ ] Update `docs/project_status.md` (Phase A complete) and commit
  `docs: Phase A reader adoption complete`.

---

## Self-review notes (already applied)

- Spec coverage: Part 1 ↔ spec §"The validator"/"Root"/"Perspective"/
  "Modality"/"Trace"; Task 11 ↔ §"Reader consequences"; Tasks 4–9 ↔ spec
  Part 2; Task 12 ↔ spec Part 3 units; templates and CLAUDE.md/architecture
  rewrites are Phase E (roadmap) — deliberately not in this plan.
- The `losses`-total rule (mean over modules) is stated in Task 5 and
  matches today's mean-over-signals exactly when no stages exist.
- `HEAD_STYLE: parallel` never reaches a backbone constructor — builders
  translate it (Task 9); `DeepPhys.HEAD_STYLES` stays `('widened',
  'per_signal')`.
