# Contract v2: the cache and the model (2026-09-01)

Design record for the two contract revisions adopted from
[the revised overhaul plan](../../revised_overhaul_plan.md), which supersedes
`updating_plan.md`. This document is what the cache validator, the reader
changes, and every subsequent model migration implement against. The
[roadmap](2026-08-31-overhaul-roadmap.md) carries the ordering; the
[migration contract](2026-08-31-model-migration-contract.md) is updated to
match before any migration runs against this.

Decisions here were settled interactively on 2026-09-01 and are recorded in
the roadmap's decision log (entry 17).

---

## Part 1 — Cache contract v2

One zarr store per recording, written by an external preprocessor (for
Neckflix: the Neckflix repo's `neckflix-preprocess`, being updated to this
contract in parallel with this repo). This repo never writes the cache.

### Layout

```text
{recording}.zarr
+-- attrs                          # root: participant (required) + free attrs
+-- <perspective>/                 # "1", "2", ... one per camera viewpoint
    +-- attrs                      # fps (required, nominal, all modalities)
    +-- <modality>/                # from the modality vocabulary below
        +-- timestamps_us/data     # (T,) microseconds
        +-- video/data             # (C, T, H, W) uint8
        +-- <trace>/data           # (T,) float; attrs: units (required)
```

### Root

- **Required attrs**: `participant` — unprefixed (`"015"`, not `P015`); the
  key the split machinery (LOSO, `PARTICIPANTS`, `--test_participants`)
  operates on.
- Any further attrs are allowed and are what `DATA.FILTERS` filters on
  (`posture`, `light`, ...). None are required.
- **`complete` and `tool_version` are gone.** Stores are assumed complete;
  admission is the offline validator (below), run after generating a cache,
  not a per-attr check in the reader.

### Perspective

A perspective is a set of modalities whose pixels are physically aligned
(e.g. Neckflix reprojects IR and depth into the RGB camera's frame). Treated
as independent samples by the loader, as today.

- **Required attrs**: `fps` — the nominal rate for **all** modalities in the
  perspective. Per-modality exact rates may drift (decision 14's
  `same_nominal_rate` reconciliation survives unchanged); `fps` is the
  nominal identity.
- **Alignment**: the first frame of every modality lies within `1/fps` of
  every other's, judged by `timestamps_us`.

### Modality

Fixed vocabulary — a store may carry any subset, nothing outside it:

| key     | meaning          | channels (C)   | canonical channel names |
| ------- | ---------------- | -------------- | ----------------------- |
| `gr`    | grayscale camera | 1              | `Y`                     |
| `rgb`   | RGB video        | 3              | `R`, `G`, `B`           |
| `ir`    | infrared         | 1              | `I`                     |
| `depth` | depth            | 1              | `D`                     |
| `t`     | thermal          | 1              | `T`                     |
| `ev`    | event camera     | not yet pinned | not yet pinned          |

This table lives once, in code, as the global channel map — it is what makes
per-dataset `channel_map` subclasses unnecessary (see Reader consequences).
`ev`'s frame representation is pinned when the first event cache exists; the
validator accepts any C for `ev` with a note until then.

Each modality group holds:

- `timestamps_us/data` — `(T,)`, microseconds, strictly increasing. Each
  sensor keeps its own clock; this is the record of it.
- `video/data` — `(C, T, H, W)` uint8, stacked in exactly that order.
  (`video/frames` was the v1 name; v2 is `data`, uniform with every other
  array.)
- One group per trace the recording carries (vocabulary below), index-aligned
  to **this modality's** frames: trace `T` equals video `T`.

**Within a perspective, every modality carries the identical trace set.**
Across modalities, `T` may differ (a sensor that died early); each modality
is internally consistent (`timestamps_us`, `video`, and every trace agree on
`T`), and the loader reconciles lengths at read time by truncating to the
shortest (see Reader consequences).

### Trace

Fixed vocabulary:

| key   | meaning                 | repo signal name |
| ----- | ----------------------- | ---------------- |
| `ecg` | ECG                     | `ECG`            |
| `abp` | arterial blood pressure | `ABP`            |
| `cvp` | central venous pressure | `CVP`            |
| `ppg` | finger PPG              | `PPG`            |
| `rr`  | respiration (chest)     | `RESP`           |

The cache-to-repo name mapping is one table in `neural_methods/signals.py`,
next to the signal classes.

- `data` — `(T,)` floating point, physical units, index-aligned to the
  modality's frames.
- **Required attrs**: `units` — a string (`"mmHg"`, `"cmH2O"`, `"arb"`, ...).
  `"arb"` is the expected value for shape-class signals. The loader carries
  it through to reporting (`label_units`); a signal whose units differ
  *across stores in one run* is refused rather than silently mixed — unit
  conversion is added only if that ever actually occurs.

### The validator (admission, made executable)

`tools/validate_cache.py` — a CLI plus an importable function:

```text
uv run python tools/validate_cache.py <cache-dir | store.zarr ...>
```

- Walks `*.zarr` under a directory (or takes explicit stores). Per store:
  every contract clause above is checked — required attrs, vocabulary
  membership, shapes, dtypes, `T` consistency within each modality,
  identical trace sets across modalities, `units` present, timestamps
  strictly increasing, first-frame alignment within `1/fps`.
- Prints one PASS line or an itemised FAIL per store; exits non-zero if any
  store failed. Importable as `validate_store(path) -> list[Violation]` so
  the preprocessor repo can call it directly at write time.
- This is the **only** admission mechanism: the reader trusts what it opens.
- Testing: one smoke test (a fixture store made invalid in a few distinct
  ways), per the minimal-testing rule.

### Reader consequences (`dataset/data_loader/`)

- `BaseZarrDataset` reads `video/data`, perspective-level `fps`, and
  `timestamps_us`; the `complete`/`tool_version` admission check is deleted.
- Unequal modality durations are reconciled at read time: frames (and their
  index-aligned traces) are dropped from the longer arrays so every modality
  in a sample spans the same window.
- **One dataset class.** The global modality-to-channel table replaces the
  per-dataset `channel_map` property; `NeckflixDataset` and `PUREDataset`
  are deleted. A new dataset is now a cache-writer (external) plus a
  markdown cache spec in `dataset/data_loader/` — the specs stay, and are
  updated to describe v2 stores.
- `label_units` (`{signal: str}`) joins the batch dict beside `label_stats`,
  populated from the traces' `units` attrs.

---

## Part 2 — Model contract v2

Adopted from CardioHydra's batch-pipeline pattern
(github.com/coenarrow/CardioHydra): the batch dict flows through the model,
modules add their outputs to it, and **losses are computed inside the model
and ride the batch** — one structure for every model, from a style-C
DeepPhys to PhysHydra.

### The forward contract

```python
batch = model(batch)
batch["predictions"]   # {signal: (B, T)}
batch["raw_losses"]    # {module: {component: () tensor}} — unweighted, model-written
batch["losses"]        # same structure, weighted — trainer-written
```

- `forward(batch) -> batch`: nothing is dropped in transit; a model may add
  intermediate keys (CWT stacks, masks, kinematics, ...) beside the
  required ones.
- Two loss keys, as in CardioHydra, so weights stay calibratable: the model
  writes `raw_losses` (unweighted components); the trainer applies the
  config weights and writes `losses` beside it, and the training total is
  the sum of `losses`. Logging carries both — comparing a component's raw
  curve against its weighted contribution is how a drowned or dominating
  term is spotted and the weights re-tuned.
- `raw_losses` carries **unweighted** per-component values. The `DictModel` base
  contributes one module entry per predicted signal, named by the signal
  (`"ABP": {"ccc": ..., "mean": ...}`), by invoking the shared per-signal
  machinery (`PerSignalLoss`, the `TRAIN.LOSS` registry, signal-class
  defaults — all unchanged) — so a simple migration inherits its loss
  reporting from the base and writes nothing. A composite model (PhysHydra)
  adds stage entries beside the signal entries
  (`"pulsatility_mask": {...}`). The masked structure is retained: an absent
  signal's components are exactly 0.
- Weights stay in config. `TRAIN.LOSS` keys per-signal weights as today;
  stage entries are weighted by the same registry keyed by stage name —
  `WEIGHTS` only, no `TYPE` (the model defines what a stage loss *is*; the
  config only scales it). A model declares its stage names
  (`loss_modules()`) so config validation can keep refusing unknown keys.
- Every module documents its batch contract in its docstring —
  `Reads: / Modifies:` — the CardioHydra convention, adopted repo-wide.

### Style C — parallel per-signal copies (the default)

For migrated single-signal architectures, the default multi-signal form is
**S complete copies of the original architecture**, one per predicted
signal, presented as one `DictModel`:

- Each copy takes the **full** demanded input stack (the input layer widens
  from the published 3 channels to `spec.in_channels` /
  `spec.camera_channels`; mostly 5: R,G,B,I,D); everything after the input
  layer stays true to the published architecture.
- Each copy emits `(B, 1, T)`; the wrapper concatenates to `(B, S, T)` in
  `model.traces` order.
- Cost is parameters, by design — the architecture stays honest to the
  paper.
- `MODEL.HEAD_STYLE`: `parallel` (style C, **the new default**), `widened`
  (A), `per_signal` (B) — A and B remain available options.
- The absolute-scale guardrails (output-bias priors, weight-decay exemption,
  activation-free readout) apply per copy, unchanged.

### Trainer consequences

`MultiSignalTrainer` slims to: forward, weight `batch["raw_losses"]` into
`batch["losses"]`, sum, step, log both. It no longer owns a loss module.
Per-component curves (now raw and weighted), the plot set, evaluation, and
checkpointing (`INTERFACE` serialization) are unchanged — the two flat loss
dicts are also the single hook point for any future tracker (wandb), which
is part of why they exist.

### Rework of the already-migrated models

DeepPhys, PhysFormer, and PhysMamba move onto this contract (style C
default, base-computed losses) before any further migration; each rework
ends with a smoke run. PhysFormer's "style B unavailable" finding
(decision 13) becomes moot under style C — full copies always compose.

---

## Part 3 — Consequences elsewhere

- **Evaluation**: reports print each signal's `units` from the cache instead
  of assuming mmHg. ISO 81060 threshold verification is unblocked (the
  purchased texts are in `standards/ISO-81060/`); IEEE 1708 stays,
  `UNVERIFIED`, until its text is sourced (roadmap decision 17).
- **Templates**: the extend-the-package templates (new dataset, new model,
  new trace) are written once the reworked contract is proven on the three
  reworked models — they document this contract, so they come after it.
- **Docs**: CLAUDE.md's and `docs/architecture.md`'s contract sections are
  rewritten when the implementation lands, not before — they describe what
  *is*.

## Out of scope

- `ev` (event camera) frame representation — pinned when the first event
  cache exists.
- Unit conversion (cmH2O/mmHg) — refused, not converted, until a real cache
  mixes units.
- wandb integration — `batch["raw_losses"]`/`batch["losses"]` are shaped
  for it; nothing is built.
