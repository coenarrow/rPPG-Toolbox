# The Zarr Cache Contract

**Normative.** Every zarr store this repository reads must satisfy this
document, and the only admission mechanism is `tools/validate_cache.py`,
which checks exactly the clauses tabulated under "What the validator checks".
[`revised_overhaul_plan.md`](../revised_overhaul_plan.md) is the statement of
intent it implements. There is one contract — this document — and it is not
versioned: a store either satisfies it or it does not. The reader
(`src/windows.py`) reads exactly this layout.

The cache is written by a dataset's cacher, which lives at
`dataset/cachers/<name>/` (for Neckflix, the submodule at
`dataset/cachers/neckflix`). Nothing on the training or evaluation path
writes it. Run the validator after
generating a cache: a store it passes is admissible, full stop. There is no
`complete` or `tool_version` attr.

```
uv run python tools/validate_cache.py <cache-dir | store.zarr ...>
```

## Layout

```text
{recording}.zarr
+-- attrs                        participant (required) + any filter attrs
+-- <perspective>/               "1", "2", ... one group per camera viewpoint
    +-- attrs                    fps (required: a positive number, or null)
    +-- <modality>/              one group per modality in the vocabulary
        +-- timestamps_us/data   (T,) microseconds, strictly increasing
        +-- video/data           (C, T, H, W), any dtype
        +-- <trace>/data         (T,) float, physical units
            attrs                units (required)
```

Every array lives at `<name>/data`, uniformly. A bare array as a direct
child of a modality group (a `video/frames` array, or a trace written as an
array) is a violation.

## Root

- `participant` — **required, and a string.** Any identifier a dataset
  chooses (`"015"`, `"subject-3"`, `"01-01"`); the contract imposes no
  format. It is opaque: the split machinery (LOSO, `PARTICIPANTS`,
  `--test_participants`) matches it exactly. Zero-padding, prefixes and the
  like are dataset conventions and belong in that dataset's cache spec
  (`dataset/data_loader/<DATASET>.md`), not here. A number is a violation.
- Any other attrs are allowed and none are required. They are the
  `DATA.FILTERS` surface (`posture`, `light`, `session`, ...). Their value
  vocabularies are per-dataset conventions, again documented in the cache
  spec.
- At least one perspective group.

## Perspective

A perspective is a set of modalities whose pixels are physically aligned
(Neckflix reprojects IR and depth into the RGB camera's frame). The loader
treats each perspective as an independent sample.

- `fps` — **required key.** Its value is one of:
  - a positive finite number: the nominal frame rate of **every** modality
    in the perspective. Per-modality exact rates may drift; the reader
    reconciles them against the model's `INTERFACE.FS`.
  - `null` (JSON) or NaN — the perspective has no frame rate. This is the
    event-camera case. A perspective without a rate cannot be resampled to
    a model's `FS` and is only usable by a model that consumes it natively.

  Anything else — the key absent, a string, zero, a negative number,
  infinity — is a violation.
- **Alignment.** When `fps` is a number, the first frame of every modality
  lies strictly within `1/fps` of every other's, judged by
  `timestamps_us`. When `fps` is null there is no budget to judge against
  and the check is skipped.
- **Identical trace sets.** Every modality in the perspective carries the
  same set of trace groups.
- Every child group is a modality from the vocabulary.

## Modality

Fixed vocabulary — a store may carry any subset, nothing outside it. The
table lives once in code (`MODALITY_CHANNELS` in
`neural_methods/signals.py`) and is the global channel map.

| key     | meaning          | C        | canonical channels |
| ------- | ---------------- | -------- | ------------------ |
| `gr`    | grayscale camera | 1        | `Y`                |
| `rgb`   | RGB video        | 3        | `R`, `G`, `B`      |
| `ir`    | infrared         | 1        | `I`                |
| `depth` | depth            | 1        | `D`                |
| `t`     | thermal          | 1        | `T`                |
| `ev`    | event camera     | unpinned | unpinned           |

Each modality group holds:

- `timestamps_us/data` — `(T,)`, microseconds, strictly increasing. Each
  sensor keeps its own clock; this is the record of it.
- `video/data` — `(C, T, H, W)`, stacked in exactly that order, `C` fixed
  by the vocabulary, `T > 0`. **Any dtype** — uint8 RGB, uint16 IR or
  depth, float — the contract does not care; the consumer casts
  (`neural_methods/frame_transforms.py`).
- One group per trace the recording carries, index-aligned to **this
  modality's** frames: trace length equals video `T`.

Across modalities in one perspective `T` may differ (a sensor that died
early). Each modality is internally consistent — `timestamps_us`, `video`
and every trace agree on `T` — and the reader truncates every modality and
its traces to the shortest at read time.

`ev` is unpinned: the validator accepts any `C` for it and prints a note, so
a PASS never silently means "not looked at". Its frame representation, and
what its `video` and trace groups are, is fixed when the first event cache
exists.

## Trace

Fixed vocabulary (`TRACE_KEYS` in `neural_methods/signals.py`):

| key   | meaning                 | repo signal |
| ----- | ----------------------- | ----------- |
| `ecg` | ECG                     | `ECG`       |
| `abp` | arterial blood pressure | `ABP`       |
| `cvp` | central venous pressure | `CVP`       |
| `ppg` | finger PPG              | `PPG`       |
| `rr`  | respiration (chest)     | `RESP`      |

- `data` — `(T,)`, a floating dtype, physical units, index-aligned to the
  modality's frames.
- `units` — **required attr**, a string (`"mmHg"`, `"cmH2O"`, `"mV"`,
  `"arb"`, ...). `"arb"` is the expected value for shape-class signals. Unit
  conversion is added only if a signal ever turns up in two units; see
  "Open items" for what the reader does with this attr today.

## What the validator checks

Every clause above, itemised per store: one `PASS` line, or a `FAIL` with
each violation; exit 1 if any store failed. Importable as
`validate_store(path) -> list[Violation]` so a preprocessor can call it at
write time.

| Level       | Checked |
| ----------- | ------- |
| store       | opens as a zarr group; at least one perspective group |
| root        | `participant` present and a string |
| perspective | `fps` present, and null/NaN or a positive finite number; child groups in the modality vocabulary; identical trace sets across modalities; first-frame spread under `1/fps` (numeric `fps` only) |
| modality    | no bare array children; `timestamps_us/data` and `video/data` present; video 4-D with `T > 0` and `C` per the vocabulary (`ev` exempt); timestamps shape `(T,)` and strictly increasing |
| trace       | key in the trace vocabulary; `data` present, shape `(T,)`, floating dtype; `units` attr present |

Not checked, by design: frame dtype, `H` and `W`, chunking or compression,
that timestamps are truly microseconds or consistent with `fps`, that trace
values are finite, and the content of any filter attr.

## Reader consequences

What `src/windows.py` does with a conformant store:

- Reads `video/data`, the perspective-level `fps` and `timestamps_us`; no
  admission check of its own.
- Enumerates a store's perspectives, modalities and traces from the groups
  it actually holds — never from an attr describing what it should hold.
- Reconciles unequal modality durations by truncating to the shortest.
- One dataset class: the modality table above replaces per-dataset
  `channel_map` subclasses. A new dataset is an external cache writer plus
  a markdown cache spec.
- Matches `participant` exactly as the store writes it — no normalisation
  in the dataset, the loader or the CLI.

## Open items

- `ev`: representation unpinned. The preprocessor's `ev` perspective
  currently carries `x`/`y`/`p` groups and no `video/data`, which fails
  today's modality clauses. Pinning it is the next contract change.
- `units` is required on every trace and the validator checks it, but the
  reader does not yet carry it into the batch. When it does, it lands beside
  `label_stats` as `label_units` (`{signal: str}`), and a signal whose units
  differ across the stores of one run is refused rather than silently mixed.

---

Last updated: 2026-09-07
