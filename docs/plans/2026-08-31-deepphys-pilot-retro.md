# DeepPhys Pilot — Config Retro

Written per the migration contract §7 step 8. What was awkward to express,
duplicated, or forced by the yacs tree while landing stage 0 and DeepPhys.
Phase 5 designs the schema from this, not from speculation.

## What the pilot decided (precedent for every later migration)

| Concern | Key | Shape |
| --- | --- | --- |
| Window | `PREPROCESS.WINDOW_SECONDS` / `STRIDE_SECONDS` | seconds; `T = WINDOW_SECONDS × FS`, snapped within 0.01 frames, refused otherwise. Stride `0.0` = no overlap |
| Target rate | `DATA.FS` | mandatory, > 0. Deliberately *not* a new `FPS` key |
| Label norm | `PREPROCESS.LABEL_NORM` | per signal, `{SIG: raw\|zscore\|minmax}`; omit for the signal's class default |
| Loss | `TRAIN.LOSS` | per signal, `{SIG: {TYPE, WEIGHTS}}`; omit for the class default |
| Head style | `MODEL.HEAD_STYLE` | `widened` (A) / `per_signal` (B) |

Defaults for both per-signal keys come from the *signal class* now recorded in
`neural_methods/signals.py` (`absolute` for ABP/CVP/SPO2, `shape` for
PPG/ECG/RESP/EDA), so a normal config states neither. That is the pattern
worth keeping: **the config states experiment choices, the registry states
facts about signals.**

## Friction worth fixing in Phase 5

1. **The four data blocks are still copy-paste.** `TRAIN.DATA`, `VALID.DATA`,
   `TEST.DATA` and `UNSUPERVISED.DATA` each restate `FS`, `CACHED_PATH`,
   `CHANNELS`, `TRACES`, `WINDOW_SECONDS`, `RESIZE`, `DATA_TYPE`, `FILTERS`,
   `ALLOW_MISSING`, `MIN_CHANNELS`, `MIN_LABELS` — ~25 duplicated lines per
   block in every config. `main.py` already *refuses* to run when the resize,
   channels, traces or filters differ between train and test, which is an
   admission that they are one fact stated four times. The only keys that
   genuinely differ per split are `STRIDE_SECONDS` (overlap while training,
   none while evaluating) and `RANDOM_CHUNK`. **Phase 5: one `DATA` block plus
   per-split overrides.**

2. **`CHANNELS` and `TRACES` are still stated twice per block** — once at
   `PREPROCESS` and once at `PREPROCESS.NECKFLIX`, with the loader falling
   back from the first to the second. The fallback has outlived its purpose;
   the `NECKFLIX` copies in `configs/neckflix/*.yaml` are dead weight kept
   only because `resolve_channels`/`resolve_traces` still look for them.

3. **`TRAIN.LOSS` and `PREPROCESS.TRACES` are coupled but live in different
   blocks.** Naming a signal in `LOSS` that is not in `TRACES` is (correctly)
   an error, so narrowing `TRACES` silently breaks a config until you also
   narrow `LOSS`. This bit the trainer tests immediately. A schema where the
   per-signal spec hangs off the trace list itself would make the coupling
   structural instead of a cross-check.

4. **`LABEL_NORM` was under `NECKFLIX` but is not Neckflix-specific.** Moved
   up to `PREPROCESS` in this change. Several other `NECKFLIX.*` keys have the
   same problem: `ALLOW_MISSING`, `MIN_CHANNELS`, `MIN_LABELS` and
   `RANDOM_CHUNK` are all generic zarr-loader concerns. Only `FILTERS` and
   `PARTICIPANTS` are genuinely dataset-shaped.

5. **yacs types are load-bearing in a hostile way.** `STRIDE_SECONDS: 0` fails
   to merge because the default is `0.0` and yacs refuses an `int` for a
   `float`. Every config in the repo has to write `0.0`. A typed schema
   should coerce, or at least say which key and what to write.

6. **Dead keys still have to be written.** `LABEL_TYPE`, `DO_CHUNK`,
   `DATA_PATH`, `DATA_FORMAT`, `DO_PREPROCESS` and `CHUNK_LENGTH` are inert
   for the zarr pipeline but still appear in every config because they are
   defaulted in `config.py`. `CHUNK_LENGTH` is now actively misleading: it
   looks like it sets the window and does nothing.

7. **Cross-block consistency is enforced by a hand-maintained list.**
   `build_model` validates the window constraint against the *one* data block
   it was handed, so a `TEST.DATA` that disagrees with `TRAIN.DATA` reaches the
   model unchecked. `main.py` now guards `WINDOW_SECONDS`, `FS` and
   `LABEL_NORM` alongside the resize/channels/traces/filters it already
   guarded — but that list has to be extended by hand every time a key becomes
   model-identity-bearing, and it exists only because the four blocks are
   separate in the first place (item 1). Collapsing them deletes the whole
   check. Surfaced by the PhysFormer migration.

8. **`MODEL.HEAD_STYLE` is global but is a per-model capability.** PhysFormer
   supports style A only and has to raise at build time. A typed schema could
   let a model declare which styles it offers.

## DeepPhys-specific notes

- **`in_channels` is ambiguous across model families and cost a real bug.**
  DeepPhys splits the stacked `DATA_TYPE` blocks into its motion and
  appearance branches itself, so each of its convs is built for *one* block's
  width (`C`), while PhysMamba consumes the stack whole (`C × blocks`).
  Passing the stacked width to DeepPhys gives the appearance branch zero
  channels — which `conv2d` catches, but only at the first forward pass.
  `ModelSpec` now names both (`camera_channels`, `in_channels`) and the
  DeepPhys builder refuses anything but exactly two `DATA_TYPE` entries.
  TS-CAN and EfficientPhys migrations should read this first: TS-CAN splits
  the same way, EfficientPhys takes `C` raw channels and diffs internally.
- **`WINDOW_SECONDS: 6.0`** reproduces the upstream `CHUNK_LENGTH: 180` at
  `FS: 30`. DeepPhys is per-frame, so it constrains the window not at all —
  the conversion matters only for comparability with the published results.
- **`RESIZE: 72`** is upstream's. DeepPhys sizes its dense layer from the
  frame size, so `RESIZE.H/W` cannot be left at 0 ("keep the cache's size")
  the way a fully-convolutional model would allow.
- The old lookup table restricting `img_size` to 36/72/96 is gone; the dense
  width is computed from the two valid convolutions and two pools.

## Open question for the user

`TRAIN.LOSS` weights (`MEAN`/`MAX`/`MIN` at `1/scale`, `CCC` at 1.0) are a
first guess, not a tuned result. They put the terms in the same order of
magnitude, which is all they are meant to do. Whether CCC should dominate the
L1 terms as heavily as it currently does is an empirical question for the
first real LOSO sweep, and the per-component curves (`*_loss_components.pdf`)
are what should answer it.

## §7a re-verification on the DATA / INTERFACE / MODEL schema (2026-08-31)

Run against the mechanically converted configs per migration contract §7a.
Verification only: `DeepPhys.py`, `_build_deepphys` and
`tests/test_deepphys_multisignal.py` needed no change and got none.

**Config.** Every value survives the conversion. `INTERFACE`: `FS 30.0`,
`WINDOW_SECONDS 6.0` (180 frames — upstream's `CHUNK_LENGTH`), `CHANNELS
[R,G,B]`, `TRACES [ABP,CVP,ECG]`, `RESIZE 72x72`, `DATA_TYPE [DiffNormalized,
Standardized]`. `MODEL`: `DeepPhys` / `widened`. `TRAIN`: `LR 9e-3`,
`BATCH_SIZE 4`, `EPOCHS 30` — all three upstream's
(`PURE_PURE_UBFC-rPPG_DEEPPHYS_BASIC.yaml`) — plus the three-signal LOSS
registry unchanged (ABP absolute at 0.05, CVP absolute at 0.2, ECG shape).
`DATA.SPLITS.TRAIN.STRIDE_SECONDS 3.0` is the 50% overlap. `_SMOKE` is
`BASE: [NECKFLIX_DEEPPHYS.yaml]` plus overrides of existing keys only.

**Built model.** `build_model(load_config(...))` gives
`SignalDictWrapper(input_mode='frames2d')` over `DeepPhys`, with the pilot's
24 `state_dict` entries key-for-key and shape-for-shape. The checks that
matter:

- `motion_conv1` and `apperance_conv1` both take **3** input channels
  (`spec.camera_channels`), not the stacked 6 — the ambiguity that cost a
  real bug, re-confirmed on the new schema.
- `final_dense_1` is `(128, 16384)`, sized from 72x72 through the two valid
  convs and two pools ((72-2)//2 = 35, (35-2)//2 = 16, 64·16·16).
- `output_layers()` is the single `final_dense_2`, a bare `nn.Linear` to 3
  outputs, bias primed to the per-trace priors `[90.0, 8.0, 0.0]`.
- At `in_channels=3, out_signals=1` only the readout width differs from the
  published network — the fidelity principle holds.
- Non-square `RESIZE`, `RESIZE: 0`, and any `DATA_TYPE` count but two are all
  refused at build with the error naming the fix.

**Runs.** `pytest tests/ -q`: 280 passed. Smoke run
(`--limit_windows 8 --test_participants P015`, real cache): all three signals
scored in physical units, all four plot families and the outputs pickle
written. Checkpoint authority: a `MODE: only_test` config with a deliberately
wrong interface (`CHANNELS [R,G,B,I,D]`, `RESIZE 64x64`) printed the diff,
adopted the checkpoint's RGB/36x36, and reproduced the training run's test
metrics.

**What the redesign fixed.** Friction items 1, 2, 4, 5, 6 and 7 above are gone
outright. Item 3 (LOSS↔TRACES) is deliberately still a cross-check rather than
structural, and item 8 (`HEAD_STYLE` as a per-model capability) is still open.

**What the new schema still cannot express** — for Phase 5:

1. **`DATA_TYPE` order is load-bearing but nothing says so.** DeepPhys slices
   the first block into its motion branch and the second into its appearance
   branch, so `['Standardized', 'DiffNormalized']` silently trains the motion
   branch on appearance frames. The builder counts the blocks but cannot check
   their roles: `INTERFACE.DATA_TYPE` is an ordered list whose ordering carries
   per-model meaning the config never states. TS-CAN will inherit this exactly.
   A model-side declaration (`expects=('motion', 'appearance')`) would let the
   builder refuse it; a list cannot.
2. **The serialized interface records the omission, not the resolution.**
   `LABEL_NORM: {}` is the normal spelling, so the checkpoint carries `{}` and
   its label units are only recovered by re-reading `signals.py` at load time.
   That is fine while the class defaults are stable and silently reinterprets
   every existing checkpoint the day one changes. Resolving `LABEL_NORM` before
   `interface_payload` would make the checkpoint self-describing at no cost to
   the config's brevity.
3. **A model's frame-shape demands are build-time errors, not declarations.**
   DeepPhys needs square, non-zero frames; the config can state neither, so
   both are discovered as a `ValueError` from `_build_deepphys`. The errors are
   clear, but this is item 8's shape again: the interface has no vocabulary for
   what a given architecture can accept.
4. **Batch size is still stated twice.** `TRAIN.BATCH_SIZE` and
   `TEST.BATCH_SIZE` are separate keys, so a `_SMOKE` variant overrides both,
   alongside `LOG.PATH` and `TRAIN.MODEL_FILE_NAME` (needed so smoke artefacts
   do not land in the real run's directory). Minor next to what item 1 removed.
