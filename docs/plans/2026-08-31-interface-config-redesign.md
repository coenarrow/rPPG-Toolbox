# Config Redesign: the DATA / INTERFACE / MODEL split

The Phase 5 config consolidation, pulled forward and reshaped around one idea
from the DeepPhys/PhysFormer pilots: **the config should state the model's
demand on the data pipeline explicitly, and the pipeline's job is to satisfy
it**. Both pilot retros voted for pieces of this (collapse the four data
blocks; delete the hand-maintained cross-block consistency list; coerce
int→float; expect per-model architecture blocks of arbitrary size); this
design lands all of it at once because the pieces only make sense together.

## The idea

The old schema conflated two different facts: what the data provides and what
the model consumes. `ModelSpec` was derived *from a data block* at build time,
so "a pretrained 5-channel model tested on 3-channel data" was inexpressible —
the data config would simply have built a 3-channel model. The new schema
splits the config into three blocks with distinct ownership:

| Block | Owns | Direction |
| --- | --- | --- |
| `DATA` | which stores participate: cache path, attribute filters, participants, admission thresholds, per-split sampling policy | facts about the data |
| `INTERFACE` | what the model consumes and emits: rate, window, channels, traces, resize, `DATA_TYPE`, per-signal label norm | the model's **demand** |
| `MODEL` | the architecture: name, head style, per-model hyperparameter blocks | facts about the model |

The `INTERFACE` block is not a restatement of the data — it is a demand, and
the loader reconciles it against what each store actually has: a demanded
channel the dataset can never provide is delivered as zeros with
`channel_mask=False`; a demanded trace no recording carries is masked out; a
store faster than `FS` is decimated; a slower one is refused unless
interpolation is asked for by name. The interface travels with the checkpoint,
and at `only_test` the checkpoint's copy is the authority (contract §1: "the
checkpoint is the authority on what it expects").

## The new schema

```yaml
BASE: []                      # optional include paths, relative to this file,
                              # deep-merged in order before this file's keys
MODE: train_and_test          # train_and_test | only_test | unsupervised_method
DEVICE: cuda:0
DEBUG: false
LOG:
  PATH: runs/neckflix_physmamba

DATA:
  DATASET: Neckflix
  CACHED_PATH: "C:/Users/20759193/neckflix_cache/rgb128"
  FILTERS: {posture: ['0', '45', '90']}   # store root attrs, verbatim
  PARTICIPANTS: []            # include list; LOSO uses --test_participants
  ALLOW_MISSING: true
  MIN_CHANNELS: 1
  MIN_LABELS: 1
  SPLITS:                     # ONLY what genuinely differs per split
    TRAIN: {STRIDE_SECONDS: 2.133333, RANDOM_WINDOWS: false}
    VALID: {}                 # defaults: stride = whole window, deterministic
    TEST: {}

INTERFACE:                    # the model's demand on the data pipeline
  FS: 30.0                    # model-facing rate; stores are resampled to it
  WINDOW_SECONDS: 4.266667    # T = WINDOW_SECONDS x FS, snapped within 0.01
  CHANNELS: [R, G, B]         # ordered; order transfers to model.channels
  TRACES: [ABP, CVP, ECG]     # ordered; order transfers to model.traces
  RESIZE: {H: 128, W: 128}    # 0 = keep the cache's own size
  DATA_TYPE: [DiffNormalized]
  LABEL_NORM: {}              # {SIG: raw|zscore|minmax}; omit for class default
  UPSAMPLING: refuse          # refuse | interpolate (see below)

MODEL:
  NAME: PhysMamba
  HEAD_STYLE: widened
  DROP_RATE: 0.2
  PHYSFORMER: {PATCH_SIZE: 4, DIM: 96, ...}   # per-model blocks, any size

TRAIN:
  EPOCHS: 30
  BATCH_SIZE: 4
  LR: 1.0e-3
  MODEL_FILE_NAME: neckflix_physmamba_rgb
  USE_AMP: true
  AMP_DTYPE: bfloat16
  PLOT_LOSSES_AND_LR: true
  LOSS:                       # per-signal registry, validated against TRACES
    ABP: {TYPE: absolute, WEIGHTS: {CCC: 1.0, MEAN: 0.05, MAX: 0.05, MIN: 0.05}}

TEST:                         # how predictions are scored, in any mode
  BATCH_SIZE: 4
  METRICS: [MAE, RMSE, MAPE, MACC, Pearson, SNR, BA]
  USE_LAST_EPOCH: true
  EVALUATION_METHOD: FFT
  EVALUATION_WINDOW: {USE_SMALLER_WINDOW: false, WINDOW_SIZE: 10}
  MODEL_PATH: ''              # only_test: the checkpoint to load

UNSUPERVISED:
  METHODS: [POS, CHROM, ICA, GREEN, LGI, PBV, OMIT]
```

Derived at runtime, never written in YAML: `LOG.EXP_NAME`,
`TEST.OUTPUT_SAVE_DIR`, `UNSUPERVISED.OUTPUT_SAVE_DIR`, `MODEL.MODEL_DIR`.

## What each retro item became

1. **Four data blocks → one** (`DATA` + `SPLITS`). A split may set
   `STRIDE_SECONDS`, `RANDOM_WINDOWS`, and override `FILTERS`/`PARTICIPANTS`;
   nothing else. The unsupervised mode uses the `TEST` split policy.
2. **The `NECKFLIX.CHANNELS`/`TRACES` fallback dies.** `INTERFACE.CHANNELS`
   and `INTERFACE.TRACES` are the only spelling; `resolve_channels` /
   `resolve_traces` are deleted from `signals.py`.
3. **`LOSS`↔`TRACES` coupling**: still a cross-check (naming an unpredicted
   signal is an error), but now against the single `INTERFACE.TRACES` rather
   than a per-block copy. Hanging the loss spec off the trace list itself was
   considered and rejected: `LOSS` is a training choice, not model identity,
   and the interface block is what gets serialized into checkpoints.
4. **Generic loader keys move out of `NECKFLIX.*`**: `ALLOW_MISSING`,
   `MIN_CHANNELS`, `MIN_LABELS` live on `DATA`; `RANDOM_CHUNK` becomes the
   per-split `RANDOM_WINDOWS`. `FILTERS`/`PARTICIPANTS` (the genuinely
   dataset-shaped keys) also live on `DATA` — a second zarr dataset would
   reuse them unchanged.
5. **yacs dies.** `config.py` is a typed dataclass schema: int→float is
   coerced (`STRIDE_SECONDS: 0` works), `FS: 29.9796` is representable,
   unknown keys are refused with the full path and the block's valid keys.
6. **Dead keys die.** `LABEL_TYPE`, `DO_CHUNK`, `DATA_PATH`, `DATA_FORMAT`,
   `DO_PREPROCESS`, `CHUNK_LENGTH`, `BEGIN`/`END`, face detection, `INFO`,
   `FILTERING`, `FOLD`, `FILE_LIST_PATH`, `EXP_DATA_NAME` — none exist in the
   new schema. `INFERENCE.*` is absorbed into `TEST` (`BATCH_SIZE`,
   `MODEL_PATH`, `EVALUATION_METHOD`, `EVALUATION_WINDOW`);
   `UNSUPERVISED.METRICS` collapses into `TEST.METRICS`.
7. **The hand-maintained cross-block consistency list is deleted**, not
   extended: with one `DATA` and one `INTERFACE` there is nothing to drift.
8. **`HEAD_STYLE` stays global** (`MODEL.HEAD_STYLE`); a model that cannot
   offer a style keeps refusing it at build time with an explanation
   (PhysFormer). A per-model capability declaration remains future work.

## Demand-driven delivery (the dataloader's new capacity)

What the loader already did: zero-fill + `channel_mask`/`label_mask` for
streams/traces a *recording* lacks; decimation for stores faster than `FS`;
consumer-side resize + `DATA_TYPE`. What is new:

- **Channels the dataset can never provide.** A demanded channel missing from
  the dataset's `channel_map` is no longer a `ValueError`: it is delivered as
  zeros with `channel_mask=False`, with one construction-time warning. This is
  what lets an RGBID-pretrained checkpoint run on an RGB-only dataset — the
  same convention `stack_frames` already applies model-side.
- **Zero-coverage warnings.** After discovery/filtering the dataset warns for
  any demanded channel or trace that no admitted sample carries: at inference
  that is benign; in training it means the model is being taught to ignore
  that input (or will never receive gradient for that trace). A warning, not
  an error — deliberately: fine-tuning a wider pretrained model on narrower
  data is legitimate.
- **Temporal upsampling, by name only.** `UPSAMPLING: refuse` (default) keeps
  today's behaviour: a store slower than `FS` is refused, because naive frame
  duplication makes DiffNormalized identically zero. `UPSAMPLING: interpolate`
  linearly blends adjacent native frames (and label samples) at the target
  positions instead — interpolation, never duplication, and only when asked
  for by name. Rates within 1% remain "the same nominal rate" and take the
  identity path (the PhysFormer retro's jitter lesson stands).

Resize and `DATA_TYPE` deliberately stay consumer-side in
`frame_transforms.py`, carried by the model: the interface block
*parameterizes* them (via `ModelSpec`), it does not relocate them. Cheap
per-sample work (windowing, fill, masks, resampling) stays in loader workers;
per-batch tensor ops stay where they can run on GPU.

## The checkpoint carries its interface

`MultiSignalTrainer.save_model` writes
`{"state_dict": ..., "interface": <INTERFACE as a dict>, "model_name": ...}`.
At `MODE: only_test`, `main.py` reads the checkpoint's interface *before
building anything*, prints any difference from the config's `INTERFACE`
block, and adopts the checkpoint's copy — the loaders then deliver what the
checkpoint demands, not what the config guessed. A bare `state_dict` (the
pre-redesign format) still loads, with a warning that the config's interface
is being trusted blind.

`ModelSpec` is unchanged in shape (channels, traces, transform, fs, window,
resize, head_style, label_norms) — it is now constructed from `INTERFACE`
instead of derived from a data block, so **model builders did not change**.

## What was deliberately kept

- The config states *choices*; `signals.py` states *facts about signals*
  (class defaults for label norm and loss family). Both retros called this
  the thing to keep, and the interface block preserves it: `LABEL_NORM: {}`
  is the normal spelling.
- `FS` is a nominal rate reconciled by tolerance against measured store rates.
- `WINDOW_SECONDS` semantics (snap within 0.01 frames, refuse otherwise).
- The batch-dict contract, `PerSignalLoss`, and every model file: untouched.

## Consumers rewired

`main.py` (naming, split construction, checkpoint-interface adoption),
`MultiSignalTrainer` (spec from interface; checkpoint payload; `TEST.*` for
the old `INFERENCE.*` reads; absorbs `plot_losses_and_lrs`, dropping the
`BaseTrainer` inheritance), `unsupervised_predictor`, `BlandAltmanPy`,
`tools/list_neckflix_folds.py`, `dataset/data_loader/neckflix_config.py`
(translation layer: `zarr_config(config, split, ...)`), and all
`configs/neckflix/*.yaml` (smoke variants become `BASE:` + overrides).
`tests/test_legacy_contract.py` builds its own plain-namespace config so the
legacy tuple-path check no longer needs yacs; the legacy trainers themselves
are untouched dead code on their Phase 6 schedule.

## Consequences for the migrated models

DeepPhys, PhysFormer and PhysMamba need **re-verification, not re-migration**:
their builders read `ModelSpec`, which is unchanged. Per model, a sub-agent
verifies the converted config reproduces the same built model, runs the test
suite and a real smoke run, and records anything the new schema cannot
express. The migration contract carries the exact recipe (§7a).

## Addendum: the Phase 5 close-out slim (2026-08-31, later)

A second pass, after the §7a re-verifications, cut the schema to exactly what
a YAML may write:

- `LOG.PATH` → top-level `LOG_PATH`; `UNSUPERVISED.METHODS` → top-level
  `UNSUPERVISED_METHODS` (both single-value blocks dissolved).
- `TEST.EVALUATION_WINDOW.{USE_SMALLER_WINDOW, WINDOW_SIZE}` → one key,
  `TEST.EVALUATION_WINDOW_SECONDS` (`0` = score each window whole).
- `TRAIN.PLOT_LOSSES_AND_LR` deleted — the standard plot set is always
  written (rank 0 only), per the plots-written-once rule.
- The four runtime-derived fields (`LOG.EXP_NAME`, `MODEL.MODEL_DIR`, the two
  `OUTPUT_SAVE_DIR`s) left the schema entirely: `main.py` attaches
  `config.RUN` (`RunPaths`: `exp_name`, `model_dir`, `output_dir` — one
  output dir, since the modes are exclusive).
- `RESIZE` gained the square scalar shorthand (`RESIZE: 128`), normalised to
  `{H, W}` before the schema sees it.
- The YAML loader resolves floats with YAML 1.2 semantics (`9e-3` is a
  number), which also fixes numbers inside the free-form `WEIGHTS` and
  `MODEL.<NAME>` blocks; `_coerce_scalar`'s string-number workarounds died.
- `LABEL_NORM` is resolved to the full per-signal map at load and serialized
  resolved — the §7a finding that a checkpoint carrying `{}` would silently
  reinterpret its units if a class default ever changed.

The other close-out action: the three legacy config directories
(`configs/train_configs/`, `configs/infer_configs/`, `physhydra_configs/`)
were distilled into the migration contract's per-model settings table and
deleted — the contract, not 132 yacs-era files, is now the `T_orig`
reference for the remaining migrations.

---

Last updated: 2026-08-31
