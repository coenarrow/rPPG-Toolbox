# Model Migration Contract

How any upstream rPPG-Toolbox architecture is brought onto the multi-signal
batch-dict pipeline. This is the instruction document for migrations — one
agent, one model, this contract — covering every model in the repo **except
PhysHydra** (designed natively around Neckflix; it follows its own path).
PhysMamba and `SignalDictWrapper` are the reference implementations; the
DeepPhys pilot, once landed, is the worked example of everything below.

**Status (2026-08-31).** The DeepPhys pilot has landed, so **§7 stage 0 is
built** and this document is now describing the tree rather than anticipating
it: physical-time windowing with a resampling loader, per-signal label
normalisation with `raw` physical units, the per-signal composite loss
(`neural_methods/loss/PerSignalLoss.py`), checkpoint-authority channel
zero-fill, and plots 1/3/4. `ModelSpec` in `MultiSignalTrainer` carries
everything §1 says is derived, and the builder applies the §2 guardrails
(output-bias priors, weight-decay exemption, window-constraint check).
Migrated: PhysMamba (the reference `DictModel`), DeepPhys (the worked
example — see `configs/neckflix/NECKFLIX_DEEPPHYS.yaml` and
[its retro](2026-08-31-deepphys-pilot-retro.md)), PhysFormer. Remaining
models are one agent each, against this contract.

**Config schema update (2026-08-31, after both pilots).** The config was
redesigned into the **DATA / INTERFACE / MODEL split**
([design](2026-08-31-interface-config-redesign.md)): `INTERFACE` states the
model's demand on the data pipeline and is serialized into every checkpoint;
the yacs tree, the four per-split data blocks and the `INFERENCE` block are
gone. `ModelSpec` is unchanged in shape — it is now built *from* `INTERFACE`
— so **model code and builders did not change**. Section §1 below describes
the current schema; §7a is the re-verification recipe for the already-migrated
models (their configs were mechanically converted and are drafts until a
sub-agent verifies them).

Two things the pilot learned that are not obvious from the text below.
**`in_channels` is ambiguous**: a model that splits the stacked `DATA_TYPE`
blocks itself (DeepPhys, TS-CAN) builds its first layer for *one* block's
width, `spec.camera_channels`, while a model that consumes the stack whole
(PhysMamba) uses `spec.in_channels`. **A cache's nominal rate is not its
exact rate**: `FS: 30` is reconciled against measured per-store rates of
29.9796–30.0 by a relative tolerance, so the `T_orig / FS` conversions below
are nominal identities.

**The fidelity principle.** A migrated model stays as close to the published
architecture as possible. Exactly three things may change:

1. The **first layer** widens to accept `in_channels` inputs.
2. The **final readout** widens to emit `out_signals` outputs (and must be
   activation-free — a bare linear/conv — so absolute-range signals are
   expressible).
3. The **losses** applied to its output (loss functions are trainer-side
   machinery, not architecture — swapping them is a minor deviation and the
   model remains "the" model).

Everything between first layer and final readout is byte-for-byte the
original. With `in_channels=3, out_signals=1` the original network must be
exactly recoverable.

## 1. Where configuration comes from

Four sources, with a strict authority order. Nothing is ever stated in two
places.

| Source | Owns | Examples |
| --- | --- | --- |
| **Zarr cache** (store attrs) | Facts about the data | native `fps` (per-stream `video` attrs), which streams/traces a recording has, physical units |
| **`DATA` block** | Which stores participate | `CACHED_PATH`, attribute `FILTERS`, `PARTICIPANTS`, `ALLOW_MISSING`/`MIN_*`, per-split `SPLITS` policy (`STRIDE_SECONDS`, `RANDOM_WINDOWS`) |
| **`INTERFACE` block** | The model's **demand** on the data pipeline | `FS`, `WINDOW_SECONDS`, `CHANNELS`, `TRACES`, `RESIZE`, `DATA_TYPE`, per-signal `LABEL_NORM`, `UPSAMPLING` |
| **`MODEL` block / checkpoint** | The architecture and its identity | `NAME`, `HEAD_STYLE`, per-model blocks (`MODEL.PHYSFORMER.*`); every checkpoint carries its `INTERFACE` and is the authority at inference |

The interface is a *demand*, not a description: the loader's job is to
satisfy it — zero-fill + mask what the data cannot provide, resample time,
resize space (consumer-side) — or refuse with an error naming the fix.
Training choices (`TRAIN.LOSS`, LR, AMP) and scoring choices (`TEST.*`,
absorbing the old `INFERENCE` block) live in their own blocks and are not
model identity.

Rules:

- **The temporal contract is physical: `INTERFACE.WINDOW_SECONDS` and
  `INTERFACE.FS`, both mandatory** (stride as the per-split
  `DATA.SPLITS.<X>.STRIDE_SECONDS`, 0 = no overlap). The frame count is
  always derived: `T = WINDOW_SECONDS × FS`, **snapped to the integer
  within a tight tolerance (0.01 frames) and refused otherwise** — the
  error names the nearest valid `WINDOW_SECONDS`. The tolerance exists only
  to absorb decimal representation (`4.266667 × 30 → exactly 128`); a
  genuinely ambiguous value (`4.27 × 30 = 128.1`) is a config error, never
  a silent round. Rationale: 150 frames from a 30 fps camera is 5 s of
  physiology; 150 frames from a 150 fps camera is 1 s — not enough to carry
  a heart rate, and a frame count without a rate cannot tell them apart.
- **Matching an original architecture's canonical input length.** Where the
  published model has a canonical frame count `T_orig`, the config that
  reproduces it is `WINDOW_SECONDS = T_orig / FS` — at 30 fps: 128 frames
  → `4.266667` (64/15 s, PhysMamba-style), 160 frames → `5.333333`
  (16/3 s), 180 frames → `6.0`. The migration agent looks up `T_orig` in
  the **legacy settings reference** (appendix) — the legacy config trees
  were deleted in the Phase 5 close-out; git history has the files — and
  records the conversion in a comment beside `WINDOW_SECONDS` in the new
  config.
- **The store's native `fps` attr is the source rate for resampling, not the
  model-facing rate.** Where native > target, the loader decimates by
  nearest-index sampling over a `WINDOW_SECONDS × native_fps` span; labels
  are index-aligned to frames and resample with the same plan. Because
  `DATA_TYPE` transforms run consumer-side on the emitted window, diffs are
  taken between *sampled* frames (Δt = 1/target fps) — DiffNormalized
  semantics survive decimation by construction. **Native < target is refused
  by default**: naive upsampling means duplicated frames, and duplicated
  frames produce zero diffs — a dark motion branch, silently.
  `INTERFACE.UPSAMPLING: interpolate` opts into linear frame/label blending
  instead — interpolation, never duplication, and only by name. Rates within
  1% are the same *nominal* rate and take the identity path (the PhysFormer
  retro's jitter lesson). The model-facing rate (`INTERFACE.FS`) flows into
  the spectral loss and HR post-processing — never hardcoded.
- **The MODEL config block carries `NAME`, `HEAD_STYLE`, and a per-model
  architecture block of whatever size the architecture needs** (PhysFormer's
  six keys are normal, not a smell — retro item 1). Anything derivable from
  the interface is derived in the builder, never restated:

  | Derived quantity | From |
  | --- | --- |
  | `in_channels` | `len(CHANNELS) × frame_transform.channel_multiplier` |
  | `out_signals` | `len(TRACES)` |
  | `img_size` | `INTERFACE.RESIZE.H/W` |
  | window `T` | `WINDOW_SECONDS × FS` (integer within 0.01, validated) |
  | `fs` | `INTERFACE.FS` (stores are resampled to it) |

- **The checkpoint is the authority on what it expects — mechanically.**
  Every checkpoint `MultiSignalTrainer` writes is
  `{"state_dict", "interface", "model_name"}`; at `MODE: only_test`,
  `main.py` reads the checkpoint's interface *before building anything*,
  prints any difference from the config's block, and adopts it — the loaders
  then deliver what the model actually demands. Frame assembly aligns to
  `model.channels`, zero-filling any channel the model expects but the batch
  lacks (see §4), and the dataset does the same for channels the *dataset*
  can never provide. A checkpoint trained on RGBID runs on RGB-only data; it
  degrades, it does not crash. The same principle holds in time: the adopted
  interface's `FS` is the rate the dynamics were learned at, and the data is
  resampled to it — a 30 fps checkpoint fed a 150 fps source sees 30 fps.
  (A bare pre-redesign `state_dict` still loads, with a warning that the
  config's interface is being trusted blind.)

## 2. Predicting multiple waveforms

Every model implements `forward_video(video) -> (B, S, T)` behind
`DictModel` — `(B, C_in, T, H, W)` in, one row per trace out. Per-frame 2-D
backbones (DeepPhys, TS-CAN, EfficientPhys, BigSmall) are wrapped with
`SignalDictWrapper(input_mode='frames2d')`, which folds T into the batch axis
and back; video-native models use `input_mode='video3d'` or subclass
`DictModel` directly (PhysMamba). Channel and signal **order is owned by
`model.channels` / `model.traces`** — dict iteration order is never
load-bearing.

Multi-signal prediction is **shared trunk + per-signal readout**, in one of
two sanctioned head styles (a builder option; both keep the trunk identical):

- **Style A — widened readout (default, most faithful):** the original final
  layer's output width goes from 1 to S. A `Linear(nb_dense, S)` *is* S
  independent linear readouts of the shared feature; nothing else changes.
- **Style B — per-signal head copies (BigSmall precedent):** one copy of the
  original dense head (e.g. `fc1+fc2`) per signal, all reading the shared
  trunk feature. BigSmall does exactly this natively for its three tasks.
  Justified where signals draw on different spatial regions (ABP from the
  carotid, CVP from the jugular): because the head reads the flattened
  spatial map, a per-signal head is a per-signal learned spatial weighting.
  Costs S× the head parameters; the conv/attention trunk stays untouched.

Start every migration with Style A. Escalate to Style B only on per-signal
metric evidence of interference (the trainer already scores per signal, so
interference is observable, not hypothetical).

Per family:

- **2-D per-frame** (DeepPhys, TS-CAN, EfficientPhys): near-mechanical; see
  the DeepPhys pilot.
- **3-D conv** (PhysNet, iBVPNet, FactorizePhys): the final output conv
  widens to S channels.
- **Transformers** (PhysFormer, RhythmFormer): head + tokenization decisions
  are made collaboratively at migration time (roadmap Phase 6), recorded in
  the decision log.
- **BigSmall**: covered by this contract with two declared extensions — it
  derives its big/small dual-resolution views *internally* from the one raw
  `forward_video` input (frame preprocessing is model-owned, so this is
  legal), and its AU categorical head is **out of scope** (the contract
  covers waveform traces only; no Neckflix signal is categorical).

**Temporal constraints are declared, never silently handled.** A model with a
divisibility requirement on T (TSM/WTSM `n_segment`) or an architecturally
fixed T (some transformers, PhysMamba's 128) declares it, and the builder
checks it against the derived window T at construction time — the config
hits it via `WINDOW_SECONDS = T_orig / FPS` (§1); the legacy trainers'
silent batch truncation does not migrate.

For absolute-class signals (§3), two optimisation guardrails are part of the
builder, not the architecture:

- **Per-signal output-bias initialisation** to a physiological prior (ABP row
  ≈ 90 mmHg, CVP ≈ 8 mmHg) so training does not begin with a ~90 mmHg
  systematic error.
- **The output head is exempt from weight decay** (decay drags raw-unit
  predictions toward zero — a systematic pressure bias, not regularisation).

## 3. Losses and absolute scale

Signals fall into two classes, and the class decides both the label
normalisation and the loss family. Both are **per-signal config**, not global
switches:

| Class | Signals | Label norm | Loss components | Prediction space |
| --- | --- | --- | --- | --- |
| **Absolute** | ABP, CVP (SpO2 later) | `raw` — physical units, untouched | CCC + L1 window-mean + L1 soft-peak max + L1 soft-peak min | mmHg, directly |
| **Shape** | PPG/BVP, ECG, RESP | per-window z-score (unchanged) | negpearson (optionally + spectral) | normalised, dimensionless |

Design decisions behind this (recorded here so no migration relitigates
them):

- **No global/dataset normalisation constants.** Absolute-class models
  predict physical units directly from an activation-free readout. The
  per-signal scale factors that any multi-signal objective needs live
  **explicitly in the loss weights** (raw ABP error is O(10² mmHg), CVP
  O(10¹), negpearson O(1) — unweighted sums let ABP own every gradient).
  CCC is the well-conditioned base term: dimensionless and O(1) in any
  units, yet it penalises mean and amplitude mismatch.
- **Stats are derived from the predicted waveform** — window mean, and
  systolic/diastolic via the differentiable soft-peak machinery from
  `PhysHydraLoss` (soft local-extrema map + temperature softmax ≈ mean
  systolic/diastolic across the window's beats). There is **no separate
  stats head**: predictions stay `{sig: (B, T)}`, and a waveform can never
  disagree with its own statistics.
- **Masking composes.** Every component reduces per-sample to `(B,)`; the
  `MaskedMultiSignalLoss` structure (masked mean over the batch with clamped
  denominator, then mean over traces) is retained, so an absent signal
  contributes exactly 0 — never NaN, never a sentinel.
- **The loss spec is a per-signal registry** — loss type plus component
  weights per trace. This is how BigSmall's heterogeneous per-task criteria
  (BCE + MSE + MSE) generalise; a future categorical signal would add a loss
  type, not a redesign.
- `fs`/`fmax` for spectral terms come from the resolved dataset rate (§1).
- Clinical tolerance penalties (the 5/8 mmHg AAMI/ISO bands drafted in
  `PhysHydraLoss`'s commented-out version) are **Phase 7**, not part of
  migration.

Honest expectations: inferring an unseen subject's absolute pressure level
from video under LOSO is the hard research question. The loss gives the model
the gradient to try; migrations are not judged on absolute-level accuracy.

## 4. Multiple channels (5 for Neckflix: R, G, B, I, D)

- **Only the first conv widens** (`in_channels`); all downstream filter
  widths are the original's. `DATA_TYPE` semantics are preserved: with
  `['DiffNormalized', 'Standardized']` the backbone receives 2×C channels
  and models that split diff/raw blocks (DeepPhys, TS-CAN) slice
  `[:C]` / `[C:2C]` exactly as upstream did with 3. EfficientPhys computes
  its diff internally and takes C raw channels.
- **Missing data is zeros + a False mask, end to end.** The dataset emits
  every demanded channel, zero-filled with `channel_mask=False` where the
  store lacks the stream — and likewise where the *dataset* has no such
  stream at all (a channel outside its `channel_map` warns once at
  construction and is delivered as zeros throughout) — and every demanded
  trace, zero-filled with `label_mask=False`
  (`dataset/data_loader/zarr_dataset.py`). The model-side counterpart is
  `stack_frames` zero-filling channels the model expects but the batch
  lacks, instead of raising. Both are contract, not per-model work. The
  dataset also warns for any demanded channel or trace that *no* admitted
  sample carries: benign at inference, but in training it means the model is
  being taught to ignore that input, or will never receive gradient for that
  trace.
- Masks make reduced input *correct*, not costless: a model trained with
  IR/depth present will lose accuracy fed zeros there. The same machinery
  enables channel-dropout augmentation (train with channels randomly
  masked) — permitted, not required, and not a migration concern.

## 5. The prediction contract

`out = model(batch)` returns **the same dict** with `predictions` added —
nothing is dropped in transit:

```python
out["predictions"]   # {signal: (B, T)}, ordered by model.traces
out["frames"] is batch["frames"]
```

- Absolute-class predictions are in **physical units** (mmHg); shape-class
  predictions are in the per-window normalised space. `label_stats` rides in
  every batch for inversion and physical-unit reporting (identity for `raw`
  signals).
- A model given a subset of its channels runs (zeros in the missing planes);
  a batch lacking one of the model's traces is simply not scored on it. No
  configuration of present/absent data is an error at forward time.
- Saved test outputs (`TEST.OUTPUT_SAVE_DIR`) are keyed per signal and carry
  metadata (`recording_id`, `camera_id`, `start_frame`), `fs`, and the
  per-signal label-norm mode, so downstream tooling
  (`tools/summarise_neckflix_outputs.py`, Phase 7 metrics) needs no model
  knowledge.

## 6. Plots

Implemented **once**, in `MultiSignalTrainer` / `evaluation/metrics_report.py`
— never per model. Cross-model consistency is the point. The standard set
(the first two exist; the rest land with the DeepPhys pilot):

1. **Training curves** — train/valid loss and LR (exists; currently inherited
   from `BaseTrainer`, absorbed into `MultiSignalTrainer` when `BaseTrainer`
   dies in Phase 6), extended with **per-signal** and **per-component**
   curves (CCC vs mean vs peak terms separately — which term dominates is
   the first thing debugging needs).
2. **HR Bland-Altman** scatter/difference plots for pulsatile signals
   (exists, in `metrics_report`).
3. **Per-signal waveform overlays** — prediction vs label for a few test
   windows per signal, physical units for absolute-class signals, window
   metadata in the title.
4. **Absolute-class agreement scatters** — predicted vs true window mean,
   systolic, and diastolic per window with the identity line. This is the
   migration-time precursor of the Phase 7 clinical agreement analysis
   (IEEE 1708 / ISO 81060 bands), which consumes the same saved outputs.

## 7. The migration recipe

**Stage 0 — shared infrastructure.** Built once, by the DeepPhys pilot;
every later migration assumes it. Where each piece lands:

- Physical-time windowing (§1): `dataset/data_loader/neckflix_config.py`
  (keys + tolerance-snap validation) and
  `dataset/data_loader/zarr_dataset.py` (decimating window sampler;
  upsampling refused).
- Per-signal `raw` label mode (§3):
  `dataset/data_loader/label_transforms.py` plus its `zarr_dataset.py`
  call site (per-signal norm mode replaces the single `label_norm`).
- Checkpoint-authority channel alignment (§4): `stack_frames` in
  `neural_methods/batch.py` zero-fills expected-but-absent channels
  instead of raising.
- Per-signal composite loss (§3): new module beside
  `neural_methods/loss/MaskedMultiSignalLoss.py`, reusing the CCC /
  soft-peak / spectral machinery from `neural_methods/loss/PhysHydraLoss.py`
  with components reduced per-sample. **The pilot defines the concrete
  per-signal config keys** (norm mode, loss type, component weights);
  later migrations follow its precedent, never invent their own.
- Plots 3–4 (§6): `MultiSignalTrainer` / `evaluation/metrics_report.py`.

Per model, in one change:

1. Read the original architecture, its legacy `<Model>Trainer.py` (the
   migration reference until step 7 deletes it), and its row in the
   **legacy settings reference** appendix (canonical `T_orig`, §1);
   identify its family (§2).
2. Parameterise `in_channels` / `out_signals` (first layer + final readout
   only); verify the final readout is activation-free. Any reshapes touched
   use einops (cross-cutting rule). Original recoverable at `3/1`.
3. Wrap (`SignalDictWrapper`, `frames2d`/`video3d`) or subclass `DictModel`.
4. Add the builder to `MODEL_REGISTRY` in
   `neural_methods/trainer/MultiSignalTrainer.py` — derives all widths per
   §1, applies bias init + weight-decay exemption for absolute-class
   signals, exposes the head-style option (Style A default). The existing
   `_build_physmamba` / `_build_deepphys` are the pattern.
5. Add `configs/neckflix/NECKFLIX_<MODEL>.yaml` in the DATA / INTERFACE /
   MODEL schema (`NECKFLIX_PHYSMAMBA.yaml` is the worked example; the
   `_SMOKE` variant is `BASE: [<real config>]` plus a handful of overrides).
   The MODEL block is `NAME`, `HEAD_STYLE` where relevant, and a per-model
   architecture block of whatever size the architecture needs.
6. **One smoke test** — the ceiling, per the testing rule
   (`tests/test_deepphys_multisignal.py` is the pattern). The contract
   tests already cover dict plumbing; do not re-test it per model.
7. **Delete the legacy `<Model>Trainer.py`** in the same change.
8. Write a short **config retro** (what was awkward, duplicated, forced by
   the yacs tree) — these feed the Phase 5 schema design.
9. Validate: full suite green, then a
   `--limit_windows 8 --test_participants P015` smoke run.

## 7a. Re-basing the already-migrated models (one sub-agent per model)

DeepPhys and PhysFormer were migrated before the config redesign. Their model
files, builders and smoke tests did not change (ModelSpec kept its shape), and
their configs were **mechanically converted** to the DATA / INTERFACE / MODEL
schema — those YAMLs are *drafts* until this recipe has been run against them.
PhysMamba's config was converted the same way and is verified by the trainer
test suite; a sub-agent taking DeepPhys or PhysFormer does, in one change:

1. **Verify the drafted config against the pilot's settings.** Diff the
   converted `NECKFLIX_<MODEL>{,_SMOKE}.yaml` against the retro and the
   pre-redesign values (this conversation's history and the pilot retros are
   the record): every INTERFACE value (`FS`, `WINDOW_SECONDS`, `CHANNELS`,
   `TRACES`, `RESIZE`, `DATA_TYPE`), the per-split strides, the MODEL block,
   the LOSS registry, LR/AMP/epochs. The `_SMOKE` variant must stay
   `BASE: [<real config>]` + overrides only.
2. **Verify the built model is unchanged.** `build_model(load_config(...))`
   must produce the same architecture the pilot landed: same `state_dict`
   keys and shapes as before the redesign, same frame transform, same window
   constraint behaviour. The fidelity re-check recipe in the model's retro
   applies if anything architectural looks off.
3. **Run the suite** (`uv run python -m pytest tests/ -q`) and the real smoke
   run (`--limit_windows 8 --test_participants P015` against the `_SMOKE`
   config, on the real cache).
4. **Exercise the checkpoint-interface path**: after the smoke run, point a
   `MODE: only_test` config with a *deliberately wrong* INTERFACE (e.g. wrong
   RESIZE) at the produced checkpoint and confirm main adopts the
   checkpoint's interface, prints the diff, and the run scores.
5. **Retro anything the new schema cannot express** — appended to the model's
   existing retro file, feeding Phase 5's remaining consolidation (legacy
   config deletion).

No trainer, loss, plot, or model-file work is in scope; if step 2 finds a
discrepancy, that is a finding to report, not something to quietly fix.

## Appendix: model roster

| Model | Family / input mode | Head notes | Status |
| --- | --- | --- | --- |
| PhysMamba | native `DictModel` | — | migrated (reference; config on the new schema, suite-verified) |
| DeepPhys | frames2d | diff/raw split; Style A (B available); attention shared | migrated (pilot — [retro](2026-08-31-deepphys-pilot-retro.md)); **§7a re-verified on the new schema (2026-08-31)** |
| TS-CAN | frames2d | diff/raw split; declare TSM `n_segment` \| T | pending |
| EfficientPhys | frames2d | raw-only input (internal diff) | pending |
| PhysNet | video3d | widen final conv to S | pending |
| iBVPNet | video3d | widen final conv to S | pending |
| FactorizePhys | video3d | widen final conv to S | pending |
| PhysFormer | native `DictModel` (transformer) | Style A; tokenization unchanged; style B N/A (grid pooled before the head) | migrated ([retro](2026-08-31-physformer-migration-retro.md)); **§7a re-verified on the new schema** (config, built model, 280-test suite, smoke run and checkpoint-interface adoption; no architectural discrepancy) |
| RhythmFormer | video3d (transformer) | head/tokenization decided at migration | pending (Phase 6) |
| BigSmall | frames2d, dual view derived internally | per-signal heads native (Style B); AU head out of scope; declare WTSM constraint | pending (Phase 6) |
| PhysHydra | — | **excluded from this contract** (Neckflix-native) | own path |

## Appendix: legacy settings reference

Distilled 2026-08-31 from `configs/train_configs/`, `configs/infer_configs/`
and `physhydra_configs/` (132 files; 141 counting the pre-overhaul hidden
`.configs/` deleted alongside), which were **deleted in the Phase 5
close-out** — this table replaces them as the canonical-settings record; git
history before that commit has the originals, including per-dataset trivia
this table deliberately drops (odd batch sizes, epoch counts and rates on
datasets this repo does not target). Values are the canonical
(most-common, cross-dataset) settings; per-dataset variants worth knowing are
in the notes. `WINDOW_SECONDS` is `T_orig / 30` (§1).

| Model | `T_orig` | `WINDOW_SECONDS` @30 | Resize | `DATA_TYPE` | Label norm (legacy `LABEL_TYPE`) | LR | Batch | Epochs | Architecture block |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TS-CAN | 180 | 6.0 | 72 | `[DiffNormalized, Standardized]` | DiffNormalized | 9e-3 | 4 | 30 | `TSCAN.FRAME_DEPTH: 10`; `DROP_RATE: 0.2` |
| EfficientPhys | 180 | 6.0 | 72 | `[Standardized]` | DiffNormalized | 9e-3 | 4 | 30 | `EFFICIENTPHYS.FRAME_DEPTH: 10`; `DROP_RATE: 0.2` |
| PhysNet | 128 | 4.266667 | 72 | `[DiffNormalized]` | DiffNormalized | 9e-3 | 4 | 30 | `PHYSNET.FRAME_NUM: 128` ("only support for factor of 512") ; `DROP_RATE: 0.2` |
| iBVPNet | 160 | 5.333333 | 72 | `[Raw]` | Standardized | 1e-3 | 4 | 30 | `iBVPNet.{CHANNELS: 3, FRAME_NUM: 160}`; `DROP_RATE: 0.1` (0.2 in two files) |
| FactorizePhys | 160 | 5.333333 | 72 | `[Raw]` | Standardized | 1e-3 | 4 | 10 | `FactorizePhys.{CHANNELS: 3, FRAME_NUM: 160, MD_TYPE: NMF, MD_R: 1, MD_S: 1, MD_STEPS: 4, MD_RESIDUAL: True, MD_INFERENCE: True}`; `MD_FSAM: True` at train, `False` in every infer config; `MD_TRANSFORM: "T_KAB"` (stated once, "default if not specified"); `DROP_RATE: 0.1` |
| RhythmFormer | 160 | 5.333333 | **128** | `[Standardized]` | Standardized | 9e-3 | 4 | 30 | **none** — no `MODEL.RHYTHMFORMER` block exists anywhere; everything is hard-coded in the class. `DROP_RATE: 0.2` |
| BigSmall | 3 | n/a — frame-depth chunking, not a physiological window | BIG 144, SMALL 9 | BIG `[Standardized]`, SMALL `[DiffNormalized]` | DiffNormalized + `USE_PSUEDO_PPG_LABEL: True` | 1e-3 | 180 | 5 | `BIGSMALL.FRAME_DEPTH: 3`; no `DROP_RATE`; FS **25** (BP4D+); the dual resolution lived in a `PREPROCESS.BIGSMALL` block, not `RESIZE` |
| PhysHydra | 128 | 4.266667 | 128 | `[DiffNormalized]` | Raw (TEST block only) | 3e-3 | 2 | 50 | none under `MODEL.*`; trace norm ranges (`ABP [0,200]`, `CVP [-20,30]`, `ECG [-1500,1500]`), postures and `RANDOM_CHUNK` lived in the legacy `NECKFLIX` block. Own migration path |

Variant notes that survive the deletion:

- **TS-CAN / EfficientPhys at other rates**: the UBFC-PHYS *infer* configs
  rescaled to `FS: 35` + `CHUNK_LENGTH: 210` (and PURE→iBVP to 160), but the
  UBFC-PHYS *train* configs kept 180 @ 35 — a real train/infer inconsistency
  in upstream; the physical-time schema makes it inexpressible (6.0 s is
  6.0 s at any rate).
- **EfficientPhys on iBVP** flipped all three axes: `[Raw]` input
  ("if use EfficientPhys, should be Raw"), Standardized labels,
  `CHUNK_LENGTH: 160`.
- **PhysNet never rescaled** its 128 frames regardless of dataset FS.
- **FactorizePhys `MD_STEPS`** was 3 (not the canonical 4) in the two
  PURE→iBVP configs.
- BP4D+ datasets ran at FS 25 (train/valid); PhysDrive TS-CAN used batch 32.

---

Last updated: 2026-08-31
