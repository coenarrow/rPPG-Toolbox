# Changelog

This file tracks development milestones. As of 2026-08-25 all branches (HPC, MacOS, phys-hydra) are unified into `main`, which has intentionally diverged from upstream rPPG-Toolbox (upstream fixes are cherry-picked only).

**Format:** Each entry records the date, the branch or development context, and a summary of changes. Entries are listed in reverse chronological order (most recent first).

---

## 2026-08-31 — Branch: `main` — Phase 5 close-out: schema slimmed, legacy config piles deleted

Going back to yacs was considered and declined: the parser was never what
made the config complex — the key count was. So the close-out cut keys, not
machinery ([design addendum](plans/2026-08-31-interface-config-redesign.md)).

- **Schema slimmed to exactly what a YAML may write.** `LOG.PATH` →
  top-level `LOG_PATH`; `UNSUPERVISED.METHODS` → `UNSUPERVISED_METHODS`;
  the two-key `TEST.EVALUATION_WINDOW` block → one
  `EVALUATION_WINDOW_SECONDS` (`0` = score each window whole); dead
  `TRAIN.PLOT_LOSSES_AND_LR` deleted (the standard plot set always writes,
  rank 0 only); the four runtime-derived fields left the schema for
  `config.RUN` (`RunPaths`: exp_name, model_dir, one output_dir — the modes
  are exclusive); `RESIZE: 128` square shorthand; the loader resolves floats
  with YAML 1.2 semantics, so `9e-3` is a number even inside the free-form
  `WEIGHTS` and `MODEL.<NAME>` blocks.
- **`LABEL_NORM` is resolved at load and serialized resolved** — checkpoints
  now carry `{ABP: raw, CVP: raw, ECG: zscore}` rather than `{}`, closing
  the §7a finding that a class-default change would silently reinterpret an
  existing checkpoint's units.
- **The legacy config piles are gone**: `configs/train_configs/`,
  `configs/infer_configs/`, `physhydra_configs/` (132 files) plus the
  pre-overhaul hidden `.configs/` (9), distilled first into the migration
  contract's **legacy settings reference** appendix — canonical `T_orig`,
  resize, data types, LR/batch/epochs, every model-specific block, and the
  per-dataset variants worth keeping (including upstream's real UBFC-PHYS
  train/infer FS inconsistency, which the physical-time schema makes
  inexpressible). `configs/` holds only `neckflix/`.
- Suite: 289. Real-cache PhysMamba smoke green end-to-end on the slimmed
  schema, metrics identical to the pre-slim run.

## 2026-08-31 — Branch: `main` — §7a re-verifications green; PURE on the zarr cache

- **DeepPhys and PhysFormer §7a re-verification (one sub-agent per model) —
  both passed, no architectural discrepancy.** The drafted configs carry the
  pilots' settings; the built models are key-for-key the pilots'
  `state_dict`s; suite and real-cache smokes green; the checkpoint-interface
  adoption path exercised with deliberately wrong interfaces. Findings
  recorded in the retros: PhysFormer's parameter shapes are grid-invariant,
  so a plausible-but-wrong `RESIZE` is caught *only* by interface adoption;
  and four things the schema still cannot express (`DATA_TYPE` order is
  load-bearing but unstated; the serialized interface records `LABEL_NORM`'s
  omission, not its resolution; frame-shape demands are build errors, not
  declarations; batch size lives in two keys).
- **PURE is on the zarr cache.** `tools/cache_pure.py` writes
  `{recording}.zarr` from the raw PNG sequences + JSON sidecar (timestamp
  alignment by default, the legacy index grid behind `--align index`, the
  choice recorded in the store's `alignment` attr); `PUREDataset` is the
  two-property `BaseZarrDataset` subclass the contract promises; `PURE.md`
  upgraded from proposed to implemented mapping. The local four-recording
  subset is cached at `D:/pure_zarr`. Suite: 288. Not yet reachable from
  `main.py`, which still hardcodes `NeckflixDataset` — dataset selection via
  `DATA.DATASET` is the open wiring.

## 2026-08-31 — Branch: `main` — Config redesign: the DATA / INTERFACE / MODEL split

Phase 5 pulled forward, reshaped around one idea from the two pilots: the
config states the model's **demand** on the data pipeline explicitly, and the
pipeline's job is to satisfy it
([design](plans/2026-08-31-interface-config-redesign.md)).

- **`config.py` rewritten**: yacs is gone (`uv remove yacs`). Typed
  dataclasses — unknown keys refused with the full path, int→float coerced
  (`STRIDE_SECONDS: 0` works, `FS: 29.9796` is representable), `BASE:`
  include files deep-merged, so every `_SMOKE` config is now the real config
  plus a handful of overrides. The four per-split data blocks collapse into
  one `DATA` block + `SPLITS` (stride/random-windows/filter overrides only),
  which deletes `main.py`'s hand-maintained train/test consistency check
  outright. `INFERENCE` is absorbed into `TEST`; the dead legacy keys
  (`CHUNK_LENGTH`, `LABEL_TYPE`, `DO_PREPROCESS`, `BEGIN`/`END`, ...) no
  longer exist to write.
- **`INTERFACE` is the model's demand, and checkpoints carry it.**
  `ModelSpec` (unchanged in shape, so no builder changed) is now constructed
  from `INTERFACE`; `MultiSignalTrainer` saves
  `{state_dict, interface, model_name}` and at `MODE: only_test` `main.py`
  adopts the checkpoint's interface over the config's before building
  anything, printing the differences — a pretrained checkpoint is tested
  against what it actually expects, not what a config restates.
- **Demand-driven delivery in the zarr loader.** A demanded channel the
  *dataset* can never provide (not just a store) is zeros +
  `channel_mask=False` with one construction-time warning, so a wider
  pretrained checkpoint runs on narrower data; zero-coverage channels/traces
  warn loudly (training would teach the model to ignore them);
  `INTERFACE.UPSAMPLING: interpolate` opts a slower store into linear
  frame/label interpolation (refusal, naming the opt-in, stays the default —
  duplication would blank DiffNormalized).
- All seven `configs/neckflix/*.yaml` converted; suite green (280 tests,
  including both pilots' smokes). The DeepPhys/PhysFormer configs are drafts
  pending the contract's new **§7a re-verification** (one sub-agent per
  model). `test_legacy_contract` now hands the legacy tuple-path trainer a
  plain namespace instead of resurrecting the yacs tree.

## 2026-08-31 — Branch: `main` — Phase 4: the DeepPhys pilot (stage 0)

Executed per [the migration contract](plans/2026-08-31-model-migration-contract.md)
§7. The pilot's job was the shared infrastructure every later migration
assumes; DeepPhys is the worked example of using it.

### Stage 0 — shared infrastructure
- **Physical-time windowing.** `PREPROCESS.WINDOW_SECONDS` / `STRIDE_SECONDS`
  replace `CHUNK_LENGTH` / `CHUNK_STRIDE`, and `DATA.FS` becomes mandatory:
  `T = WINDOW_SECONDS x FS`, snapped to a whole frame within 0.01 and refused
  otherwise with the nearest valid duration named. 150 frames is 5 s of
  physiology at 30 fps and 1 s at 150 fps, and a frame count alone cannot tell
  them apart. The loader reads each store's own measured rate and decimates it
  to `FS` by nearest-index sampling; a genuinely slower store is refused
  (upsampling duplicates frames, and duplicated frames make DiffNormalized
  identically zero — a silently dark motion branch). Rates within 1% count as
  the same nominal rate, which is what the real cache needs: its 332 stores
  carry 29.9796 and 30.0 mixed *within* single recordings.
- **Per-signal label normalisation.** `PREPROCESS.LABEL_NORM` is a per-signal
  map with a new `raw` mode, and it moved out of the `NECKFLIX` block (it is
  not Neckflix-specific). Defaults come from a signal's *class*, now recorded
  in `neural_methods/signals.py`: absolute (ABP, CVP, SPO2) load raw in
  physical units because their level is part of the prediction; shape (PPG,
  ECG, RESP, EDA) stay per-window z-scored.
- **Per-signal composite loss.** `neural_methods/loss/PerSignalLoss.py`:
  `TRAIN.LOSS` becomes a registry of `{TYPE, WEIGHTS}` per trace over the
  components CCC / MEAN / MAX / MIN / NEGPEARSON / MSE / SPECTRAL, each reduced
  **per sample** so the masked structure composes and an absent signal
  contributes exactly 0. Systolic and diastolic come from differentiable soft
  peaks over the predicted waveform — no separate stats head, so a waveform can
  never disagree with its own statistics. The soft-peak softness is expressed
  as a fraction of the window's range, without which `PhysHydraLoss`'s absolute
  `tau`/`temperature` saturate to a hard argmax on a raw mmHg trace.
  `MaskedMultiSignalLoss.py` deleted; its structure lives on inside the new
  module.
- **Checkpoint-authority channel alignment.** `stack_frames` zero-fills a
  channel the model expects but the batch lacks instead of raising, so an
  RGB+IR+depth checkpoint runs on RGB-only data — degraded, not crashed. A
  batch carrying *none* of the model's channels is still an error.
- **Absolute-scale guardrails**, in the builder rather than any architecture:
  each output row's bias starts at its signal's physiological prior (ABP 90
  mmHg, CVP 8), and the readout is exempt from weight decay, which would
  otherwise drag raw-unit predictions toward zero as a systematic pressure bias.
  Models declare their readout via `DictModel.output_layers()` and any window
  constraint via `temporal_divisor` / `temporal_length`, checked at
  construction — the legacy trainers' silent batch truncation did not migrate.
- **Standard plots**, written once in `MultiSignalTrainer` /
  `evaluation/metrics_report.py`: per-signal and per-component training curves,
  per-signal waveform overlays, and predicted-vs-true agreement scatters
  (window mean, systolic, diastolic) for absolute-class signals — the
  migration-time precursor of the Phase 7 clinical bands.
- `ModelSpec` collects everything §1 says is derived from the data spec, so a
  `MODEL` block is `NAME` plus genuinely architectural keys.

### DeepPhys
- `in_channels` / `out_signals` parameterised, final dense activation-free,
  both sanctioned head styles (`MODEL.HEAD_STYLE`: `widened` = A, default;
  `per_signal` = B), `view` replaced with einops. At `3`/`1` with the default
  head it is the original, layer for layer and name for name.
- `configs/neckflix/NECKFLIX_DEEPPHYS.yaml` (+ smoke) — the template later
  migrations copy. `WINDOW_SECONDS: 6.0` reproduces upstream's
  `CHUNK_LENGTH: 180` at `FS: 30`.
- `DeepPhysTrainer.py` deleted.
- **Bug the pilot found:** `in_channels` means different things across model
  families. DeepPhys splits the stacked `DATA_TYPE` blocks into its motion and
  appearance branches itself, so its convs take *one* block's width, while
  PhysMamba consumes the stack whole. `ModelSpec` now names both
  (`camera_channels`, `in_channels`), and the DeepPhys builder refuses anything
  but exactly two `DATA_TYPE` entries.

Config friction is recorded in
[the pilot retro](plans/2026-08-31-deepphys-pilot-retro.md) for Phase 5.
Validated: full suite green, plus `--limit_windows 8 --test_participants P015`
smoke runs of both DeepPhys and PhysMamba against the real `rgb128` cache,
predicting ABP and CVP directly in mmHg.

---

## 2026-08-31 — Branch: `main` — Overhaul Phases 1–3

Executed per `docs/plans/2026-08-31-overhaul-roadmap.md`:

- **Phase 1**: repo renamed **remote-physiology**; root scratch, upstream
  weights (`final_model_release/`), `figures/` and stale docs deleted;
  `.slurm_scripts/` tracked as reference material; last legacy state tagged
  `pre-overhaul`.
- **Phase 2**: zarr cache contract documented dataset-agnostically in
  `docs/architecture.md`; the twelve legacy per-dataset loaders distilled into
  markdown cache specs (`dataset/data_loader/<NAME>.md`) and deleted along
  with `BaseLoader`, face detection and the `.npy`-cache tools (~39K lines);
  `neckflix_main.py` promoted to the single entry point `main.py`; the
  unsupervised predictor's legacy tuple path removed.
- **Phase 3**: attribute filtering generalized — the hardcoded
  `NECKFLIX.{POSTURES,PERSPECTIVES,LIGHT,SESSIONS}` keys replaced by one
  `NECKFLIX.FILTERS` node keyed by whatever root attrs a store carries
  (`FILTERS: {posture: ['0','45'], light: ['D']}`); `PARTICIPANTS`/CLI stay
  the separate, id-normalising LOSO surface. `BaseZarrDataset` was already
  attribute-generic; a new dataset is now a `channel_map` subclass plus a
  markdown cache spec, with no filter plumbing.

---

## 2026-08-31 — Branch: `main`

Carried the Neckflix loader's dicts through the whole pipeline; got all seven
unsupervised methods and PhysMamba running on the zarr cache.

Design: `docs/superpowers/specs/2026-08-31-neckflix-dict-contract-design.md`.

### The contract
- `neural_methods/batch.py`: the batch-dict key names and the two einops shape
  moves (`stack_frames`, `split_signals`) plus their inverses and traversal
  helpers. A model now returns *the input dict plus* `predictions`, so frames,
  labels, stats, masks, metadata and predictions travel together, each
  identifiable by key.
- `neural_methods/model/DictModel.py`: base class owning channel/trace order and
  the frame transform. Subclasses implement only
  `forward_video(video) -> (B, S, T)`. A plain tensor in still gets a plain
  tensor out, so the upstream tuple-contract trainers are unaffected.
- `neural_methods/frame_transforms.py`: `DATA_TYPE`
  (Raw/Standardized/DiffNormalized, concatenated along channels) and spatial
  resize moved consumer-side, since the zarr loader emits raw pixels.
- Every reshape in new or touched code is einops.

### Models and training
- `PhysMamba` is a `DictModel`: configurable input channels and output signals,
  `AdaptiveAvgPool3d((None, 1, 1))` so one model serves any `CHUNK_LENGTH`,
  einops throughout. Architecture unchanged.
- `SignalDictWrapper` rebuilt as a `DictModel` adapter (`frames2d` / `video3d`),
  which is how DeepPhys joins the contract untouched.
- `neural_methods/trainer/MultiSignalTrainer.py`: one trainer for every
  dict-contract model, driven by `MODEL_REGISTRY` (PhysMamba, DeepPhys). Masked
  multi-signal loss, DDP, AMP, signal-keyed saved outputs, and per-signal
  metrics reported in both normalised and **physical** units (mmHg for ABP/CVP,
  via each window's own `label_stats` and the exact inverse of its
  normalisation).
- `mamba_compat.MambaRef`: pure-PyTorch selective-SSM block used where
  `mamba_ssm` has no wheel (Windows), so PhysMamba is runnable and testable
  everywhere. Fused CUDA path untouched where the package exists; warns once
  about the memory cost of materialising the hidden state.

### Wiring
- `neckflix_main.py` rebuilt around the zarr loader: LOSO by participant filter,
  optional DDP, `--valid_participants`, `--limit_windows` for smoke runs, and a
  refusal when `USE_LAST_EPOCH: False` has no validation split to select on
  (which would otherwise silently test epoch 0).
- `dataset/data_loader/neckflix_config.py`: yacs -> loader plain-dict
  translation, including `P015` -> `"015"`.
- `config.py`: `CHUNK_STRIDE`, `TRAIN.LOSS`, and `NECKFLIX.{LABEL_NORM,
  ALLOW_MISSING, MIN_CHANNELS, MIN_LABELS, PERSPECTIVES, LIGHT, SESSIONS,
  PARTICIPANTS}`, added uniformly to all four data blocks.
- Configs: `configs/neckflix/NECKFLIX_UNSUPERVISED.yaml`,
  `NECKFLIX_PHYSMAMBA.yaml`, `NECKFLIX_PHYSMAMBA_SMOKE.yaml`.
- `tools/list_neckflix_folds.py` enumerates LOSO folds (metadata-only, login-node
  safe); `tools/summarise_neckflix_outputs.py` turns a saved test pickle into a
  per-window table and per-signal summary offline (replacing the broken
  `neckflix_metrics.py` scratch file's job); SLURM scripts for the unsupervised
  sweep, a LOSO array, and a 4-GPU fold.

### Unsupervised methods — three were hard-broken by NumPy 2
- **POS** and **ICA** used `np.mat`, removed in NumPy 2.0. Rewritten on plain
  ndarrays with einops reshapes; pinned in tests against frozen pre-NumPy-2
  originals (`tests/reference/`) to 1e-10 / 1e-8.
- **PBV** relied on `linalg.solve` treating a 2-D right-hand side as a stack of
  vectors, which NumPy 2.0 changed to a matrix solve. Fixed with an explicit
  trailing axis.
- POS's per-frame overlap-add is now a strided sliding window (~960x faster);
  GREEN/LGI/OMIT/CHROM reshapes moved to einops.
- `unsupervised_predict` accepts both dataset contracts and, for Neckflix,
  scores each window against **every** trace the recording carries (ABP/CVP/ECG),
  honouring `label_mask`. `unsupervised_predict_many` runs all seven methods in
  one pass over the data.
- `evaluation/metrics_report.py` extracted so both predictors share the metric
  aggregation.
- A fourth `np.mat` break sat outside `unsupervised_methods/`:
  `BaseLoader.generate_pos_psuedo_labels` and its verbatim twin in
  `BP4DPlusBigSmallLoader` each carried an inline copy of POS, so
  `USE_PSUEDO_PPG_LABEL: True` was broken for every dataset. Both now call the
  shared `POS_WANG.pos_signal`, pinned to the original to 1e-10.

### Shared post-processing (exact-equivalent, much faster)
- `_detrend`: the smoothness-priors detrend built a dense n x n matrix and
  inverted it. `I + lambda^2 D'D` is symmetric pentadiagonal, so it is now a
  banded Cholesky solve — O(n) instead of O(n^3), 0.75 s -> 0.09 ms at n=300,
  matching the dense form to 5e-11. Benefits every dataset.
- `_compute_macc`: the per-lag `corrcoef` loop is one circular cross-correlation;
  by FFT it is bit-identical and ~350x faster.

### Robustness fixes surfaced by running it
- Bland-Altman plots survive degenerate data (`gaussian_kde` raises on a
  singular covariance; density colouring now degrades to a constant).
- The HR report says *why* Pearson is undefined instead of printing
  `nan +/- nan`.
- `MultiSignalTrainer` indexes its GPU by `LOCAL_RANK`, not the global rank
  (multi-node correctness), and honours `DEVICE: cpu`; `neckflix_main` picks the
  DDP backend from where the model will actually live.

### mamba-ssm now builds on Windows (native, no WSL)
Three independent blockers, all confirmed rather than guessed:
- **No Windows wheel exists.** PyPI carries an sdist only; the GitHub release
  assets are `linux_x86_64` / `linux_aarch64`. Windows must compile
  `selective_scan_cuda` itself.
- **`triton` is unsatisfiable on Windows** — wheels-only on PyPI, zero Windows
  files, no sdist — and `selective_scan_interface` imports it unguarded. This
  bites before any compiler runs: uv resolves, then refuses to install anything.
  Fixed with `triton-windows` (the same 3.2 series torch 2.6 pairs with, which
  imports as `triton`) plus `override-dependencies` confining `triton` to Linux.
- **The CUDA sources do not compile with MSVC.** `#ifndef USE_ROCM` sits inside a
  `BOOL_SWITCH(...)` macro argument (UB; GCC accepts it, MSVC/EDG says `"#" not
  expected here`); `M_LOG2E` needs `_USE_MATH_DEFINES`; and `BOOL_SWITCH` passes
  an enclosing `constexpr` as a template argument from inside a lambda, which
  needs `/Zc:lambda` (`error C2975`).

`vendor/mamba-ssm` is upstream 2.3.1 with those three fixes, regenerated and
verifiable by `tools/vendor_mamba_windows.py` (`--check`). `pyproject.toml`
points `mamba-ssm` there **only** under `sys_platform == 'win32'`; the lock keeps
separate entries for linux-x86_64 (PyPI 2.3.1) and aarch64 (2.3.2.post1), so HPC
resolution is unchanged. `causal-conv1d` needed no patch, only the marker
widened. `uv sync` builds both in ~21 min and caches the result.

Verified against a Quadro RTX 5000 (sm_75), CUDA 12.4, MSVC 14.44:
- Kernel correctness: `selective_scan_cuda` vs `selective_scan_ref`, forward
  2.8e-05 on a signal of scale 168, worst gradient 3.0e-04 — fp32
  accumulation-order noise.
- PhysMamba at the production 128x128x128: **229 ms/step and 1.38 GiB**, against
  1959 ms and 5.13 GiB for the `MambaRef` fallback — 8.6x faster, 3.7x smaller.

### A latent bug this exposed
Installing `mamba-ssm` made PhysMamba and PhysHydra unrunnable on CPU — the
fused kernels raise `Expected x.is_cuda() to be true` — so 24 CPU tests failed
on exactly the machines that have the fast path. That was already true of the
HPC; Windows had been passing only because `MambaRef` stood in.
`mamba_compat.PortableMamba` subclasses `Mamba` and runs the reference math over
*the same module's own weights* off CUDA. The parameter sets are provably
identical (strict `load_state_dict` both ways), so there is one checkpoint and
one parameter set, and the two paths agree to 1e-07 in output and 4e-06 in
gradients.

### Verified
- 241 tests pass, none skipped, with `mamba_ssm` installed — the parallel-scan
  test that used to skip for want of the package now runs against it. Included
  is a regression test that drives the real `PhysMambaTrainer` over the upstream
  tuple contract, since PhysMamba and the post-processing both changed
  under the non-Neckflix datasets.
- All seven unsupervised methods over 1243 real windows (44 participants), the
  whole sweep in ~4 minutes on CPU. Best HR agreement against the ABP reference:
  CHROM (MAE 3.83 bpm, Pearson 0.64), POS (4.12, 0.62), ICA (5.79, 0.48);
  GREEN and PBV worst (~9.4). CVP and ECG references agree far less well, as
  expected — ABP is a pressure pulse, whereas an FFT "HR" from CVP or from
  bandpassed ECG is unreliable. The per-signal table says so rather than
  averaging them together.
- PhysMamba trains and tests end to end on the real cache predicting ABP + CVP +
  ECG together, on CPU at 32x32 and on GPU at the production 128x128x128. A
  small learning check (8 epochs, 600 windows at 72x72, P015 held out) takes the
  loss from ~0.99 to ~0.40 and reaches ABP waveform Pearson 0.76 / MACC 0.87 /
  HR MAE 8.7 bpm on the unseen subject -- untuned, but well clear of chance, and
  ordered ABP > CVP > ECG exactly as the unsupervised baselines suggest.

---

## 2026-08-25 — Branch: `main`

Unified branches, wired local Neckflix subset, replaced the vendored mamba fork.

### Changes
- Merged HPC and MacOS branches; absorbed phys-hydra history; deleted all branches except `main`
- Reduced drift vs upstream; upstream/main fully merged at b7500b8 (intentional divergence begins here)
- Added local physhydra configs targeting the Neckflix subset cache (P030-P039)
- Added `mamba-ssm-macos` (darwin) and newest official `mamba-ssm` + `causal-conv1d` (linux) as platform-marked deps
- **Removed vendored `tools/mamba` fork.** PhysMamba/PhysHydra now use vanilla mamba_ssm via `neural_methods/model/mamba_compat.py`: `make_mamba(..., bimamba=True)` returns a composition-based `BiMamba` (two vanilla blocks, one time-reversed, summed). NOT parameter-compatible with fork-trained checkpoints; fresh training required. Separate in/out projections per direction (small param increase vs fork's shared projections)
- MPS support: trainers select cuda -> mps -> cpu; parallel (Hillis-Steele) selective scan with analytical-backward autograd for non-CUDA mamba_ssm (~340x faster than CPU loop at debug res; full 128x128 res trains at ~19 s/step, 18 GB on M1 Pro)
- Added tests/test_mamba_compat.py (BiMamba semantics + parallel-scan-vs-reference)

## 2026-02-12 — Branch: `HPC`

Completed DDP migration for remaining models and added multi-GPU SLURM infrastructure.

### Changes
- Migrated FactorizePhys trainer to DDP with UBFC-rPPG validation config
- Migrated BigSmallTrainer to DDP, fixing pre-existing bugs in the process
- Migrated iBVPNet trainer to DDP with UBFC-rPPG validation config
- Added 2-GPU SLURM scripts for all models (EfficientPhys, FactorizePhys, PhysFormer, PhysMamba, PhysNet, RhythmFormer, TS_CAN, iBVPNet)
- Fixed SLURM port conflicts with dynamic master_port allocation
- Updated CLAUDE.md with GPU scheduling policy for development runs

---

## 2026-02-11 — Branch: `HPC`

Migrated all 7 main model trainers to DDP and validated the full UBFC-rPPG pipeline end-to-end.

### Changes
- Migrated EfficientPhys, TS_CAN, PhysNet, PhysMamba, PhysFormer, and RhythmFormer trainers to DDP
- Added UBFC-rPPG DeepPhys validation config and SLURM scripts (DeepPhys was already DDP-ready)
- Created `.configs/` directory with UBFC-rPPG validation YAML configs for all 7 models
- Validated full pipeline (preprocess, train, validate, model selection, test, metrics, Bland-Altman plots) for each model on V100 and A100 GPUs

---

## 2026-02-09 — Branch: `HPC`

Merged upstream changes from the original rPPG-Toolbox repository.

### Changes
- Merged upstream/main to bring in LADH and SUMS dataset support (upstream PR #405)

---

## 2025-11-06 to 2025-11-19 — Branch: `HPC`

Integrated the PhysHydra model and iterated on the Neckflix data loader and training pipeline.

### Changes
- Added PhysHydra model and trainer to the repository
- Registered PhysHydra as a model option in main.py with a global NUM_WORKERS argument
- Updated Neckflix loader with begin/end parameters, data_format support, squeeze fix, and unnormalised trace handling
- Fixed memory leakage in training loop and added config file saving to output directories
- Cleaned up loss function naming and removed debug memory printouts
- Added uv.lock to .gitignore

---

## 2025-09-05 to 2025-09-08 — Branch: `HPC`

Initial HPC branch setup with Neckflix dataset integration and environment configuration.

### Changes
- Created the HPC branch from the MacOS development branch
- Added NeckflixLoader with base HDF5 loading functionality for multimodal data (RGB/IR/Depth + ABP/CVP/ECG)
- Registered Neckflix dataset in main.py with default train/test splits
- Added initial experiment configs for Neckflix
- Configured HPC environment: pyproject.toml, adjusted Python version and requirements, updated setup.sh
- Set up .gitignore for venv and cache directories
- Reorganized test configs out of root directory

---

## Pre-HPC — Upstream contributions (selected)

Notable upstream changes present in the fork prior to HPC branch creation.

### Changes
- PhysFormerTrainer update (upstream PR #404, 2025-09-01)
- PhysDrive dataset support (upstream PR #393, 2025-05-25)
- Retinaface removal and documentation housekeeping (upstream PR #384, 2025-04-15)
- FactorizePhys model addition (upstream PR #371, 2025-03-08)
- RhythmFormer and PhysMamba fixes (upstream PR #369, 2025-03-01)
- Test-time chunk length fixes (upstream PR #362, 2025-02-23)
