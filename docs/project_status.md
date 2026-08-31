# Project Status: remote-physiology

## Completed

- Overhaul Phases 0–3.5 (see
  [the roadmap](plans/2026-08-31-overhaul-roadmap.md)): repo renamed to
  **remote-physiology** with root scratch, upstream weights and stale docs
  cleaned out; the zarr cache contract documented dataset-agnostically in
  `docs/architecture.md`; the twelve legacy per-dataset loaders distilled
  into markdown cache specs (`dataset/data_loader/*.md`) and deleted along
  with `BaseLoader`, face detection and the `.npy`-cache tools; the zarr
  entry point promoted to `main.py` (single entry point, batch-dict
  contract only); attribute filtering generalized to `NECKFLIX.FILTERS`
  (any store root attr, no hardcoded key list), so a new dataset is a
  `channel_map` subclass plus a markdown cache spec; dependencies refreshed
  from the three platform reports — torch 2.12.1+cu126 fleet-wide,
  pyproject audited to a minimal floor-pinned set (verified on Windows GPU;
  HPC V100 smoke pending)
- Base toolbox setup with `uv` package management
- Multi-GPU distributed training setup
- Neckflix loader rebuilt on the external zarr cache (lazy, metadata-only
  construction, participant/posture/perspective filters, per-window label
  normalisation)
- Batch-dict contract end to end: loader dicts survive models, losses and
  evaluation, keyed by canonical channel/signal name (see
  `docs/architecture.md`; the original design spec lives in git history)
- All seven unsupervised methods running on Neckflix, scored per trace
  (POS/ICA/PBV also repaired after NumPy 2 removed the APIs they used)
- PhysMamba running on Neckflix, predicting ABP + CVP + ECG together, verified
  at production resolution on GPU (untuned learning check: ABP waveform Pearson
  0.76 on a held-out subject after 8 epochs over 600 windows)
- `MultiSignalTrainer` + `MODEL_REGISTRY`: adding a model is a builder function
  and a registry line, not a new trainer
- PhysFormer migrated onto the dict contract (Phase 4, ahead of its Phase 6
  slot; see [the retro](plans/2026-08-31-physformer-migration-retro.md)):
  widened stem + activation-free `Conv1d` readout, token grid derived rather
  than hardcoded, legacy trainer and DLDL loss computer deleted. Verified
  numerically identical to the pre-migration module at 3-in/1-out, and run
  end-to-end for two epochs on GPU at the published 128x128 / 160-frame
  settings — a plumbing check, not a learning result (two epochs over 2000
  windows leaves the readout near its prior)
- Pure-PyTorch Mamba fallback, so PhysMamba is runnable and testable without a
  `mamba_ssm` wheel *and* on CPU where one is installed (`PortableMamba`; the
  fused kernels are CUDA-only and otherwise make the model CPU-unusable)
- `mamba-ssm` builds natively on Windows: `vendor/mamba-ssm` carries the three
  MSVC fixes upstream lacks, `triton-windows` replaces the Linux-only `triton`,
  and `uv sync` does the rest. 8.6x faster and 3.7x smaller than the fallback at
  128x128x128

- **DeepPhys pilot (roadmap Phase 4) — done.** §7 stage 0 of the migration
  contract is built and in use by every dict-contract model: physical-time
  windowing (`WINDOW_SECONDS` + mandatory `FS`, tolerance-snapped `T`,
  decimating loader, upsampling refused), per-signal label normalisation with
  `raw` physical units for ABP/CVP, the per-signal composite loss
  (`PerSignalLoss`: CCC + L1 window mean + L1 soft systolic/diastolic, masked
  per sample), checkpoint-authority channel zero-fill in `stack_frames`, and
  the standard plot set (per-signal/per-component loss curves, waveform
  overlays, absolute-agreement scatters). DeepPhys itself is migrated
  (`NECKFLIX_DEEPPHYS.yaml`, both head styles, legacy trainer deleted); the
  friction is written up in
  [the pilot retro](plans/2026-08-31-deepphys-pilot-retro.md)

- **Config redesign: the DATA / INTERFACE / MODEL split (Phase 5 pulled
  forward) — done.** `config.py` is a typed schema (yacs removed);
  `INTERFACE` states the model's demand on the data pipeline and rides in
  every checkpoint (adopted over the config at `only_test`); the zarr loader
  delivers on demand (dataset-agnostic channel zero-fill + masks,
  zero-coverage warnings, opt-in `UPSAMPLING: interpolate`); one `DATA`
  block + per-split overrides replaces the four copy-paste blocks and the
  hand-maintained consistency check. All seven configs converted, `_SMOKE`
  variants now `BASE:` + overrides; suite green (280).
  [Design](plans/2026-08-31-interface-config-redesign.md); contract §1/§4/§7a
  updated.

- **§7a re-verification of DeepPhys and PhysFormer — both passed** (one
  sub-agent per model, 2026-08-31): configs carry the pilots' settings, built
  models are key-for-key the pilots' `state_dict`s, suite + real-cache smokes
  green, checkpoint-interface adoption exercised with deliberately wrong
  interfaces. No architectural discrepancy; the schema gaps found
  (load-bearing `DATA_TYPE` order, `LABEL_NORM` serialized as omission not
  resolution, frame-shape demands as errors not declarations, duplicated
  batch-size key) are recorded in the two retros for Phase 5's remaining pass

- **PURE on the zarr cache**: `tools/cache_pure.py` (offline bridge from the
  raw PNG+JSON recordings; timestamp alignment by default, recorded in the
  store's `alignment` attr), `PUREDataset` (`channel_map`-only
  `BaseZarrDataset` subclass), `PURE.md` upgraded to the implemented mapping,
  one smoke test (suite now 288). Local four-recording subset cached at
  `D:/pure_zarr`

- **Phase 5 closed** (2026-08-31): the schema slimmed to YAML-writable keys
  only (`LOG_PATH` / `UNSUPERVISED_METHODS` flattened,
  `EVALUATION_WINDOW_SECONDS`, derived paths on `config.RUN`, `RESIZE`
  square shorthand, YAML 1.2 floats, `LABEL_NORM` serialized resolved so
  checkpoints are self-describing), and the 141 legacy config files
  (`configs/train_configs/`, `configs/infer_configs/`,
  `physhydra_configs/`, the pre-overhaul `.configs/`) distilled into the
  migration contract's legacy settings reference and deleted — `configs/`
  holds only `neckflix/`. Suite 289; real-cache smoke green

- **Phase 7 closed (2026-09-01): evaluation rebuilt as a layered library.**
  `evaluation/` is now `records` → `beats` → `levels` → `uncertainty` →
  `scoring/` (`waveform`, `rate`, `clinical`, `standards`) → `report` →
  `plots`: one tidy frame (`level x unit x signal x metric x value x se`)
  computed once and consumed identically by the trainer, the unsupervised
  predictor and the offline LOSO summariser. Beats are reference-anchored
  (non-maximum suppression + a pulsatility gate against a model that finds
  no real beats); standard errors are HAC where samples are autocorrelated
  and moving-block bootstrap where a statistic (Pearson, CCC) has no closed
  form. Absolute-class signals (ABP, CVP) carry per-signal beat labels
  (systolic/MAP/diastolic for ABP, peak/mean/trough for CVP) through the
  agreement scatters and the per-subject Bland-Altman plots; IEEE 1708
  grading and the ISO 81060-3 pooled mean-error/SD verdict are computed at
  the participant and cohort levels, `n`-gated so a single-subject LOSO fold
  declines to grade rather than reporting a meaningless pass/fail.
  `TEST.METRICS` is gone, replaced by `TEST.REPORT.BOOTSTRAP` /
  `TEST.REPORT.PLOTS` — what applies to a signal follows from its class, so
  these two keys gate cost only, never content. `evaluation/prototypes/`
  (the notebooks this design was mined from) is deleted: every capability it
  held has a home — `find_peaks` in `evaluation/beats.py`; `mean_se` and the
  moving-block bootstrap in `evaluation/uncertainty.py`; the `get_rmse` /
  `get_mae` / `get_pearson_r` / `get_ccc` / `get_macc` family in
  `evaluation/scoring/waveform.py`; `get_hr_fft` / `get_snr` in
  `evaluation/scoring/rate.py`; `aggregate_data` split across
  `evaluation/levels.py` (the contiguity/grouping rules) and
  `evaluation/report.py` (the aggregation itself); the Bland-Altman cells in
  `evaluation/plots.py`. Verified end to end with a real smoke run
  (`NECKFLIX_PHYSMAMBA_SMOKE`, the 332-store local zarr cache, held-out
  participant 015): the digest printed the threshold-provenance block with
  its `UNVERIFIED` warning and the unmet study-design notes, ABP/CVP beats
  were labelled correctly, waveform/agreement/Bland-Altman plots were
  written for the two absolute signals and a waveform plot for ECG, and
  `*_metrics.csv` / `*_report.json` / the outputs pickle were all written.
  Suite at 288, green. **The clinical thresholds in
  `evaluation/scoring/standards.py` are deliberately left `UNVERIFIED` and
  must be checked line by line against the purchased ISO 81060-3:2022 and
  IEEE 1708-2014 / 1708a-2019 texts before any report from this phase is used
  to support a clinical claim** — Phase 7 delivers the computable machinery
  and its provenance trail, not a verified clinical grade.

- **HPC verification of the Phase 3.5 stack — passed** (2026-08-31): the
  torch 2.12.1 + cu126 stack builds and runs on the cluster from a fresh
  clone (`uv sync --no-dev`), confirmed by a PhysMamba smoke. Decision 11's
  open sm_70 question is closed if that smoke ran on the **V100** partition;
  the partition used is not recorded here.

- **Neckflix cache regenerated with all camera streams** (2026-09-01):
  `D:/neckflix_zarr/rgbid256` — all 332 recordings / 50 participants from
  `Z:/Dataset/Neckflix`, RGB + IR + depth (event camera excluded), 256x256,
  written by `neckflix-preprocess` 1.0.0 (native `uv`, no Docker — the ECF
  plugin is only needed for events). 331 cached + 1 skipped (a pre-run
  trial), 0 failed; 71 GB; every store admissible (`complete: true`,
  `tool_version 1.0.0`, `resized_to [256,256]`). Verified a faithful
  superset of the RGB-only `C:/Users/20759193/neckflix_cache/rgb128`:
  identical frame counts, fps and trace coverage on P001_S01_R1_0_D.
  Coverage across the loader's 655 samples (recording x perspective):
  **rgb 100%, ir 99%, depth 53%**; trace coverage per recording is
  ABP+CVP+ECG 206, CVP+ECG 109, CVP 8, ABP+CVP 4, ABP+ECG 2, none 3.
  `CACHED_PATH` in `configs/neckflix/` still points at the old `rgb128` —
  re-pointing is the one outstanding step.

## In Progress

- PhysHydra model development — still on the legacy tuple contract, not yet a
  `DictModel`

## To Do

- Wire dataset selection into `main.py`: `DATA.DATASET` exists in the schema
  but the entry point hardcodes `NeckflixDataset`, so `PUREDataset` is not
  yet reachable; a full-dataset PURE cache (all subjects/setups) is the other
  half
- Remaining model migrations per
  [the migration contract](plans/2026-08-31-model-migration-contract.md),
  now that stage 0 exists: one agent per model for TS-CAN, EfficientPhys,
  PhysNet, iBVPNet, FactorizePhys, RhythmFormer, BigSmall; PhysHydra follows
  its own path
- A first real LOSO sweep to tune the per-signal loss weights — the current
  `CCC 1.0` / `L1 1/scale` split only guarantees the terms are the same order
  of magnitude, and the per-component curves are what should settle it
- IR/depth experiments: the `rgbid256` cache now carries them, but **depth
  is present in only 53% of samples** (344/655) against IR's 99% — a
  depth-consuming config wants `ALLOW_MISSING` and should expect
  `channel_mask['D']` to be load-bearing, not incidental
- Full LOSO sweeps on HPC (`.slurm_scripts/Neckflix_PhysMamba_LOSO.slurm`)
- Evaluation follow-ups before the Phase 7 machinery supports a clinical
  claim (spec §17): verify every constant in `evaluation/scoring/standards.py`
  against the purchased ISO 81060-3:2022 and IEEE 1708-2014 / 1708a-2019
  texts, then flip `VERIFIED_AGAINST_STANDARD_TEXT`; run the beat detector
  over a full recording from `CACHED_PATH` and confirm the detected rate
  against the ECG-derived rate, including on the noisiest posture; confirm
  the participant/cohort levels on a real LOSO sweep once one exists
- Roadmap Phases 4, 6, 8: model migrations, docs finalization (Phase 5
  config consolidation and Phase 7 evaluation are closed)

---

Last updated: 2026-09-01
