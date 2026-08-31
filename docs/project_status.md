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

## In Progress

- Neckflix zarr cache generation (`rgb128`; participants still being added)
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
- Pressure-specific metrics: the migration-time precursors are in
  (predicted-vs-true window mean / systolic / diastolic scatters per absolute
  signal); Phase 7 turns them into IEEE 1708 / ISO 81060 agreement bands with
  per-subject aggregation
- A first real LOSO sweep to tune the per-signal loss weights — the current
  `CCC 1.0` / `L1 1/scale` split only guarantees the terms are the same order
  of magnitude, and the per-component curves are what should settle it
- IR/depth channels: the contract and loader already support them; no cache has
  been generated with them yet
- Full LOSO sweeps on HPC (`.slurm_scripts/Neckflix_PhysMamba_LOSO.slurm`)
- HPC verification of the Phase 3.5 stack: fresh clone, `module load cuda`
  (12.6.3), `uv sync --no-dev`, then a PhysMamba smoke via SLURM on a
  **V100** node — the one open question is whether triton 3.8 still JITs
  for sm_70 (A100/H100 are safe regardless)
- Roadmap Phases 4–8: model migrations, config consolidation, clinical
  metrics, docs finalization

---

Last updated: 2026-08-31
