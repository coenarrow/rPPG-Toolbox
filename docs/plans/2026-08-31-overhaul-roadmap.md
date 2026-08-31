# Overhaul Roadmap: rPPG-Toolbox → remote-physiology

Companion to `updating_plan.md` (the *what*); this document is the *order and
the mechanics*, plus the repo cleanup that plan doesn't cover. Phases are
sequenced by dependency: each one makes the next one smaller.

Ordering note: config work is **interleaved with model migration**, not done
up front. A couple of models migrate first against the current config, each
migration ends with a short config retro, and the consolidation happens once
real usage has shaped the schema (Phase 5).

## Cross-cutting rules

- **Dependencies go through `uv add`, never pip.** `pyproject.toml` + `uv.lock`
  are the single source of truth, and every addition considers all three
  platforms (Windows dev, Linux HPC, macOS dev/demos) — see the mamba-ssm
  platform split for what happens when a package doesn't.
- **All tensor reshaping uses einops** (`rearrange` / `reduce` / `einsum`),
  not `view` / `permute` / `reshape` — the shape spelled out at the call site
  is the point. Applies to migrated model code too, even where the original
  architecture used raw reshapes.
- **Testing stays minimal.** This is research code that will be open sourced,
  not production code for consumers. Specs and implementation plans should not
  demand exhaustive test suites: the existing contract tests plus one smoke
  test per migration is the ceiling, and errors get fixed as they appear.
- **The old base classes go, not get adapted.** `BaseLoader` (preprocessing
  machinery) and `BaseTrainer` are artifacts of the old contracts;
  `BaseZarrDataset` and `MultiSignalTrainer` replace them outright. Delete
  rather than maintain compatibility shims.
- Git history is the archive. Anything deleted (loaders, configs, weights,
  scratch) is one checkout of the `pre-overhaul` tag away.

---

## Phase 0 — Land what's in flight

The entire dict-contract changeset (~30 modified + ~30 untracked files,
including `vendor/`, `MultiSignalTrainer`, `batch.py`, the test suite) is
uncommitted. Nothing else happens until it's landed in a few logical commits
(vendoring/pyproject, contract core, tests, tools, docs). Include
`updating_plan.md` and this roadmap.

Then tag the result `pre-overhaul` so everything deleted later is one checkout
away.

## Phase 1 — Rename + repo hygiene

Cheap, mechanical, and best done *before* the overhaul so later diffs are
reviewable.

**Rename** to `remote-physiology` now, not at the end — GitHub redirects old
URLs, and every doc written in Phases 2–8 then carries the right name.

**Root scratch files** (all tracked, all deletable — git history keeps them):

| File | Action |
| --- | --- |
| `debug` | delete (captured debug output) |
| `test.py`, `plt_attention.py` | delete (scratch) |
| `pytorch_learning.py` | delete (confirmed: no relocation needed) |
| `metrics.ipynb`, `neckflix_metrics.ipynb`, `neckflix_metrics.py`, `bp_metrics.py`, `neckflix_example_metrics.csv` | park in `evaluation/prototypes/` — raw material for Phase 7; delete when Phase 7 consumes them |
| `requirements.txt`, `setup.sh` | delete — `pyproject.toml` is the source of truth |

**Tracked outputs**:

- `final_model_release/` — delete from the tree (decided). The 36 upstream
  single-signal `.pth` weights can't load into multi-signal variants and
  upstream still hosts them.
- `model_outputs/PURE_PURE_UBFC_deepphys_outputs.pickle` — untrack, ignore dir.
- `figures/` — deleted entirely (decision 9): the interim README carries no
  images, so nothing references it; Phase 8 pulls images back from history if
  the final README wants any.

**`.gitignore` fixes**:

- Remove `uv.lock` from `.gitignore` — it's tracked (and must be, as the
  reproducible-env artifact); the ignore entry is a no-op lie.
- Start tracking `.slurm_scripts/` (decided). These are **reference
  material** — templates to copy and adapt per experiment, not a maintained
  API — and should be treated as such in docs.
- Add `model_outputs/`; keep `runs/`, `logs/` ignored.

## Phase 2 — Cache contract + delete the legacy pipeline

The `updating_plan.md` "Caching" stage, plus its forced consequences.

1. **Document the cache contract** in README + `docs/architecture.md`: one zarr
   store per recording, `perspective → stream → video/frames + trace/data`,
   root attrs (`complete`, `tool_version`, participant, …), admission rules.
   Neckflix as the worked example.
2. **Replace each legacy loader with a markdown cache spec.** Twelve loaders
   (BP4D+, BP4D+BigSmall, COHFACE, LADH, MMPD, PhysDrive, PURE, SCAMPS, SUMS,
   UBFC-PHYS, UBFC-rPPG, iBVP). For each: read the loader, capture into
   `dataset/data_loader/<NAME>.md` everything a future cache-writer needs —
   raw file layout, video/trace formats, sampling rates, quirks (e.g. PURE's
   image sequences, SCAMPS' mat files) — *then* delete the `.py`. This is the
   only record of those parse details once the code is gone; the spec is the
   deliverable, the deletion is the afterthought.
3. **Consequences** (legacy loaders are load-bearing for the old pipeline):
   - `main.py` dies with the loaders → `neckflix_main.py` generalizes into the
     single entry point (renamed `main.py` at the end of this phase).
   - `BaseLoader.py` goes entirely (face crop, chunking, `.npy` cache, file
     lists) — replaced by `BaseZarrDataset`, no shim.
   - Legacy tools tied to the `.npy` cache (`tools/preprocessing_viz/`,
     `tools/output_signal_viz/`, `tools/motion_analysis/`) — audit, then
     delete.
4. **CLAUDE.md note**: "implementing a cache/loader for a new dataset → read
   the markdown spec in `dataset/data_loader/`."

Keep: `zarr_dataset.py`, `NeckflixLoader.py`, `neckflix_config.py`,
`label_transforms.py`, the unsupervised methods (already migrated).

## Phase 3 — Dataset & dataloading generalization

Mostly landed for Neckflix. Remaining work:

- Promote the Neckflix filter model to `BaseZarrDataset` generically:
  attribute include-filters driven by whatever attrs a store carries, not
  hardcoded posture/perspective/light.
- A new dataset = a `channel_map` subclass + a markdown cache spec.
- Standard dict keys stay owned by `neural_methods/batch.py`.

## Phase 3.5 — Dependency refresh

Deliberately slotted *here*: before this point a bump would break code
scheduled for deletion; after the migrations every model would need
revalidating against new APIs. This is the window where the living surface is
smallest and everything that survives has contract tests.

> **⛔ HARD PAUSE before starting this phase.** Check the system requirements
> on all three platforms before touching anything: Windows dev box (CUDA
> toolkit / MSVC for the vendored `mamba-ssm` build, `triton-windows`
> compatibility), Linux HPC (available CUDA modules and driver versions on the
> cluster — these cap the torch version), and macOS (`mamba-ssm-macos`, MPS
> support). No upgrade target is chosen until the constraint set from all
> three is known. This is a stop-and-review point, not a step in a batch.

Then, in two separate commits:

1. **General pass** — `uv lock --upgrade` for the uncoupled deps (zarr,
   numpy, scipy, einops, plotting, …); pyproject edits for any deliberate
   major bumps. Run the contract tests + one smoke run.
2. **The torch / mamba-ssm / triton cluster** — coupled: the vendored
   `mamba-ssm` compiles against a specific torch, `triton-windows` has its own
   torch matrix, and the HPC CUDA modules cap torch from the other side. One
   commit, verified with a smoke train run on **both** Windows and HPC before
   anything builds on it. If the cluster's modules aren't ready, defer this
   half — it must not block Phase 4.

## Phase 4 — Model migration, wave 1 (+ config retros)

Migrate the easiest models first, against the config system as it stands.
Each migration: `DictModel` with `forward_video(video) -> (B, S, T)`, a
`MODEL_REGISTRY` builder line, a config, a smoke test — and **delete the
legacy trainer** as each model moves onto `MultiSignalTrainer`.

The full instruction set for every migration (config sources, head styles,
per-signal losses and absolute scale, physical-time windowing, the
prediction contract, plots, the per-model recipe) is
[the migration contract](2026-08-31-model-migration-contract.md).

Wave 1 = the 2-D per-frame backbones (near-mechanical via
`SignalDictWrapper(input_mode='frames2d')`):

1. DeepPhys — **done 2026-08-31** (the pilot; §7a re-verified on the new
   schema)
2. TS-CAN
3. EfficientPhys

**After each migration, a config retro**: note what was awkward to express,
what was duplicated, what the yacs tree forced. These notes drive Phase 5 —
the schema gets designed from friction observed, not speculation.

## Phase 5 — Config consolidation

**Substantially landed 2026-08-31**, pulled forward after the two pilot
retros converged on the same friction list — see
[the design](2026-08-31-interface-config-redesign.md). What landed: the
DATA / INTERFACE / MODEL schema (`config.py` rewritten as typed dataclasses,
yacs deleted from the tree and from `pyproject.toml`), one `DATA` block +
per-split `SPLITS` overrides, `INFERENCE` absorbed into `TEST`,
checkpoint-carried `INTERFACE` with only_test adoption, demand-driven
delivery in the zarr loader (dataset-agnostic channel fill, zero-coverage
warnings, opt-in interpolating upsampling), and all seven
`configs/neckflix/*.yaml` re-pointed (`_SMOKE` = `BASE:` + overrides).

Still open in this phase:

- One config system: kill the Hydra split (`physhydra_configs/`), delete the
  legacy `configs/train_configs/` + `configs/infer_configs/` piles;
  `configs/` holds only current-format experiment files. (Deferred until the
  remaining migrations stop needing the legacy configs as the `T_orig`
  reference — contract §1.)
- ~~§7a of the migration contract: sub-agent re-verification of the converted
  DeepPhys and PhysFormer configs.~~ **Done 2026-08-31** — both passed with
  no architectural discrepancy; the schema gaps found are recorded in the two
  retros and feed this phase's remaining pass.

## Phase 6 — Model migration, wave 2

The remaining architectures, written directly against the new config, in
rough order of difficulty. Per `updating_plan.md`, the multi-signal head
design is decided collaboratively per model, staying true to each original
architecture:

1. **3-D conv** — PhysNet, iBVPNet, FactorizePhys: output head widens to S
   signals.
2. **Transformers** — PhysFormer, RhythmFormer: head + tokenization decisions.
3. **Special cases** — BigSmall (already multi-task; map its task heads onto
   the signal dict) and PhysHydra (our own model, still on the legacy tuple
   contract).

Standardized plots are implemented **once** in `MultiSignalTrainer` /
`metrics_report.py`, so cross-model consistency is free.

Exit criterion: `neural_methods/trainer/` contains `MultiSignalTrainer` and
nothing else — `BaseTrainer` is deleted with the last legacy trainer, not
kept as a parent.

## Phase 7 — Evaluation & clinical metrics

Can overlap Phases 4–6 — it consumes saved outputs through the stable batch
contract.

1. **Design doc first**: map IEEE 1708-2014 / 1708a-2019, ISO 81060-2:2018 /
   81060-3:2022, and ESH 2023 onto computable metrics — per-beat
   systolic/diastolic detection, mean-error/SD acceptance bands, per-subject
   vs pooled aggregation, grading. Be explicit about which criteria are
   *computable from our data* vs *study-design requirements* (subject counts,
   reference-device protocol, cuff procedure) that a metrics report can note
   but not satisfy.
2. Extend `evaluation/metrics_report.py`; denormalised (mmHg) reporting via
   the `label_stats` already carried in every batch.
3. Consume, then delete, `evaluation/prototypes/` from Phase 1.

## Phase 8 — Docs finalization

- README rewritten for `remote-physiology`: mission, cache contract,
  batch-dict contract, model table, clinical-metrics summary; any images it
  wants come back from git history (`figures/` was deleted in Phase 1);
  upstream rPPG-Toolbox credited as the fork origin. An interim
  accuracy-pass README/CLAUDE.md landed during Phase 1 — this phase is the
  final polish, not the first correction.
- CLAUDE.md refreshed to remove legacy-pipeline instructions.
- `docs/architecture.md`, `docs/changelog.md`, `docs/project_status.md`
  updated; `updating_plan.md` retired into `docs/plans/`.

---

## Decision log (2026-08-31)

1. Repo renamed to **remote-physiology** (executes in Phase 1).
2. `final_model_release/` deleted from the tree in Phase 1.
3. `.slurm_scripts/` tracked, as reference material only.
4. `pytorch_learning.py` deleted, no relocation.
5. Config consolidation happens **after** wave-1 model migrations, informed
   by per-migration retros — not up front.
6. Testing kept minimal throughout (see cross-cutting rules).
7. `BaseLoader` and `BaseTrainer` are deleted, not adapted — the new
   contracts (`BaseZarrDataset`, `MultiSignalTrainer`) replace them.
8. Dependency refresh happens between Phases 3 and 4 (Phase 3.5), preceded by
   a hard pause to check system requirements on all three platforms; the
   torch/mamba-ssm/triton cluster is its own commit and may be deferred.
9. `figures/` deleted entirely in Phase 1 (not pruned in Phase 8): the
   interim README carries no images. Executed superpowers plans/specs and the
   2026-02 UBFC validation plan also deleted — process artifacts, retrievable
   from git history; `docs/architecture.md` is the living contract reference.
10. Phase 3 filter design: one generic `NECKFLIX.FILTERS` yacs node
    (`new_allowed`, so YAML can name any attr) keyed by store root attrs plus
    the `perspective` pseudo-attr; the fixed
    `POSTURES`/`PERSPECTIVES`/`LIGHT`/`SESSIONS` keys are deleted, not
    aliased. Participants remain a separate surface (`PARTICIPANTS` /
    `--test_participants`) because their ids are normalised (`P015` → `015`);
    a `participant` key inside `FILTERS` is refused rather than allowed to
    bypass that normalisation. Experiment names derive their filter segment
    generically from `FILTERS`.
11. Phase 3.5 executed from the three platform reports
    (`docs/plans/platform-reports/`, generated by
    `tools/platform_report.py`): **torch 2.12.1 + cu126 everywhere** — the
    V100 partition is sm_70 (dropped from cu128+/cu130), the cluster driver
    (575.57) caps CUDA at 12.9, and triton-windows (3.8 = torch 2.12) rules
    out 2.13 on Windows. mamba-ssm held at 2.3.1, pinned exact on all
    platforms (2.3.2+ drags tilelang/quack-kernels, Linux-only). Driver
    uniformity across GPU nodes assumed (only k177 was sampled); whether
    triton 3.8 still JITs for sm_70 is verified by the first V100 smoke run,
    with A100/H100 as the fallback partitions. pyproject audited to a
    minimal floor-pinned set: exact pins only where blocking, dev tooling in
    the `dev` group (`uv sync --no-dev` on the HPC), orphans dropped
    (pyqt5, opencv, thop, tensorboardX, scikit-image, neurokit2, pdf/crypto
    tools).
12. Phase 4 redesigned around
    [the migration contract](2026-08-31-model-migration-contract.md), which
    supersedes the bare recipe above. Wave 1 rescoped to a **DeepPhys
    pilot** that lands the shared infrastructure (physical-time windowing —
    `WINDOW_SECONDS` + mandatory `FPS`, tolerance-snapped exact T,
    decimating loader, upsampling refused; per-signal label norm with `raw`
    physical units for ABP/CVP; the per-signal composite loss — CCC + L1
    mean/soft-peaks in mmHg for absolute-class signals, negpearson for
    shape-class — inside the masked per-sample structure; `stack_frames`
    checkpoint-authority zero-fill; standard plots), after which each
    remaining model is one sub-agent following the contract. Decided
    against: global/dataset normalisation constants (models predict
    physical units off an activation-free head; per-signal scale factors
    live in loss weights), a separate stats head (stats derive from the
    predicted waveform), and per-window norm for pressure signals (needs
    test-time ground truth — circular). TS-CAN/EfficientPhys follow the
    pilot rather than accompanying it.
13. **PhysFormer migrated ahead of its Phase 6 slot**, at the user's
    explicit direction, alongside (not after) the DeepPhys pilot. Ordering
    caveat recorded rather than silently absorbed: the contract schedules
    transformers for Phase 6 precisely so head and tokenization are decided
    collaboratively, and here they were decided by the migrating agent and
    written up for review in
    [the PhysFormer retro](2026-08-31-physformer-migration-retro.md).
    The decisions: **tokenization unchanged** from the paper (3-D stem then
    a 4x4x4 tube embedding), **head style A** (widened `Conv1d` readout),
    and **style B declared unavailable** for this architecture — its head
    reads a feature whose token grid is already mean-pooled away, so
    per-signal head copies would each see the identical vector; the
    equivalent would be a per-signal pooling over the token grid, which is a
    new head, not a builder flag. PhysFormer's published DLDL frequency/KL
    loss did **not** migrate: it needs one heart rate per window (undefined
    for ABP level or CVP) and was hardcoded to CUDA. The frequency term is
    expressed through the per-signal loss registry's `spectral` component
    instead — the right slot, but a log-spectrum shape match rather than
    DLDL's soft classification over a bpm grid, and the one place this
    migration is knowingly weaker than the paper.
14. **The Neckflix cache's nominal frame rate is not its exact one.** All
    332 stores of the local `rgb128` cache carry three distinct per-stream
    `video.fps` values — 29.97961373390558 (x329), exactly 30.0 (x325) and
    29.98051282051282 (x1) — mixed within single recordings. Stage 0's
    first cut refused `FS: 30` outright as an upsample, which blocked every
    Neckflix config in the repo; the fix is a *relative* nominal-rate
    tolerance (`same_nominal_rate`), so `FS` names the rate the config
    intends and the loader reconciles it per store. Consequence to keep in
    mind when reading configs: "160 frames at 30 fps is 5.333333 s" is a
    nominal identity, not an arithmetic one.

15. **Stage-0 config surface (the DeepPhys pilot's precedent).** The
    physical window is `PREPROCESS.WINDOW_SECONDS` / `STRIDE_SECONDS` with
    the frame count derived and tolerance-snapped; the target rate is the
    repo's existing `DATA.FS`, made mandatory, rather than a second `FPS`
    key saying the same thing. `LABEL_NORM` moved out of the `NECKFLIX`
    block up to `PREPROCESS` (it is not Neckflix-specific) and became a
    per-signal map; `TRAIN.LOSS` changed from a global base-loss string to a
    per-signal `{TYPE, WEIGHTS}` registry. Both default from a signal's
    *class*, newly recorded in `neural_methods/signals.py` — absolute
    (ABP/CVP/SPO2, loaded raw in physical units, scored with CCC + L1 mean
    and soft peaks) or shape (PPG/ECG/RESP/EDA, per-window z-scored, scored
    with negpearson) — so an ordinary config states neither key.
    `MaskedMultiSignalLoss` was **deleted** rather than kept beside the new
    `PerSignalLoss`: its masked per-sample structure is retained inside the
    new module, and two parallel losses with one trainer would have been a
    second way to say the same thing. `CHUNK_STRIDE` is gone;
    `CHUNK_LENGTH` survives only as inert legacy yacs bulk that Phase 5
    deletes. Friction observed is written up in
    [the DeepPhys pilot retro](2026-08-31-deepphys-pilot-retro.md), whose
    headline item for Phase 5 is that the four `DATA` blocks are one fact
    stated four times.

Last updated: 2026-08-31
