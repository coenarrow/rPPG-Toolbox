# Overhaul Roadmap: rPPG-Toolbox → remote-physiology

Companion to [the revised overhaul plan](../../revised_overhaul_plan.md)
(the *what*; it superseded `updating_plan.md` on 2026-09-01 — git history
keeps the original). This document is the *order and the mechanics*. Phases
are sequenced by dependency: each one makes the next one smaller.

The remaining work was re-planned on 2026-09-01 around **contract v2** —
[the design](2026-09-01-contract-v2-design.md): cache contract v2
(validator-gated, `timestamps_us`, per-trace `units`, fixed vocabularies)
and model contract v2 (style-C parallel copies by default, losses computed
inside the model and riding the batch). Decision log entry 17 records the
choices; the original phase texts for 4/6/8 are in git history.

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
- **The old base classes go, not get adapted.** Delete rather than maintain
  compatibility shims.
- Git history is the archive. Anything deleted (loaders, configs, weights,
  scratch) is one checkout of the `pre-overhaul` tag away.

---

## Completed phases (history)

Details live in [project_status.md](../project_status.md) and the linked
design docs; the summaries here exist so the live phases below read in
context.

- **Phase 0 — Land what's in flight** (done): the dict-contract changeset
  landed in logical commits; result tagged `pre-overhaul`.
- **Phase 1 — Rename + repo hygiene** (done): renamed to
  `remote-physiology`; root scratch, `final_model_release/`, `figures/`,
  `requirements.txt`/`setup.sh` deleted; `.slurm_scripts/` tracked as
  reference material; `.gitignore` fixed.
- **Phase 2 — Cache contract + delete the legacy pipeline** (done): twelve
  legacy loaders distilled into markdown cache specs
  (`dataset/data_loader/*.md`) and deleted with `BaseLoader` and the
  `.npy`-cache tools; the zarr entry point promoted to `main.py`.
- **Phase 3 — Dataset & dataloading generalization** (done): generic
  attribute filters; standard dict keys owned by `neural_methods/batch.py`.
- **Phase 3.5 — Dependency refresh** (done): torch 2.12.1 + cu126
  fleet-wide from the three platform reports (decision 11); verified on
  Windows and HPC.
- **Phase 4 (original) — wave-1 migrations**: DeepPhys landed as the pilot
  (decision 12) and PhysFormer ahead of its slot (decision 13); TS-CAN and
  EfficientPhys were **not** migrated before the 2026-09-01 re-plan folded
  the remainder into Phases B–C below.
- **Phase 5 — Config consolidation** (closed 2026-08-31): the
  DATA / INTERFACE / MODEL typed schema
  ([design](2026-08-31-interface-config-redesign.md)), yacs and the 141
  legacy config files deleted, checkpoint-carried `INTERFACE`.
- **Phase 7 — Evaluation & clinical metrics** (closed 2026-09-01): the
  layered `evaluation/` library ([design](2026-08-31-evaluation-clinical-metrics.md),
  decision 16); IEEE 1708 + ISO 81060-3 implemented, thresholds
  `UNVERIFIED` pending the line-by-line check (spec §17).

The original Phase 6 (wave-2 migrations) and Phase 8 (docs finalization)
never started under their old definitions; their successors are Phases C
and E below.

---

## Phase A — Cache contract v2

The contract: [contract-v2 design, Part 1](2026-09-01-contract-v2-design.md).
The external preprocessor (the Neckflix repo) is being updated to write it
in parallel; this repo owns the validator and the reader.

1. **Validator first, now** — `tools/validate_cache.py` + importable
   `validate_store`: the contract made executable, runnable against stores
   as the preprocessor work produces them, *before* this repo's reader
   changes. One smoke test. This is the only admission mechanism —
   `complete`/`tool_version` checks are gone.
2. **Reader adoption, when a regenerated cache exists**: `BaseZarrDataset`
   reads `video/data`, perspective-level `fps`, `timestamps_us`; unequal
   modality durations truncated to the shortest at read time; `label_units`
   added to the batch dict from the traces' `units` attrs.
3. **One dataset class**: the global modality→channel table replaces
   per-dataset `channel_map` subclasses; `NeckflixDataset` and `PUREDataset`
   deleted. The markdown cache specs stay, updated to describe v2 stores.

Gate: the regenerated Neckflix cache passes the validator and a smoke run
end to end.

## Phase B — Model contract v2

The contract: [contract-v2 design, Part 2](2026-09-01-contract-v2-design.md).
Independent of Phase A — runs against the current `rgbid256` cache.

1. `DictModel` base: `forward(batch) -> batch` with `predictions`,
   `raw_losses` (model-written, unweighted) and `losses` (trainer-weighted)
   riding the batch; the per-signal loss machinery invoked from the base
   (written once — simple models inherit it); `Reads:/Modifies:`
   docstring convention.
2. The **style C wrapper** — S parallel copies of the original
   architecture, input widened to the demanded channels — as
   `HEAD_STYLE: parallel`, the new default; A/B remain options.
3. Rework DeepPhys, PhysFormer, PhysMamba onto the new contract;
   `MultiSignalTrainer` slims to forward / weight-and-sum / step / log.
   `PhysMambaTrainer.py` (missed in the PhysMamba migration) is deleted
   here.
4. Update [the migration contract](2026-08-31-model-migration-contract.md)
   to contract v2 **before** Phase C dispatches any agent.

Gate: smoke runs for all three reworked models; suite green.

## Phase C — Remaining migrations

One agent per model against the updated migration contract, in rough order
of difficulty: TS-CAN, EfficientPhys (2-D, near-mechanical), PhysNet,
iBVPNet, FactorizePhys (3-D conv), RhythmFormer (transformer), BigSmall
(multi-task heads onto the signal dict). **PhysHydra last and on its own
path** — it is the CardioHydra-pattern composite (per-stage losses beside
the per-signal entries), not a style-C wrap.

Each migration deletes its legacy trainer. With the last one go
`BaseTrainer` and the legacy evaluation trio it kept alive —
`evaluation/metrics.py`, `evaluation/BlandAltmanPy.py`,
`evaluation/bigsmall_multitask_metrics.py` (`evaluation/post_process.py`
stays: `scoring/rate.py`, `scoring/waveform.py` and the unsupervised
methods use it).

Exit criterion: `neural_methods/trainer/` contains `MultiSignalTrainer` and
nothing else.

## Phase D — Evaluation follow-ups

- Verify the ISO 81060 thresholds and aggregation line by line against the
  purchased texts in `standards/ISO-81060/` (spec §17 of
  [the Phase 7 design](2026-08-31-evaluation-clinical-metrics.md)); flip
  `VERIFIED_AGAINST_STANDARD_TEXT` for what passes. Note the on-disk
  81060-2 edition is 2019+A2:2024, newer than the 2018 edition the design
  cites.
- IEEE 1708 stays, `UNVERIFIED`, until its text is sourced (decision 17).
- Report per-signal `units` from the cache (Phase A's `label_units`)
  instead of assuming mmHg.
- Run the beat detector over a full recording and confirm the detected rate
  against the ECG-derived rate; confirm participant/cohort levels on a real
  LOSO sweep once one exists.

## Phase E — Templates + docs finalization

- The **extend-the-package templates** — new dataset, new model, new trace
  (label) — written for agent reuse once the contract is proven on Phase
  B's three reworked models.
- README rewritten for `remote-physiology`: mission, cache contract v2,
  batch-dict contract, model table, clinical-metrics summary; images from
  git history if wanted; upstream rPPG-Toolbox credited.
- CLAUDE.md and `docs/architecture.md` rewritten to the implemented
  contract v2; `docs/changelog.md` and `docs/project_status.md` updated.

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
    *(2026-09-01: the style-B finding becomes moot under contract v2's
    style C — full parallel copies always compose.)*
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

16. **Phase 7 evaluation design (2026-09-01).** The package lives at
    `evaluation/scoring/`, not `evaluation/metrics/` — a package named
    `metrics/` would shadow the retained legacy module `evaluation/metrics.py`
    (still imported by the legacy trainers that die in Phase C).
    Four design choices, carried through
    from [the design doc](2026-08-31-evaluation-clinical-metrics.md):
    (a) **a six-level hierarchy** (beat → window → section → recording →
    participant → cohort) rather than one fixed aggregation, because the
    standards themselves disagree on the right level (ISO 81060-3 wants a
    per-subject mean/SD, IEEE 1708 a per-subject MAE) and a report that
    hardcodes one loses the others; a `Section` is a maximal contiguous run
    of windows stitched back into one trace, so beat detection sees a
    continuous recording rather than window-sized fragments, and overlapping
    windows safely degrade to one section each rather than double-counting;
    (b) **reference-anchored beats** — beat boundaries are detected on the
    label trace only, and both prediction and label are read inside those
    same intervals, so every reference beat yields exactly one comparison
    and agreement statistics carry no selection bias from a prediction whose
    own beats are hard to find; a separate `detection_quality` (sensitivity/
    PPV/IBI error, gated by an autocorrelation pulsatility test) is kept
    apart on purpose, so a model that finds few beats can't look accurate on
    only the ones it found; (c) **uniform `max`/`mean`/`min` beat statistics
    with per-signal display labels** — the computation is identical across
    absolute-class signals, only `neural_methods/signals.beat_labels` maps
    the triple onto the signal's own vocabulary (systolic/MAP/diastolic for
    ABP, peak/mean/trough for CVP), so a new absolute signal needs a label
    map, not a new metric; (d) **which families apply to a signal follows
    from its class** (`FAMILIES` in `evaluation/scoring/__init__.py`), never
    from a config key, so `TEST.METRICS` is deleted outright — replaced by
    `TEST.REPORT.BOOTSTRAP` / `TEST.REPORT.PLOTS`, which gate cost
    (bootstrap resamples, which figures) but never content. Every clinical
    threshold in `evaluation/scoring/standards.py` is marked `UNVERIFIED`
    with a named source and ships with a printed provenance line (including
    the study-design requirements a metrics report can note but never
    satisfy); flipping `VERIFIED_AGAINST_STANDARD_TEXT` waits on a
    line-by-line check against the purchased ISO 81060-3:2022 and
    IEEE 1708-2014/1708a-2019 texts (spec §17) — this phase is the
    computable machinery, not a verified clinical grade. Verified end to end
    on a real smoke run (`NECKFLIX_PHYSMAMBA_SMOKE`, the 332-store local
    zarr cache, held-out participant 015); suite at 288, green.

17. **Contract v2 (2026-09-01).** `revised_overhaul_plan.md` supersedes
    `updating_plan.md`; the remaining phases were re-planned around
    [the contract-v2 design](2026-09-01-contract-v2-design.md). The
    decisions, settled interactively: **(a)** cache contract v2 — fixed
    modality (`gr`/`rgb`/`ir`/`depth`/`t`/`ev`) and trace
    (`ecg`/`abp`/`cvp`/`ppg`/`rr`) vocabularies, per-modality
    `timestamps_us`, `video/data` replacing `video/frames`,
    perspective-level nominal `fps` with first-frame alignment under
    `1/fps`, a required `units` attr per trace, root attrs slimmed to
    `participant` plus free filter attrs. **(b)** `complete`/`tool_version`
    admission deleted: stores are assumed complete, and the offline
    validator (`tools/validate_cache.py`) is the only admission mechanism.
    **(c)** One dataset class: the modality→channel table is global, so
    per-dataset `channel_map` subclasses go. **(d)** Model contract v2 à la
    CardioHydra (github.com/coenarrow/CardioHydra): `forward(batch) ->
    batch` with `raw_losses` (unweighted, model-written) and `losses` (weighted, trainer-written) riding the batch, computed inside the model so weights stay calibratable — the
    per-signal machinery invoked from the `DictModel` base so simple models
    inherit it, composite models (PhysHydra) adding stage entries;
    `Reads:/Modifies:` docstrings adopted. **(e)** **Style C** — S full
    parallel copies of the original architecture, input widened to the
    demanded channels (mostly 5) — is the default (`HEAD_STYLE: parallel`)
    for *all* migrations; A/B remain options; DeepPhys/PhysFormer/PhysMamba
    are reworked before further migrations. **(f)** IEEE 1708 is kept
    (`UNVERIFIED` until its text is sourced): it is the cuffless-device
    standard, its A–D grading is the research-progress dial, and it is the
    number cuffless-BP papers quote. The ISO texts are in
    `standards/ISO-81060/`, so ISO verification is unblocked.

Last updated: 2026-09-01
