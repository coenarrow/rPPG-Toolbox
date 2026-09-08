# Model migrations onto the multi-signal contract

**Goal:** put every remaining upstream rPPG-Toolbox model in
`neural_methods/model/` on the multi-signal contract, one sub-agent per
model, so that each one trains and tests on PURE through `run_experiment.py`
on its own paper interface and paper recipe. PhysHydra is excluded.

**Spec:** `docs/adding_a_model.md` is the recipe and the authority; `CLAUDE.md`
carries the cross-cutting rules. DeepPhys (per-frame), PhysMamba and
PhysFormer (clip) are the finished exemplars; `neural_methods/model/_template.py`
is the skeleton.

**Models, in execution order:** PhysNet, TS-CAN, EfficientPhys, iBVPNet,
FactorizePhys, RhythmFormer, BigSmall. Then one docs sweep.

## Global Constraints

Binding for every task. Copied from `CLAUDE.md` and `docs/adding_a_model.md`.

1. **Extending the repo is cheap because everything shared is written once.**
   A migration touches: the backbone module in `neural_methods/model/`, a
   config class + builder + two registry lines in `src/models.py`,
   `configs/models/<name>.yaml`, `configs/interfaces/<name>_interface.yaml`,
   `configs/training/<name>_training.yaml`, `tests/test_<name>.py`, and the
   README "Algorithms" section. Never a new trainer, loader, loss module or
   plot set. If a shared piece almost does what is needed, extend the shared
   piece for every model; a second implementation of anything is a bug.
2. **All tensor reshaping uses einops** (`rearrange` / `reduce` / `einsum`),
   never `view` / `permute` / `reshape` / `flatten` — including code migrated
   from upstream.
3. **Testing stays minimal.** One smoke test per migration
   (`tests/test_<name>.py`, the build-and-forward test from
   `docs/adding_a_model.md` step 5), and nothing more.
4. **Legacy code is deleted, not adapted.** No compatibility shims. Git
   history and the `pre-overhaul` tag are the archive.
5. **Every model accepts any frame size and any window length.** Structural
   constants of the upstream code (patch sizes, sequence lengths, `frames=`,
   `img_size` if-chains, fixed token grids, temporal strides) become
   constructor arguments derived from the interface by the builder, config
   switches the paper ablates, or adaptive stages around the published network
   that are exact no-ops at the paper's shape. Never a hard-coded refusal,
   never a silent crop or truncation. `temporal_divisor` / `temporal_length`
   are interim only and a task that leaves one in place must say so in its
   report. The only refusal that remains is a frame the stem pools to nothing,
   raised by the builder via `_require_min_frame`.
6. **`configs/interfaces/<name>_interface.yaml` is the paper**: the upstream
   `configs/train_configs/PURE_PURE_UBFC-rPPG_<MODEL>*.yaml` at the
   `pre-overhaul` tag (read it with `git show pre-overhaul:<path>`), expressed
   as FS, WINDOW_SECONDS (= CHUNK_LENGTH / FS to six decimals), RESIZE,
   INPUT_PREPROCESSING (= DATA_TYPE in order), TRACES `[PPG]`,
   LABEL_PREPROCESSING (nearest of raw / zscore), LOSS (the published
   criterion; a term the loss module lacks is noted in a comment, never
   substituted). Nothing Neckflix-specific. Its twin
   `configs/training/<name>_training.yaml` is the recipe: EPOCHS, BATCH_SIZE,
   LR from the upstream TRAIN block; OPTIMIZER, WEIGHT_DECAY, SCHEDULER from
   the `optim.*` / `lr_scheduler.*` calls in the legacy
   `neural_methods/trainer/<Name>Trainer.py`; PRECISION float32 unless the
   trainer autocasts. Every key of both files is required; copy the
   `_interface_template.yaml` / `_training_template.yaml` comment style used
   by the DeepPhys and PhysMamba files.
7. **A finished migration ends with a command in `README.md`** under
   "Algorithms": the model added to the "On the multi-signal contract today"
   line and the exact `run_experiment.py` command that trains and tests it on
   PURE with participant `01` held out on its paper interface and recipe, in
   the same shape as the DeepPhys block already there.
8. **Dependencies go through `uv add`, never pip.** No migration here should
   need one.
9. The backbone contract (`docs/adding_a_model.md` step 1): constructor takes
   `in_channels` (required, default 3) plus published sizes as defaults; no
   `params` argument, no loss, no `get_config`, no device handling, no input
   normalisation the dataset already does; `forward` is exactly
   `(N, C_in, H, W) -> (N, 1)` (per_frame) or `(B, C_in, T, H, W) -> (B, 1, T)`
   (clip); `output_layers()` returns the activation-free readout(s), each with
   a one-element bias; nothing about traces or channels by name.

## Process rules for every task

- Work in `c:\Users\20759193\source\repos\remote-physiology` on branch
  `migrate-models`. Run Python only as `uv run python` / `uv run pytest`
  (bare `python` lacks the deps). `pytest` prints a
  "Windows fatal exception 0xc0000139" stack dump at import from the
  causal_conv1d DLL; it is pre-existing noise, not a failure.
  `tests/test_config_keys.py` fails before this work starts (it imports a
  deleted legacy module) and is out of scope.
- Do not grep or list the whole repo root: `.venv`, `.uv_cache` and `runs`
  make it time out. Scope searches to `neural_methods`, `src`, `configs`,
  `tests`, `docs`, `README.md`.
- The PURE cache is at `D:/pure_zarr` (see `configs/datasets/pure.yaml`);
  the box has a CUDA GPU. The smoke run for a model is
  ```
  uv run python run_experiment.py --datasets pure \
      --test-participant-dataset pure --test-participant-id 01 \
      --model <name> --interface configs/interfaces/<name>_interface.yaml \
      --training configs/training/<name>_training.yaml --limit-windows 8
  ```
  It must reach `evaluate` and write `runs/<name>_pure_01/` with
  `config.yaml`, `losses.csv`, `test_records.pt`, `rates.csv`, `summary.csv`
  and `digest.txt`. Paste the tail of its output and the digest into the
  report. Then also build the model on the standard interface
  (`configs/interfaces/interface_neckflix.yaml`) and push one synthetic batch
  through it in a Python one-liner (window and frame size from that file) to
  show the adaptive stages work; paste that too. Remove `runs/<name>_pure_01`
  afterwards.
- **Committing:** the working tree carries a large uncommitted overhaul that
  is not yours. Commit with `git add <explicit paths>` naming only the files
  this task creates, edits or deletes; never `git add -A`, `git add .` or
  `git commit -a`. One commit per task is fine. Use
  `git rm` for deletions. End the commit message with
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- **Legacy trainer:** once the training YAML has transcribed the optimiser
  and schedule, delete the legacy `neural_methods/trainer/<Name>Trainer.py`
  named in the task with `git rm` (the YAML's header comment cites it by
  name and the `pre-overhaul` tag, as the PhysMamba recipe does). Do not touch
  `BaseTrainer.py`, `MultiSignalTrainer.py` or `PhysHydraTrainer.py`.
- Config helpers come from `src.config` (`ConfigError`, `build`, `load_yaml`),
  which `src/models.py` already imports. The top-level `config.py` is gone;
  never write `from config import ...`.
- Edit the upstream module in place; do not write a new module beside it.
  Delete its `if __name__ == "__main__"` blocks, debug flags, `print`s and
  dead imports.

## Task 1: PhysNet

**Files:** `neural_methods/model/PhysNet.py`, `src/models.py`,
`configs/models/physnet.yaml`, `configs/interfaces/physnet_interface.yaml`,
`configs/training/physnet_training.yaml`, `tests/test_physnet.py`,
`README.md`; delete `neural_methods/trainer/PhysnetTrainer.py`.

Read `docs/adding_a_model.md` in full first, then `neural_methods/model/PhysMamba.py`
and `_build_physmamba` in `src/models.py`: PhysNet is the same shape of
migration (a clip backbone with a 3D-conv stem, temporal pooling by 4 and two
2x transposed-conv upsamples).

Upstream sources: `git show pre-overhaul:configs/train_configs/PURE_PURE_UBFC-rPPG_PHYSNET_BASIC.yaml`
and `neural_methods/trainer/PhysnetTrainer.py` (its `optim.Adam(...)` and
`OneCycleLR(...)` calls).

Specifics:

- Rename the class to `PhysNet` (module `PhysNet.py`, `NAME: PhysNet`,
  config `physnet.yaml`). Keep the layers exactly as published.
- `in_channels` widens the first conv. `frames` goes away: the final
  `AdaptiveAvgPool3d((frames, 1, 1))` becomes `(None, 1, 1)`, which is the
  identity in time.
- Any window length: average-pool the stem output in time to the nearest
  multiple of 4 before the trunk and linearly interpolate the prediction back
  to the window length after it, skipped when the window already divides by
  4 (copy the PhysMamba pattern and docstring wording). Any frame size: the
  spatial pools reduce to a minimum frame; expose `MIN_FRAME` on the module
  and have the builder call `_require_min_frame` like PhysMamba does.
- Output `(B, 1, T)` (the readout is `ConvBlock10`, a `Conv3d(64, 1, ...)`;
  after the spatial pool `rearrange` to `b 1 t`). `output_layers()` returns it.
- Drop `pdb`, `math`, `_triple` if unused; einops for every reshape.
- Config class `PhysNetConfig` with `NAME` and `INPUT` (the stem's
  preprocessing block), builder `_build_physnet`, two registry lines, import
  at the top of `src/models.py` beside the PhysMamba one.
- Interface, recipe, smoke test, README block, smoke run, standard-interface
  forward: per the Global Constraints and Process rules.

## Task 2: TS-CAN

**Files:** `neural_methods/model/TS_CAN.py`, `src/models.py`,
`configs/models/tscan.yaml`, `configs/interfaces/tscan_interface.yaml`,
`configs/training/tscan_training.yaml`, `tests/test_tscan.py`, `README.md`;
delete `neural_methods/trainer/TscanTrainer.py`.

Read `docs/adding_a_model.md` in full first, then `neural_methods/model/DeepPhys.py`
and `_build_deepphys` in `src/models.py`: TS-CAN is DeepPhys plus temporal
shift modules, and the DeepPhys migration shows how the two-branch input
(`MOTION_INPUT`, `APPEARANCE_INPUT`) and the frame-derived dense width are
handled.

Upstream sources: `git show pre-overhaul:configs/train_configs/PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml`
(note `MODEL.TSCAN.FRAME_DEPTH` there) and `neural_methods/trainer/TscanTrainer.py`.

Specifics and rulings:

- Delete the `MTTS_CAN` class (multi-task variant; the wrapper makes copies)
  and `Attention_mask.get_config`. Class `TSCAN`, `NAME: TSCAN`.
- **Ruling: TS-CAN takes clips (`per_frame=False`).** The temporal shift
  needs to know where each clip starts and ends; folded `(b t)` rows cannot
  tell it. The module receives `(B, C_in, T, H, W)`, splits the motion and
  appearance blocks off the channel axis (motion first, appearance second,
  as DeepPhys does), folds to `(b t)` itself with einops, runs the published
  2D network, and returns `(B, 1, T)`.
- **Ruling: the temporal shift is adaptive within the clip.** `TSM` shifts
  over segments of `frame_depth` consecutive frames of the same clip. When
  `T` is not a multiple of `frame_depth`, the trailing partial segment of
  each clip is shifted as its own shorter segment (TSM zero-pads at segment
  ends, so a short segment is well defined). When `T % frame_depth == 0`
  the computation is exactly the published one.
- **Ruling: `FRAME_DEPTH` is a config key** (`TSCANConfig`: `NAME`,
  `MOTION_INPUT`, `APPEARANCE_INPUT`, `FRAME_DEPTH`), because the upstream
  YAML exposes it and the PURE recipe sets it to a value that differs from
  the class default; `validate` requires it positive. `tscan.yaml` carries
  the PURE recipe's value.
- The `img_size` if-chain (36/72/96/128) becomes a dense width derived from
  the frame the builder passes (do what DeepPhys does; if you can derive it
  for non-square frames without changing the paper path, do so, otherwise
  mirror DeepPhys's square-frame refusal).
- einops for every `view`; drop `params`; `output_layers()` returns the final
  `nn.Linear(nb_dense, 1)`.
- Interface, recipe, smoke test, README block, smoke run, standard-interface
  forward: per the Global Constraints and Process rules.

## Task 3: EfficientPhys

**Files:** `neural_methods/model/EfficientPhys.py`, `src/models.py`,
`configs/models/efficientphys.yaml`, `configs/interfaces/efficientphys_interface.yaml`,
`configs/training/efficientphys_training.yaml`, `tests/test_efficientphys.py`,
`README.md`; delete `neural_methods/trainer/EfficientPhysTrainer.py`.

Read `docs/adding_a_model.md` in full first, then the migrated
`neural_methods/model/TS_CAN.py` and `_build_tscan` in `src/models.py`
(Task 2 finished them): EfficientPhys is the single-branch sibling of TS-CAN
and reuses the same clip-shaped, in-clip-adaptive temporal shift.

Upstream sources: `git show pre-overhaul:configs/train_configs/PURE_PURE_UBFC-rPPG_EFFICIENTPHYS.yaml`
and `neural_methods/trainer/EfficientPhysTrainer.py`. Note the upstream
trainer appends a copy of the last frame to each chunk before the forward so
that the in-network `torch.diff` yields one output per frame.

Specifics and rulings:

- **Ruling: the in-network frame difference stays.** It is the paper's point
  (end-to-end from standardised frames), not dataset preprocessing. With clip
  input, `diff` runs along the time axis of each clip and a zero frame is
  appended so the output has `T` rows; that is numerically what upstream's
  duplicated last frame produced. `BatchNorm2d(3)` becomes
  `BatchNorm2d(in_channels)`.
- **Ruling:** clips in (`per_frame=False`), fold inside, adaptive in-clip
  TSM and `FRAME_DEPTH` config key exactly as TS-CAN; share the `TSM` module
  rather than keeping a second copy (import it from `TS_CAN.py`, or move it
  to a small shared module under `neural_methods/model/` if that reads
  better; one implementation either way).
- Config `EfficientPhysConfig`: `NAME`, `INPUT`, `FRAME_DEPTH`. Drop the
  `channel` argument and `get_config`. Derive the dense width from the frame
  as TS-CAN does.
- Interface, recipe, smoke test, README block, smoke run, standard-interface
  forward: per the Global Constraints and Process rules.

## Task 4: iBVPNet

**Files:** `neural_methods/model/iBVPNet.py`, `src/models.py`,
`configs/models/ibvpnet.yaml`, `configs/interfaces/ibvpnet_interface.yaml`,
`configs/training/ibvpnet_training.yaml`, `tests/test_ibvpnet.py`,
`README.md`; delete `neural_methods/trainer/iBVPNetTrainer.py`.

Read `docs/adding_a_model.md` in full first, then the migrated
`neural_methods/model/PhysNet.py` (Task 1) for the clip-backbone pattern.

Upstream sources: `git show pre-overhaul:configs/train_configs/PURE_PURE_UBFC-rPPG_iBVPNet_BASIC.yaml`
(note `MODEL.iBVPNet.*` and the chunk length) and
`neural_methods/trainer/iBVPNetTrainer.py`.

Specifics and rulings:

- **Ruling: one `InstanceNorm3d(in_channels)`** replaces the 1/3/4-channel
  branching. Instance norm is per (sample, channel), so a single norm over
  every channel is numerically identical to upstream's split RGB / thermal
  norms; the `print`s and `assert`s go.
- **Ruling: the in-network `torch.diff` along time stays** (as for
  EfficientPhys); append a zero frame after it so the output has `T` rows,
  replacing the `view(-1, length - 1)`. State this in the docstring.
- `frames` constructor argument goes; any temporal pooling / upsampling
  divisor gets the PhysNet adaptive stage; spatial minimum via `MIN_FRAME`
  and `_require_min_frame`.
- Output `(B, 1, T)`; `output_layers()` returns the final `Conv3d(nf[2], 1, ...)`.
  Delete the `__main__` block and `debug` plumbing. einops throughout.
- Config `iBVPNetConfig`: `NAME`, `INPUT`. `NAME: iBVPNet`, config file
  `ibvpnet.yaml`, interface `ibvpnet_interface.yaml`.
- Interface, recipe, smoke test, README block, smoke run, standard-interface
  forward: per the Global Constraints and Process rules.

## Task 5: FactorizePhys

**Files:** `neural_methods/model/FactorizePhys/FactorizePhys.py`,
`neural_methods/model/FactorizePhys/FSAM.py`, `src/models.py`,
`configs/models/factorizephys.yaml`, `configs/interfaces/factorizephys_interface.yaml`,
`configs/training/factorizephys_training.yaml`, `tests/test_factorizephys.py`,
`README.md`; delete `neural_methods/model/FactorizePhys/FactorizePhysBig.py`,
`neural_methods/model/FactorizePhys/test_FactorizePhys.py`,
`neural_methods/model/FactorizePhys/test_FactorizePhysBig.py` and
`neural_methods/trainer/FactorizePhysTrainer.py`.

Read `docs/adding_a_model.md` in full first, then the migrated
`neural_methods/model/iBVPNet.py` (Task 4): FactorizePhys shares its stem
conventions (instance norm, in-network diff, `frames` argument).

Upstream sources: `git show pre-overhaul:configs/train_configs/` has no PURE
FactorizePhys training file; use
`git show pre-overhaul:configs/infer_configs/PURE_UBFC-rPPG_FactorizePhys_FSAM_Res.yaml`
for the data side (`MODEL.FactorizePhys.*`, chunk length, resize, data
types) and `git show pre-overhaul:configs/train_configs/` for the closest
FactorizePhys training file at that tag (list the directory; pick the one
that trains on PURE or, failing that, UBFC-rPPG, and say which in the YAML
comment), plus `neural_methods/trainer/FactorizePhysTrainer.py` for the
optimiser and schedule.

Specifics and rulings:

- Keep the FSAM path of the paper (`FSAM_Res`): NMF factorisation with
  residual. **Ruling:** the `md_config` dict collapses to constructor
  defaults at the paper's values; the only config key besides `NAME` and
  `INPUT` is `FSAM: true|false`, the ablation the paper reports. The
  `device` argument goes (modules follow their parameters).
- Delete `FactorizePhysBig` and the two in-tree test scripts: the migrated
  module accepts any frame size, so the "Big" variant is redundant.
- Instance norm, in-network diff and `frames` removal exactly as iBVPNet.
  The BVP head's `Conv3d(..., padding=(1, 0, 0))` spatial reduction assumes
  a particular feature map size; make it adaptive (a spatial pool to what
  the head expects, identity at the paper's frame) rather than a refusal.
- `FSAM.py` is the same code under the same rules: einops for every
  `view` / `reshape` / `permute`, no `print`, no debug flags.
- Output `(B, 1, T)`; `output_layers()` returns the head's final 1-channel
  conv. Config `FactorizePhysConfig`; `NAME: FactorizePhys`; file
  `factorizephys.yaml`.
- Interface, recipe, smoke test, README block, smoke run, standard-interface
  forward: per the Global Constraints and Process rules.

## Task 6: RhythmFormer

**Files:** `neural_methods/model/RhythmFormer.py`, `src/models.py`,
`configs/models/rhythmformer.yaml`, `configs/interfaces/rhythmformer_interface.yaml`,
`configs/training/rhythmformer_training.yaml`, `tests/test_rhythmformer.py`,
`README.md`; delete `neural_methods/trainer/RhythmFormerTrainer.py` and
`neural_methods/loss/RythmFormerLossComputer.py` (check nothing else imports
the latter; scope the grep to `neural_methods`, `src`, `tests`,
`unsupervised_methods`).

Read `docs/adding_a_model.md` in full first, then `neural_methods/model/PhysFormer.py`
and `_build_physformer` in `src/models.py`: PhysFormer is the closest
finished migration (a transformer over a conv stem with a token grid read off
the stem output and a temporal stride wrapped in an adaptive stage).

Upstream sources: `git show pre-overhaul:configs/train_configs/PURE_PURE_UBFC-rPPG_RHYTHMFORMER_BASIC.yaml`
and `neural_methods/trainer/RhythmFormerTrainer.py` (optimiser, schedule, and
the loss it builds from `RythmFormerLossComputer`).

Specifics and rulings:

- `Fusion_Stem`: the hard-coded `12` is four frame differences times three
  channels; it becomes `4 * in_channels`. The `view(N, D, 64, H // 4, W // 4)`
  token grid is read off the stem output's actual shape with einops.
- Temporal patching / pooling inside `TPT_Block` (`t_patch`, the stride-2
  time convs) gives a temporal divisor: wrap it in the PhysFormer/PhysMamba
  adaptive stage (pool time to a multiple before the trunk, interpolate back
  after), identity at the paper's window. Spatial: the BRA region sizes
  must divide the token grid; make the region partition adaptive (pad or
  pool the token grid to a whole number of regions, identity at the paper's
  frame) and expose `MIN_FRAME` for `_require_min_frame`.
- **Ruling on the loss:** the interface `LOSS` is `{NEGPEARSON: 1.0}` with
  a comment that upstream adds frequency-domain cross-entropy and
  label-distribution terms the loss module does not carry, in the same wording
  as `physformer_interface.yaml`. Do not port the loss computer.
- einops for every `view` / `reshape` / `permute` / `flatten`, including the
  vendored BRA helpers. Duplicate imports at the top go.
- Output `(B, 1, T)`; `output_layers()` returns `ConvBlockLast`
  (`Conv1d(embed_dim[-1], 1, 1)`). Config `RhythmFormerConfig`: `NAME`,
  `INPUT`.
- Interface, recipe, smoke test, README block, smoke run, standard-interface
  forward: per the Global Constraints and Process rules.

## Task 7: BigSmall

**Files:** `neural_methods/model/BigSmall.py`, `src/models.py`,
`configs/models/bigsmall.yaml`, `configs/interfaces/bigsmall_interface.yaml`,
`configs/training/bigsmall_training.yaml`, `tests/test_bigsmall.py`,
`README.md`; delete `neural_methods/trainer/BigSmallTrainer.py`.

Read `docs/adding_a_model.md` in full first, then the migrated
`neural_methods/model/TS_CAN.py` and `_build_tscan` (Task 2): BigSmall's
WTSM is the same in-clip temporal shift with `frame_depth` 3, and its two
branches read two preprocessing blocks the way TS-CAN's do.

Upstream sources: `git show pre-overhaul:configs/train_configs/BP4D_BP4D_BIGSMALL_FOLD1.yaml`
(there is no PURE recipe: BigSmall was published on BP4D+; say so in both
YAML headers) and `neural_methods/trainer/BigSmallTrainer.py`.

Specifics and rulings:

- **Ruling: BigSmall migrates as a single-trace backbone.** Keep the big
  and small branches and the BVP head; delete the AU and respiration heads.
  The wrapper makes one copy per trace, and RESP is a trace like any other
  on the standard interface.
- **Ruling: the small branch is derived in-module.** Upstream preprocessed
  two resolutions (big 144x144 Standardized, small 9x9 DiffNormalized). Here
  the interface has one `RESIZE`; the small branch average-pools its
  DiffNormalized block down to the published small size (a constructor
  default, 9), which at 144 px is an exact 16x16 mean. Note in the docstring
  that this pools after preprocessing where upstream resized before it.
  Config `BigSmallConfig`: `NAME`, `BIG_INPUT`, `SMALL_INPUT`, `FRAME_DEPTH`
  (upstream `MODEL.BIGSMALL.FRAME_DEPTH`), the two inputs distinct.
- **Ruling on the paper interface's window.** Upstream chunks to 3 frames
  only as a batching device for the 3-frame WTSM and then stitches every
  chunk back into the whole video before scoring; strided windows do not
  stitch, so a 3-frame window here would score nothing. The interface states
  FS 25 and the toolbox's standard 180-frame chunk (WINDOW_SECONDS 7.2) with
  this reasoning in the comment; the in-clip WTSM makes any length exact
  when divisible by 3 and adaptive otherwise.
- Clips in (`per_frame=False`), fold inside, in-clip adaptive WTSM shared
  with TS-CAN's `TSM` if it is the same shift (it is `n_segment=3` with a
  different name; use one implementation). Dense width derived from the
  frame. einops throughout; drop `params`.
- `output_layers()` returns the BVP readout (`nn.Linear(..., 1)`).
- Interface, recipe, smoke test, README block, smoke run, standard-interface
  forward: per the Global Constraints and Process rules.

## Task 8: docs sweep

**Files:** `docs/adding_a_model.md`, `README.md`.

After Tasks 1 to 7, bring the guide up to date with what now exists:

- The intro's list of models on the contract and the "not yet registered"
  list in "Migrating an upstream rPPG-Toolbox model" (only PhysHydra remains
  unmigrated, and it is out of scope, not pending).
- The `per_frame` table in step 1: TS-CAN, EfficientPhys and BigSmall take
  clips and fold inside so their temporal shift can be adaptive; only
  DeepPhys is per-frame. Say why in one sentence.
- The "three paper interfaces today" table in step 4 gains one row per
  migrated model (frames, window, input, loss, recipe), read off the YAMLs.
- The training-recipe table's "`optim.*` call in
  `neural_methods/trainer/<Name>Trainer.py`" now points at the `pre-overhaul`
  tag, since the legacy trainers are deleted.
- The `TemplateNet` paragraph on `temporal_divisor` / `temporal_length`
  stays only if some migrated module still declares one; otherwise say the
  trainer still honours them for a model that is mid-migration.
- README "Algorithms": confirm every migrated model has its command block,
  the "On the multi-signal contract today" line lists all of them, and the
  paragraph after the commands still reads true.

No code changes in this task. Commit `docs/adding_a_model.md` and
`README.md` only.
