# remote-physiology (fork of rPPG-Toolbox)

## Project Mission

Camera-based estimation of physiological signals — not just heart rate. Every
model predicts **multiple signals at once**: BVP, arterial and central venous
pressure waveforms (ABP, CVP), respiration, and others as they appear. Primary
dataset: the multimodal Neckflix dataset (RGB/IR/Depth video + ABP/CVP/ECG
traces). Validation includes clinical BP standards (IEEE 1708, ISO 81060,
ESH 2023) and large-scale LOSO sweeps on an HPC cluster.

## Design Principle

**Extending the repo should be cheap, because everything shared is written
once.** A new dataset is a `channel_map` subclass plus a markdown cache spec;
a new model is a backbone `nn.Module`, a config class and builder in
`src/models.py`, a YAML in `configs/models/` and one smoke test
(`docs/adding_a_model.md` is the recipe) — never a new trainer, loader, loss
module, or plot set. When new work needs something a
shared piece almost does, extend the shared piece for everyone rather than
writing a parallel copy beside it; a second implementation of anything is a
bug in the first one's design. This principle is why the per-model and
per-dataset recipes below are short — keep them that way.

## Overhaul In Progress

The repo is mid-overhaul from the upstream single-signal design to the
multi-signal contract.

## Cross-Cutting Rules

- **Dependencies go through `uv add`, never pip.** `pyproject.toml` + `uv.lock`
  are the single source of truth (`requirements.txt` and `setup.sh` are gone).
  Every addition considers all three platforms: Windows dev, Linux HPC, macOS.
- **All tensor reshaping uses einops** (`rearrange` / `reduce` / `einsum`),
  not `view` / `permute` / `reshape` — including migrated model code.
- **Tests are a cost, not a safety net.** This is research code; the suite
  exists to catch a broken build, not to specify behaviour. Write a test only
  when it is (a) the one build-and-forward smoke test a model migration
  requires, or (b) a unit test of a pure function that fits in a dozen lines
  with no fixtures. Never add tests to a refactor or a design change, never
  test a test helper, and never add a test "for coverage". The verification
  for a change is the run command in `README.md`, not a new test. This
  overrides the test-driven-development and verification skills' defaults.
- **Stale tests are deleted in the same change that stales them.** A test
  that imports a module, name, or fixture argument that no longer exists, or
  that asserts behaviour the current design has replaced, is deleted, not
  fixed, skipped, marked xfail, or left failing. The design is the spec; the
  old tests are not. After a design change, run only the tests that cover
  the files you touched. If a test elsewhere fails, the default is that the
  test is stale: delete it and name it in the summary so a human can veto,
  rather than adapting the code back to it. A collection error from
  `uv run pytest --collect-only -q` names a file to delete, not a file to
  fix.
- **Legacy code is deleted, not adapted.** No compatibility shims; git history
  and the `pre-overhaul` tag are the archive.
- **Every model accepts any frame size and any window length.** Structural
  constants of the upstream code (patch sizes, sequence lengths, fixed token
  grids) become constructor arguments derived from the interface, config
  switches, or adaptive stages around the published network that are exact
  no-ops at the paper's shape — never hard-coded refusals, and never a silent
  crop or truncation. The only refusal left is a frame the stem pools to
  nothing, named by the builder.
- **`configs/interfaces/<name>_interface.yaml` is the paper.** That directory
  is what the per-model interfaces are for: when migrating a model, its
  `<name>_interface.yaml` (the architecture's `NAME` lowercased) *is* the
  rPPG-Toolbox configuration of it — rate, window, resize, input
  preprocessing, the single PPG trace the paper predicts, its label
  preprocessing and its loss. Nothing Neckflix-specific goes in it. Its
  twin, `configs/training/<name>_training.yaml`, *is* the rPPG-Toolbox
  training recipe of the model: epochs, batch size, optimiser, rate, decay,
  schedule, precision, read off the upstream `train_configs/` file *and*
  the upstream trainer class (the optimiser and schedule live there, not in
  the YAML). Nothing in code declares or checks the paper setup; the files
  do. Model comparisons run every model on the same standard interface; the
  paper files are where a migration is checked against the paper.
- **A finished migration ends with a command in `README.md`.** When a model
  migrated from rPPG-Toolbox is done, add under "Algorithms" the exact
  `scripts/train.py` command that trains it on the PURE dataset
  (`--datasets pure`) with only the first participant held out
  (`--test-participant-dataset pure --test-participant-id 01`), on its paper
  interface and paper training recipe (inference and evaluation are the
  same two commands for every model, on the run directory it prints).
  That command is the migration's proof of life; a model without one is
  not finished.
