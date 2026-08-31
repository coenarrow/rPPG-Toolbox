# Architecture Overview

## System Overview

This repository is a fork of the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox), originally designed for camera-based remote photoplethysmography (rPPG) -- estimating heart rate from facial video by detecting subtle skin color changes caused by blood flow.

Our extension adds support for **cardiovascular pressure waveform estimation** (Arterial Blood Pressure / ABP, Central Venous Pressure / CVP) using the multimodal **Neckflix dataset**. This shifts the output from a single heart rate value to full waveform prediction, and the input from RGB-only facial video to multimodal camera streams (RGB, infrared, depth) captured at the neck region.

### Key References

- Original paper: [rPPG-Toolbox: Deep Remote PPG Toolbox](https://arxiv.org/abs/2210.00716)
- Original repository: [ubicomplab/rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)


## Data Flow

The cache is an *external input*: zarr stores written by a dataset's
preprocessor (for Neckflix, the `ghcr.io/coenarrow/neckflix` container),
which this repo reads and never writes. There is no in-repo preprocessing
step, no `.npy` cache, and no file-list CSVs; the legacy pipeline that did
all of that lives at the `pre-overhaul` tag.

```
External preprocessor (per dataset; Neckflix: ghcr.io/coenarrow/neckflix)
    |
    v
Zarr cache: one *.zarr store per recording (raw frames + traces)
    |
    v
BaseZarrDataset subclass -- metadata-only construction, lazy per-window reads
    - root-attr include/exclude filters (the LOSO mechanism)
    - strided or random windows over frames
    |
    v
Batch dict {frames, labels, label_stats, channel_mask, label_mask, metadata}
    |
    v
DictModel  -- frame transform (resize + DATA_TYPE) then the architecture
    |         returns the same dict plus "predictions": {signal: (B, T)}
    v
PerSignalLoss / per-signal evaluation
```

See "The Zarr Cache Contract" and "The Batch-Dict Contract" below.


## Directory Layout

```
remote-physiology/
|-- config.py                  Central configuration management (yacs; Phase 5
|                              replaces it with the typed pattern)
|-- main.py                    The entry point (zarr pipeline)
|
|-- dataset/
|   |-- BP4D_BigSmall_Subject_Splits/  BigSmall 3-fold membership CSVs
|   |-- data_loader/
|       |-- zarr_dataset.py    BaseZarrDataset: lazy loader over zarr caches
|       |-- NeckflixLoader.py  NeckflixDataset (channel map only)
|       |-- neckflix_config.py yacs config -> zarr loader plain-dict config
|       |-- label_transforms.py Per-window label normalisation + inverses
|       |-- <Dataset>.md       Cache specs: how to build each legacy
|                              dataset's zarr stores (PURE, MMPD, BP4D+, ...)
|
|-- neural_methods/
|   |-- batch.py               The Neckflix batch-dict contract (keys + einops moves)
|   |-- frame_transforms.py    Consumer-side DATA_TYPE + resize on (B,C,T,H,W)
|   |-- signals.py             Canonical signal/channel registries
|   |
|   |-- model/                 Neural network architectures (one file per model)
|   |   |-- DictModel.py       Base: batch dict in, batch dict + predictions out
|   |   |-- SignalDictWrapper.py  Adapter for 2-D/3-D backbones
|   |   |-- mamba_compat.py    mamba_ssm shim + pure-torch MambaRef fallback
|   |   |-- DeepPhys.py
|   |   |-- EfficientPhys.py
|   |   |-- TS_CAN.py
|   |   |-- PhysNet.py
|   |   |-- PhysMamba.py
|   |   |-- PhysFormer.py
|   |   |-- RhythmFormer.py
|   |   |-- BigSmall.py
|   |   |-- iBVPNet.py
|   |   |-- FactorizePhys/     (directory-based model with submodules)
|   |   |-- PhysHydra.py       Pressure estimation model
|   |
|   |-- trainer/               Training/validation/testing routines (one per model)
|   |   |-- BaseTrainer.py     Shared DDP setup, rank management, model unwrapping
|   |   |-- MultiSignalTrainer.py  One trainer for every dict-contract model
|   |   |-- PhysMambaTrainer.py
|   |   |-- PhysHydraTrainer.py
|   |   |-- ...
|   |
|   |-- loss/                  Loss function implementations
|       |-- NegPearsonLoss.py
|       |-- PhysHydraLoss.py
|       |-- PhysNetNegPearsonLoss.py
|       |-- PerSignalLoss.py
|       |-- RythmFormerLossComputer.py
|
|-- evaluation/
|   |-- metrics.py             Metric computation (MAE, RMSE, MAPE, Pearson, SNR, BA)
|   |-- post_process.py        Signal post-processing (FFT, peak detection)
|   |-- metrics_report.py      Shared HR-metric aggregation and printing
|   |-- BlandAltmanPy.py       Bland-Altman agreement analysis and plotting
|   |-- bigsmall_multitask_metrics.py
|
|-- unsupervised_methods/      Traditional signal processing methods
|   |-- methods/
|   |   |-- CHROME_DEHAAN.py
|   |   |-- GREEN.py
|   |   |-- ICA_POH.py
|   |   |-- LGI.py
|   |   |-- OMIT.py
|   |   |-- PBV.py
|   |   |-- POS_WANG.py
|   |-- unsupervised_predictor.py
|   |-- utils.py
|
|-- configs/
|   |-- neckflix/              Current-format experiment configs
|   |-- train_configs/         Legacy configs (die in Phase 5)
|   |-- infer_configs/         Legacy configs (die in Phase 5)
|-- physhydra_configs/         Legacy PhysHydra configs (die in Phase 5)
|
|-- .slurm_scripts/            SLURM reference templates (copy and adapt)
|-- logs/                      SLURM job stdout/stderr files (gitignored)
|-- runs/                      Training run outputs, checkpoints, plots (gitignored)
|-- tools/                     LOSO fold listing, output summarising, mamba vendoring
|-- vendor/mamba-ssm           Patched mamba-ssm for the Windows build
```


## Configuration System

All experiments are controlled via YAML configuration files in the **DATA /
INTERFACE / MODEL split** (design:
`docs/plans/2026-08-31-interface-config-redesign.md`). `config.py` loads them
(`load_config`) into typed dataclasses: unknown keys are refused with the
full path, ints coerce to floats, and `BASE: [<file>]` deep-merges include
files (the `_SMOKE` configs are the real config plus a few overrides).

### Key Configuration Parameters

**Top-level:**
- `MODE`: `train_and_test`, `only_test`, or `unsupervised_method`
- `DEVICE`: Target device (e.g., `cuda:0`)
- `LOG.PATH`: Output directory for runs (default: `runs/exp`)

**`DATA` — which stores participate** (one block; splits differ only via `SPLITS`):
- `CACHED_PATH`: Path to the zarr cache (one `*.zarr` store per recording)
- `DATASET`: Dataset identifier (`Neckflix`)
- `FILTERS`: Generic attribute include filters keyed by the store's own root attrs plus the `perspective` pseudo-attr, e.g. `FILTERS: {posture: ['0','45'], light: ['D']}`
- `PARTICIPANTS`: Include list (normalised ids; LOSO uses `--test_participants`)
- `ALLOW_MISSING` / `MIN_CHANNELS` / `MIN_LABELS`: Admission thresholds for partial recordings
- `SPLITS.TRAIN/VALID/TEST`: Per-split policy only — `STRIDE_SECONDS` (`0.0` = no overlap), `RANDOM_WINDOWS`, optional `FILTERS`/`PARTICIPANTS` overrides. Unsupervised runs use the TEST policy

**`INTERFACE` — the model's demand on the data pipeline.** Serialized into
every checkpoint; at `only_test` the checkpoint's copy is adopted over the
config's. The loader delivers what it demands — channels/traces the data
lacks come back as zeros + a False mask (even channels the dataset can never
provide), with warnings for zero coverage:
- `FS`: **Mandatory.** The frame rate the model sees, in Hz. Each store's own measured `fps` attr is reconciled against it: a faster store is decimated by nearest-index sampling, a store at the same *nominal* rate (within 1%) is taken as-is, and a genuinely slower one is refused unless `UPSAMPLING: interpolate` opts into linear frame/label blending (duplication would make DiffNormalized identically zero)
- `WINDOW_SECONDS`: The window as a **duration**. The frame count is derived, `T = WINDOW_SECONDS x FS`, snapped to a whole frame within 0.01 and refused otherwise — a frame count without a rate cannot distinguish 5 s of physiology from 1 s
- `CHANNELS` / `TRACES`: Camera channels to load and signals to predict; the *order* transfers to `model.channels` / `model.traces`
- `RESIZE`: Frame dimensions the model sees (`H`, `W`; `0` = keep the cache's) — resizing is consumer-side, so it need not match the cache resolution
- `DATA_TYPE`: Normalization methods (e.g., `['DiffNormalized', 'Standardized']`), consumer-side, concatenated along channels
- `LABEL_NORM`: **Per signal**, `{SIG: raw|zscore|minmax}`. Omit a signal for its class default: absolute-class (ABP, CVP) load `raw` in physical units, shape-class (PPG, ECG, RESP) are per-window z-scored

**`MODEL`:**
- `NAME`: Model architecture identifier (e.g., `DeepPhys`, `PhysMamba`, `PhysFormer`)
- `HEAD_STYLE` (`widened` default / `per_signal`), `DROP_RATE`, and per-model architecture blocks of any size (`MODEL.PHYSFORMER.*`)

**`TRAIN`:**
- `BATCH_SIZE`, `EPOCHS`, `LR`, `USE_AMP`/`AMP_DTYPE`
- `MODEL_FILE_NAME`: Checkpoint naming prefix
- `LOSS`: The per-signal registry (`{SIG: {TYPE, WEIGHTS}}`; omit for class defaults)
- `PLOT_LOSSES_AND_LR`: Whether to generate loss/LR plots

**`TEST` — scoring, in every mode** (absorbs the old `INFERENCE` block):
- `BATCH_SIZE`; `METRICS` (omit for the standard set)
- `USE_LAST_EPOCH`: If false, uses validation-based best epoch selection
- `EVALUATION_METHOD`: `FFT` or `peak detection` for heart rate derivation
- `EVALUATION_WINDOW`: Optional sliding window evaluation
- `MODEL_PATH`: The checkpoint to load at `only_test`

**`UNSUPERVISED`:**
- `METHODS`: The traditional methods to score (POS, CHROM, ICA, ...)


## Model / Trainer Patterns

### Models (`neural_methods/model/`)

Each model is a standalone Python file containing a `torch.nn.Module` subclass. Models are purely architectural -- they define the network structure and forward pass but contain no training logic.

Current models: DeepPhys, EfficientPhys, TS_CAN, PhysNet, PhysMamba, PhysFormer, RhythmFormer, BigSmall, iBVPNet, FactorizePhys, PhysHydra. PhysMamba is a `DictModel`; the rest await migration (roadmap Phases 4 and 6).

### Trainers (`neural_methods/trainer/`)

`MultiSignalTrainer` is the one trainer for every dict-contract model (see "Training on the contract" below). The per-model `<Model>Trainer.py` files and `BaseTrainer` are legacy: with `main.py`'s old dispatcher gone they are unreachable from the entry point, and each is kept only as migration reference until its model moves onto `MultiSignalTrainer`, then deleted (`BaseTrainer` goes with the last one). `tests/test_legacy_contract.py` drives `PhysMambaTrainer` directly to pin the tuple-contract behavior the unmigrated models still rely on.


## Entry Point

### `main.py`

Reads a zarr cache through a `BaseZarrDataset` subclass (currently `NeckflixDataset`), splits by participant (LOSO), and dispatches to either `MultiSignalTrainer` or the unsupervised predictor. DDP is optional: it is used when launched under `torch.distributed.run` and skipped otherwise, so the same script runs on a laptop and on a multi-GPU node. (Formerly `neckflix_main.py`; the upstream tuple-contract entry point it replaces is at the `pre-overhaul` tag.)

### Usage

```bash
# All seven unsupervised methods (CPU), scored per trace
uv run python main.py --config_file configs/neckflix/NECKFLIX_UNSUPERVISED.yaml

# PhysMamba, one LOSO fold
uv run python main.py --config_file configs/neckflix/NECKFLIX_PHYSMAMBA.yaml --test_participants P015

# Same fold across 4 GPUs
uv run python -m torch.distributed.run --nproc_per_node=4 main.py --config_file configs/neckflix/NECKFLIX_PHYSMAMBA.yaml --test_participants P015

# Enumerate LOSO folds (metadata-only; safe on a login node)
uv run python tools/list_neckflix_folds.py --config_file configs/neckflix/NECKFLIX_PHYSMAMBA.yaml --prefix P

# Summarise a finished run offline (per-signal, physical units)
uv run python tools/summarise_neckflix_outputs.py runs/neckflix_physmamba --by signal participant
```

`MultiSignalTrainer` saves one self-describing record per scored window --
prediction, label, and the `label_stats` that normalised them -- so
`tools/summarise_neckflix_outputs.py` recomputes every number, including the
inverse back to physical units, without touching the zarr cache.


## Naming Conventions

| Category | Convention | Examples |
|----------|-----------|----------|
| Models | PascalCase filenames | `PhysMamba.py`, `PhysFormer.py`, `DeepPhys.py` |
| Datasets | `<Dataset>Loader.py` (class) + `<Dataset>.md` (cache spec) | `NeckflixLoader.py`, `PURE.md` |
| YAML configs | `configs/neckflix/NECKFLIX_<MODEL>[_<VARIANT>].yaml` | `NECKFLIX_PHYSMAMBA_SMOKE.yaml` |
| SLURM scripts | `<Dataset>_<Model>_<Options>.slurm` | `Neckflix_PhysMamba_4GPU.slurm` |
| Loss functions | Descriptive names | `NegPearsonLoss.py`, `PerSignalLoss.py` |


## Extending the Toolbox

### Adding a New Dataset

The zarr cache is mandatory. A new dataset needs no loader machinery -- only:

1. An external preprocessor writing stores that satisfy the cache contract
   (for a known legacy dataset, its markdown cache spec in
   `dataset/data_loader/` says exactly what to write).
2. A `BaseZarrDataset` subclass declaring the `channel_map`.
3. A YAML config pointing `CACHED_PATH` at the cache.

### Adding a New Model

No new trainer -- `MultiSignalTrainer` serves every dict-contract model:

1. Make the architecture a `DictModel` subclass implementing `forward_video(video) -> (B, S, T)`, taking its input width from `self.in_channels` and its output width from `self.out_signals`. A 2-D per-frame backbone needs no change at all -- wrap it in `SignalDictWrapper(backbone, channels, traces, input_mode='frames2d')`.
2. Add a builder to `MODEL_REGISTRY` in `neural_methods/trainer/MultiSignalTrainer.py`.
3. Point a config at it with `MODEL.NAME: <ModelName>`.

### Adding New Metrics

1. Add the metric function to `evaluation/metrics.py`
2. Register it in the metric dispatcher within the evaluation code
3. Add the metric name to the `METRICS` list in experiment YAML configs


## The Zarr Cache Contract

Every dataset reaches this repository the same way: a directory of **zarr
stores, one per recording**, written by an external preprocessor. This repo
reads the cache and never writes it. The contract below is what
`dataset/data_loader/zarr_dataset.py` (`BaseZarrDataset`) actually enforces —
it is dataset-agnostic; Neckflix (next section) is the worked example, and
each legacy dataset has a markdown cache spec in `dataset/data_loader/`
describing how its raw files map onto this schema.

**Store discovery.** The configured cache directory is globbed for `*.zarr`;
the store's filename stem is the recording name. Nothing else in the
directory is consulted.

**Admission gate** (violations skip the store with a warning; a cache where
every store fails raises):

- Root attrs must carry `complete: true` — the JSON boolean, not a string.
  Partial preprocessor runs never get it, so half-written stores are invisible.
- Root attrs must carry `tool_version` parsing to `>= 1.0.0` (unparseable
  counts as 0). Pre-1.0.0 Neckflix stores were delta-encoded and would decode
  to garbage; the floor is the raw-frame format.

**Structure.** Inside a store:

```
{recording}.zarr
  attrs: complete, tool_version, [recording], [any filterable attributes...]
  {perspective}/              one group per camera view ("1", "2", ...)
    {stream}/                 one group per modality (rgb, ir, depth, ...)
      video/frames            (C, T, H, W) array of raw pixels
      video/  attrs: num_frames (required), fps
      {trace}/data            (T,) float, physical units, lowercase key
```

- A group under a perspective counts as a *stream* only if it has a `video`
  child; anything else (e.g. a root-level `events/` group) is ignored.
- `video.attrs["num_frames"]` is **required** — window enumeration reads it
  without touching pixels; a stream missing it is an error, not a skip.
- `fps` is written by the preprocessor but currently informational: the
  sampling rate used for evaluation comes from the config's `FS`.
- Trace groups sit *inside* each stream group, keyed **lowercase** (`abp`,
  `cvp`, `ecg`, `bvp`, ...), index-aligned to that stream's frames. A trace
  may be shorter than `num_frames` (trailing-NaN trimming); the loader
  NaN-pads the tail. When several streams carry the same trace, the loader
  takes the position-wise finite mean across copies.

**Root attrs are the filter surface.** Any root attr (`participant`,
`posture`, `light`, `session`, ...) can drive include/exclude filters and
LOSO fold enumeration; `perspective` works as a pseudo-attribute. Attrs are
free-form per dataset — the cache spec for each dataset says which it
guarantees. A store lacking a filtered attr fails any non-empty include and
passes an exclude-only filter, with a warning.

**Per-dataset code is a `channel_map` only.** A subclass declares how
canonical channel names map onto `(stream_group, channel_index)`:

```python
class NeckflixDataset(BaseZarrDataset):
    @property
    def channel_map(self):
        return {"R": ("rgb", 0), "G": ("rgb", 1), "B": ("rgb", 2),
                "I": ("ir", 0), "D": ("depth", 0)}
```

Everything else — admission, filtering, windowing, dense zero-filled
channels, per-window label normalisation, masks — is `BaseZarrDataset`.

**What the cache does NOT contain**: no face crops, no chunking, no
normalised pixels, no train/test splits. Frames are raw uint8/uint16 at one
resolution; `DATA_TYPE` normalisation and resizing happen consumer-side
(`neural_methods/frame_transforms.py`), and splits are attribute filters at
load time. One cache serves every experiment.


## Neckflix Dataset

The Neckflix dataset is a multimodal collection for cardiovascular pressure estimation — the worked example of the cache contract above. Its stores are produced by the external Neckflix preprocessor (`ghcr.io/coenarrow/neckflix` >= 1.0.0); the HDF5 loader they replaced is gone.

**Cache layout** (`{cache_dir}/{recording}.zarr`, zarr v3):

```
P015_S01_R3_0_D.zarr
  attrs: recording, participant ("015", unprefixed), session, repeat,
         posture, light, source_resolution, resized_to, tool_version, complete
  1/                          <- perspective (camera) key
    rgb/
      video/frames            (C, T, H, W) uint8, chunked (C, 32, H, W)
      video/  attrs: fps, num_frames
      abp/data                (T,) float64, physical units, index-aligned to frames
      cvp/data
      ecg/data
    ir/  depth/               same shape, uint16, C=1
  2/                          second perspective, same structure
```

Stores are admitted only if `complete is true` and `tool_version >= 1.0.0`; anything else is skipped with a warning. Which traces a recording carries **varies** — some have ABP+CVP, some CVP+ECG, some all three — so `ALLOW_MISSING` plus the per-sample `label_mask` is the normal configuration rather than an edge case.

**Camera modalities:** RGB, IR, Depth (event streams are not consumed).
**Physiological traces:** ABP, CVP, ECG.

Construction is metadata-only: no pixel data is read until `__getitem__`, so instantiating a dataset to enumerate LOSO folds is cheap enough for a login node (`tools/list_neckflix_folds.py`).


## The Batch-Dict Contract

Everything downstream of the Neckflix loader passes nested dicts keyed by canonical channel and signal names, so any tensor in the pipeline is identifiable by its key. `neural_methods/batch.py` is the single owner of those key names.

**Per sample** (dataset `__getitem__`):

```python
{"frames":       {ch:  (1, T, H, W) float32},   # raw pixel values, zero-filled where absent
 "labels":       {sig: (T,)         float32},   # per-window normalised
 "label_stats":  {sig: {stat: ()    float32}},  # physical units, for exact inversion
 "channel_mask": {ch:  ()           bool},      # True = real data, not zero fill
 "label_mask":   {sig: ()           bool},
 "metadata":     {"recording_id": str, "camera_id": str, "start_frame": int}}
```

`default_collate` handles the nesting: every tensor gains a leading batch axis and the metadata strings become lists. No custom `collate_fn` exists or is needed.

**Through a model:**

```python
out = model(batch)
out["predictions"]                    # {signal: (B, T)}
out["frames"] is batch["frames"]      # everything else passes through untouched
```

A model is a `DictModel` (`neural_methods/model/DictModel.py`). It owns the canonical channel and trace *order*; the dicts themselves are never iterated for order. A subclass implements only:

```python
def forward_video(self, video):   # (B, C_in, T, H, W) -> (B, S, T)
```

so retrofitting an architecture is a signature change, not a rewrite. `C_in` is `len(channels) * frame_transform.channel_multiplier`, since a `DATA_TYPE` of two transforms feeds the backbone two channel blocks of the same clip.

`DictModel.forward` also accepts a plain `(B, C, T, H, W)` tensor and returns a plain tensor, which is how the upstream tuple-contract trainers keep working against the same model classes.

**Frame preprocessing is consumer-side.** The loader emits raw pixels; `neural_methods/frame_transforms.py` applies `Raw` / `Standardized` / `DiffNormalized` and an optional spatial resize on `(B, C, T, H, W)` tensors, carried by the model that needs it. One cache resolution therefore serves models that want another.

**Reshaping is einops.** New and touched code uses `rearrange` / `reduce` / `einsum` rather than `view` / `permute` / `reshape`, so every shape move states its own meaning.

### Training on the contract

`neural_methods/trainer/MultiSignalTrainer.py` is the one trainer for every dict-contract model, driven by a registry:

```python
MODEL_REGISTRY = {'PhysMamba': _build_physmamba, 'DeepPhys': _build_deepphys}
```

Adding a model is a builder function plus a registry line. The builder receives a `ModelSpec` carrying everything derived from the data blocks (channel widths, output width, window `T`, frame size, target rate, per-signal label modes) and `build_model` then applies the shared guardrails: the declared window constraint is checked, and each output row's bias is initialised to its signal's physiological prior (ABP 90 mmHg, CVP 8) so training does not start with a ~90 mmHg systematic error.

The trainer owns DDP, AMP, `PerSignalLoss` (per-signal composite terms under a masked per-sample mean, so a signal absent from a whole batch contributes exactly 0 rather than NaN), the weight-decay exemption for the output readout, checkpointing, and evaluation reported **per predicted signal**.

Each signal is reported three ways: waveform Pearson/MAE/RMSE in its own prediction space, the same MAE/RMSE in physical units, and the inherited HR metrics. What the physical figure *means* depends on the signal's label mode. For a `raw` signal (ABP, CVP) the inverse is the identity and the mmHg error is genuine absolute-level accuracy. For a per-window normalised signal each window carries its own `label_stats`, so the figure is the error given a perfect estimate of that window's scale — *shape* expressed in the signal's units, with absolute level a question the normalisation deliberately removed.

### Splits

Splits are participant filters, not percentage slices. LOSO is two dataset instances over the same cache:

```python
train_cfg["filters"]["participant"] = {"include": [],      "exclude": ["015"]}
test_cfg["filters"]["participant"]  = {"include": ["015"], "exclude": []}
```

`dataset/data_loader/neckflix_config.py` builds those from the YAML plus `--test_participants`, including the id convention mismatch: the CLI says `P015`, the store's root attr says `"015"`.


## Pressure Estimation vs Heart Rate Estimation

The extension from heart rate estimation to pressure waveform estimation introduces several architectural differences:

| Aspect | Heart Rate (rPPG) | Pressure Waveform (ABP/CVP) |
|--------|-------------------|----------------------------|
| **Output** | Single scalar value (BPM) | Full temporal waveform |
| **Loss functions** | Rate estimation loss | Waveform regression loss (e.g., negative Pearson, MSE on waveform) |
| **Metrics** | MAE/RMSE on BPM | Waveform morphology, pressure ranges, clinical agreement |
| **Temporal resolution** | Aggregated over window | Sample-level prediction at high sampling rate |
| **Input** | RGB facial video | Multimodal (RGB, IR, Depth) at neck region |
| **Post-processing** | FFT or peak detection | Direct waveform output |

Models targeting pressure estimation (e.g., PhysHydra) may use multi-output architectures for simultaneous ABP and CVP prediction, and require adapted loss functions for waveform regression.


## Dataset Locations

| Dataset | Path | Notes |
|---------|------|-------|
| Neckflix (raw) | `/group/pgh004/carrow/repo/Neckflix/dataset` | Raw captures; input to the external preprocessor |
| Neckflix (zarr cache) | set by `CACHED_PATH` in the config | One `*.zarr` store per recording; what this repo reads |
| PURE | `/group/pgh004/carrow/zipped_datasets/PURE` | Standard rPPG dataset |
| UBFC-rPPG | `/group/pgh004/carrow/zipped_datasets/UBFC-rPPG` | Standard rPPG dataset |

Preprocessed caches are stored at paths defined by `CACHED_PATH` in the YAML config, typically under `PreprocessedData/`.


## Python Environment

The project uses **uv** as its Python package manager. The virtual environment (`.venv/`) is auto-managed by uv.

**Key dependencies:**
- PyTorch with CUDA support
- `zarr` (>=3.3, <4) for the Neckflix cache
- `einops` for every shape move in the Neckflix pipeline
- `mamba-ssm` (Linux/CUDA) or `mamba-ssm-macos`; where neither has a wheel,
  `neural_methods/model/mamba_compat.py` falls back to a pure-PyTorch Mamba block
- `h5py` only for the legacy `PhysHydraTrainer` (leaves with it in Phase 6)
- Standard scientific Python stack (numpy, scipy, matplotlib)

All scripts are invoked via `uv run python` to ensure the correct environment is used. Full dependency specifications are in `pyproject.toml`, with the lockfile at `uv.lock`.


## Related Repositories

**Neckflix** (`/mmfs1/data/group/pgh004/carrow/repo/Neckflix`, container `ghcr.io/coenarrow/neckflix`):
- Raw data preprocessing pipeline for Kinect Azure captures
- Writes the zarr cache this repo consumes; the coupling is the store schema plus two root-attr gates (`complete`, `tool_version`), not a Python import
- Configuration examples for different modalities and physiological traces
- Utilities for frame processing and trace filtering
- Shares similar config patterns (YAML structure) and package management (uv) with this repository
