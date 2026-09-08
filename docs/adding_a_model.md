# Adding a model

How to put a new architecture on the multi-signal contract, whether you are
writing it from scratch or migrating one of the upstream rPPG-Toolbox models.
DeepPhys is the worked example throughout: it is the simplest of the ten
models on the contract today (DeepPhys, PhysFormer, PhysMamba, PhysNet,
iBVPNet, TS-CAN, FactorizePhys, EfficientPhys, BigSmall, RhythmFormer), and
every file it touches is the file yours will touch.

The authority on *how* a model is run is [`run_experiment.py`](../run_experiment.py)
and the modules it imports from [`src/`](../src/). `main.py` and the
`neural_methods/trainer/` package are legacy and are not what this guide
describes.

## What a model is here

A model is **one complete copy of a single-trace architecture per predicted
trace**, wrapped in `MultiTraceModel` ([`src/models.py`](../src/models.py)),
speaking the batch dict: frames in, predictions out, nothing else.

- **Every width comes from the interface**, never from the model's YAML. The
  first layer takes the interface's channel count, and there is one copy of
  the network per entry of `TRACES`. Layer sizes are defined once, in the
  architecture's module, at their published values.
- **The loss is the trainer's, not the model's.** The interface's `LOSS` block
  states it per trace; a model that computes its own loss is wrong.
- **The trainer reaches into the model in exactly one place**: the readouts
  returned by `output_layers()`. It seeds each readout's bias with the trace's
  physiological prior and exempts the readouts from weight decay.

Because the wrapper owns channel order, trace order, the dict, and the
per-frame/clip folding, an architecture never sees a dict at all. It sees a
tensor and returns a tensor.

## The five things you touch

| # | What | Where | Exists for DeepPhys as |
| --- | ------ | ------- | ------------------------ |
| 1 | The backbone, a plain `nn.Module` | `neural_methods/model/<Name>.py` | [`neural_methods/model/DeepPhys.py`](../neural_methods/model/DeepPhys.py) |
| 2 | Config class + builder + two registry lines | `src/models.py` | `DeepPhysConfig`, `_build_deepphys`, the `"DeepPhys"` entries |
| 3 | The model config | `configs/models/<name>.yaml` | [`configs/models/deepphys.yaml`](../configs/models/deepphys.yaml) |
| 4 | The paper interface | `configs/interfaces/<name>_interface.yaml` | [`configs/interfaces/deepphys_interface.yaml`](../configs/interfaces/deepphys_interface.yaml) |
| 5 | One smoke test | `tests/test_<name>.py` | (not yet written; see step 5) |

Nothing else. No new trainer, loader, loss, dataset or plot. If your model
needs something the shared pieces almost do, extend the shared piece for
every model rather than adding a parallel copy beside it.

Copyable starting points:

- [`neural_methods/model/_template.py`](../neural_methods/model/_template.py)
  for the backbone (step 1);
- [`configs/models/_model_template.yaml`](../configs/models/_model_template.yaml)
  for the model config (step 3);
- [`configs/interfaces/_interface_template.yaml`](../configs/interfaces/_interface_template.yaml)
  for the paper interface (step 4);
- the code blocks in steps 2 and 5 below for the registration and the test.

## Step 1: the backbone

**File:** `neural_methods/model/<Name>.py`, one architecture per file, named
after the architecture as its paper names it.

The backbone is an `nn.Module` with no knowledge of dicts, traces or
channels. Its whole contract is:

### Constructor

```python
def __init__(self, in_channels: int = 3, ...published sizes as defaults...):
```

- `in_channels` is **required** and is the only width the wrapper always
  passes. It is `len(interface.CHANNELS) * len(input_blocks)`: the interface's
  channels stacked in order, once per named preprocessing block, blocks
  concatenated in the order the builder lists them (step 2). DeepPhys is
  handed `in_channels=len(CHANNELS)` and reads two blocks of that width,
  motion first and appearance second, slicing them off the channel axis
  itself.
- Any other width the interface determines (`img_size` for DeepPhys, whose
  dense layer is sized from the frame) is a constructor argument too, passed
  by the builder.
- Every published hyperparameter (filter counts, dropout, kernel sizes) is a
  default in the signature and is **not** exposed through YAML. At the
  defaults the module is the published network, layer for layer.

### `forward`

Exactly one of two shapes, declared by the builder's `per_frame` flag:

| `per_frame` | Input | Output | Who |
| ------------- | ------- | -------- | ----- |
| `True` | `(N, C_in, H, W)`, one frame per row; `T` is folded into `N` by the wrapper | `(N, 1)` | DeepPhys |
| `False` | `(B, C_in, T, H, W)`, a whole clip | `(B, 1, T)` | PhysNet, PhysFormer, PhysMamba, iBVPNet, FactorizePhys, RhythmFormer, TS-CAN, EfficientPhys, BigSmall |

TS-CAN, EfficientPhys and BigSmall take clips rather than per-frame rows even
though they are built around a temporal shift: that shift has to know where
each clip starts and ends to be adaptive within it, and a batch the wrapper
has already folded to `(N, C, H, W)` cannot tell it that, so all three fold
`(b t)` back inside the module instead. The shift itself is the shared `TSM`
in [`neural_methods/model/TS_CAN.py`](../neural_methods/model/TS_CAN.py),
imported by EfficientPhys directly and by BigSmall with `wrap=True` for its
wrap-around variant. DeepPhys is the only backbone that stays `per_frame`.

The output is one trace, width one. Never widen the readout to several
signals and never add per-signal heads on a shared trunk: the wrapper makes
the copies. A clip backbone must return three dimensions, `(B, 1, T)`, not
`(B, T)`; the wrapper concatenates copies on axis 1.

The input is preprocessed already. The dataset resizes to `RESIZE` and
applies every `INPUT_PREPROCESSING` block before the model sees anything, so
the backbone does no normalisation of its own.

### `output_layers()`

```python
def output_layers(self):
    return (self.final_layer,)
```

The activation-free readout(s) of this one copy, as a tuple. Each must be a
layer whose `bias` has exactly one element (an `nn.Linear(k, 1)` or an
`nn.Conv1d(k, 1, ...)`). The trainer refuses a model whose readouts do not
match its traces one to one.

### Any frame size, any window length

Every model must accept whatever `RESIZE` and `WINDOW_SECONDS` an interface
states, because the point of a standard interface is that every model runs
on the same one. Upstream code hard-codes things that break this: a token
grid written as `view(B, C, P // 16, 4, 4)` (true only at 128x128), a
`frame=160` baked into a head, a temporal stride the window must divide.
Each of those becomes one of:

- a **constructor argument derived from the interface** by the builder
  (`img_size` for DeepPhys's dense layer), so the module is sized for the
  frames it will actually see;
- a **config switch** when the paper leaves it free (a patch size);
- an **adaptive stage around the published network** that is an exact no-op
  at the paper's shape. PhysFormer average-pools the stem's output to the
  nearest whole tube and reads the token grid off the result; PhysFormer and
  PhysMamba pool time to a multiple of their stride before the trunk and
  interpolate the prediction back to the window length after it. At the
  paper's frame size and window every one of those is skipped, so the paper
  path is the published network, bit for bit, and no parameter is added or
  renamed.

Prefer the adaptive stage for anything the architecture cannot derive: it
keeps one module serving every interface. What is never acceptable is a
silent crop, truncation or reinterpretation. The one refusal that remains is
a frame the stem pools to nothing (8 px for PhysFormer, 16 px for
PhysMamba); the builder names it, via `_require_min_frame`.

Two interim class attributes exist for a migration that is not there yet:

```python
temporal_divisor = 4     # the window length must be a multiple of this
temporal_length = 128    # the window length must be exactly this
```

None of the ten migrated models declares either any more — every one reached
the adaptive stage described above — but `src/trainer.py` still honours them
for a model that is mid-migration: it reads them off the first copy and
refuses a mismatched `WINDOW_SECONDS` rather than truncating. They are a stop
on the way to the adaptive stage, not a destination.

### House rules

- Tensor reshaping uses einops (`rearrange`, `reduce`, `einsum`), not
  `view` / `permute` / `reshape`. This applies to migrated code too.
- No `params` argument, no loss, no `get_config`, no device handling.
- Nothing about traces or channels by name. A backbone cannot tell ABP from
  CVP and must not try.

## Step 2: registration in `src/models.py`

Three additions, all in [`src/models.py`](../src/models.py), each beside its
DeepPhys counterpart.

### The config class

```python
@dataclass
class MyNetConfig:
    NAME: str = ""
    INPUT: str = ""          # which INPUT_PREPROCESSING block the net reads

    def validate(self, interface: InterfaceConfig, where: str) -> None:
        _require_input_block(self.INPUT, interface, f"{where}: INPUT")
```

- **Fields are the YAML keys.** Every field is required in the file, unknown
  keys are refused, so the dataclass *is* the schema of `configs/models/<name>.yaml`.
- Fields are **experiment switches**, not sizes: which preprocessing block a
  branch reads, a head variant, a flag the paper ablates. If you find yourself
  adding `HIDDEN_DIM`, stop; it belongs in the module's defaults.
- `validate(interface, where)` is called at load, after the interface is
  loaded, and raises `ConfigError` for anything the interface cannot satisfy.
  `_require_input_block` checks a named block is both a known preprocessing
  and one the interface actually produces. A model with no switches still
  has `NAME` and an empty `validate`.

### The builder

```python
def _build_mynet(cfg: MyNetConfig, interface: InterfaceConfig) -> MultiTraceModel:
    width = len(interface.CHANNELS)
    return MultiTraceModel(
        make_copy=lambda: MyNet(in_channels=width),
        channels=interface.CHANNELS, traces=interface.TRACES,
        input_blocks=[cfg.INPUT], per_frame=False)
```

Inputs: the loaded config and the loaded interface. Output: a
`MultiTraceModel`. The builder is where every width is derived and where any
interface requirement beyond the config's is enforced (DeepPhys refuses a
non-square `RESIZE` here). `MultiTraceModel` takes:

| Argument | Meaning |
| ---------- | --------- |
| `make_copy` | zero-argument callable returning one fresh backbone; called once per trace |
| `channels` | `interface.CHANNELS`, the order the channel axis is stacked in |
| `traces` | `interface.TRACES`, the order copies are made and predictions keyed in |
| `input_blocks` | the preprocessing block names, in the order the backbone expects them on its channel axis; `C_in = len(channels) * len(input_blocks)` |
| `per_frame` | `True` for `(N, C, H, W) -> (N, 1)` backbones, `False` for `(B, C, T, H, W) -> (B, 1, T)` |

### The two registry lines

```python
MODEL_CONFIGS = {
    "DeepPhys": DeepPhysConfig,
    "MyNet": MyNetConfig,
}

MODEL_BUILDERS = {
    "DeepPhys": _build_deepphys,
    "MyNet": _build_mynet,
}
```

The key is what `NAME:` must say in the YAML. Use the architecture's proper
name, matching the module's class.

Also add the import at the top of the file beside `DeepPhys`'s:

```python
from neural_methods.model.MyNet import MyNet
```

## Step 3: the model config

**File:** `configs/models/<name>.yaml`. The stem is what `--model <name>`
resolves; keep it lowercase (`deepphys.yaml`, `mynet.yaml`). Copy
[`configs/models/_model_template.yaml`](../configs/models/_model_template.yaml).

```yaml
NAME: MyNet          # a key of MODEL_CONFIGS in src/models.py
INPUT: DiffNormalized   # every other key is a field of the config class
```

Inputs: read by `load_model_config(name, interface)` in `src/models.py`,
typed by `NAME`, every field required, unknown keys refused, then
`validate`d against the interface. Output: the config dataclass instance,
which `build_model` hands to your builder and which the run writes into its
`config.yaml` (and the checkpoint) alongside every other config it ran on.

Several experiments on one architecture are several files in
`configs/models/` with the same `NAME` and different switches.

## Step 4: the paper interface

**File:** `configs/interfaces/<name>_interface.yaml`, where `<name>` is the
architecture's `NAME` lowercased (`PhysFormer` reads
`physformer_interface.yaml`). This is what the `configs/interfaces/`
directory is for: when migrating a model, this file **is** the paper's
configuration of it. Nothing in code declares the paper setup. Copy
[`configs/interfaces/_interface_template.yaml`](../configs/interfaces/_interface_template.yaml)
and set every key to the rPPG-Toolbox definition of the model:

| Key | Upstream source |
| ----- | ----------------- |
| `FS` | the rate the published config trains at (30 for the UBFC recipes) |
| `WINDOW_SECONDS` | `CHUNK_LENGTH / FS`, written to six decimals so it snaps to a whole frame |
| `RESIZE` | the published `RESIZE.H` / `RESIZE.W` |
| `INPUT_PREPROCESSING` | the published `DATA_TYPE` list, in order |
| `TRACES` | `[PPG]`: the upstream models predict BVP and nothing else |
| `LABEL_PREPROCESSING` | the published `LABEL_TYPE`, as the nearest of `raw` / `zscore` |
| `LOSS` | the published criterion (`MSE` for DeepPhys, `NEGPEARSON` for PhysMamba and PhysFormer) |

Nothing Neckflix-specific goes in this file. The pressure traces, their
label preprocessing and their loss belong to the standard interface every
model is compared on. Where the loss module lacks a published term (the
DLDL frequency loss of PhysFormer, say), note it in the file rather than
substituting something.

**Its twin:** `configs/training/<name>_training.yaml`, the paper's training
recipe. Copy
[`configs/training/_training_template.yaml`](../configs/training/_training_template.yaml)
and fill it from two upstream sources, because the YAML alone does not say
how the model was optimised:

| Key | Upstream source |
| ----- | ----------------- |
| `EPOCHS`, `BATCH_SIZE`, `LR` | `TRAIN` block of the `train_configs/` file |
| `OPTIMIZER`, `WEIGHT_DECAY` | the `optim.*` call in `neural_methods/trainer/<Name>Trainer.py` at the `pre-overhaul` git tag (the legacy per-model trainers are deleted from the working tree; `git show pre-overhaul:neural_methods/trainer/<Name>Trainer.py` is where they still live) |
| `SCHEDULER` | the `lr_scheduler.*` call in the same trainer; a `StepLR` that never fires inside `EPOCHS` is `Constant` |
| `PRECISION` | `float32` unless the trainer autocasts |

`DEVICE` and `NUM_WORKERS` are machine keys, not the paper's. If the
trainer needs an optimiser or schedule the shared `src/trainer.py` lacks,
add it there for every model (one line in its `OPTIMIZERS` or `SCHEDULERS`
and one name in `src/training.py`), never a per-model training loop. Note
in the file any trainer-side difference that remains, such as the readout
being exempt from weight decay here when upstream decayed everything.

Nothing in code checks a run against this file; passing it as
`--interface` is what makes a run the paper's configuration. Model
comparisons then run every model on one standard interface instead, which
is why step 1 insists the model accepts any size. The ten paper interfaces
today:

| Model | Frames | Window | Input | Loss | Recipe |
| ------- | -------- | -------- | ------- | ------ | -------- |
| DeepPhys | 72x72 | 180 frames | DiffNormalized + Standardized | MSE | AdamW 9e-3, OneCycle, 30 epochs |
| PhysFormer | 128x128 | 160 frames | DiffNormalized | negative Pearson | Adam 1e-4, decay 5e-5, constant, 10 epochs |
| PhysMamba | 128x128 | 128 frames | DiffNormalized | negative Pearson | Adam 3e-3, decay 5e-4, OneCycle, 20 epochs |
| PhysNet | 72x72 | 128 frames | DiffNormalized | negative Pearson | Adam 9e-3, OneCycle, 30 epochs |
| iBVPNet | 72x72 | 160 frames | Raw | negative Pearson | Adam 1e-3, OneCycle, 30 epochs |
| TS-CAN | 72x72 | 180 frames | DiffNormalized + Standardized | MSE | AdamW 9e-3, OneCycle, 30 epochs |
| FactorizePhys | 72x72 | 160 frames | Raw | negative Pearson | Adam 1e-3, OneCycle, 10 epochs |
| EfficientPhys | 72x72 | 180 frames | Standardized | MSE | AdamW 9e-3, OneCycle, 30 epochs |
| BigSmall | 144x144 | 180 frames | Standardized + DiffNormalized | MSE | AdamW 1e-3, OneCycle, 5 epochs |
| RhythmFormer | 128x128 | 160 frames | Standardized | negative Pearson | AdamW 9e-3, OneCycle, 30 epochs |

## Step 5: one smoke test

**File:** `tests/test_<name>.py`. One test, and only one: build the model
from its paper interface and push one synthetic batch through. This is
the ceiling for a migration; do not add tests opportunistically.

```python
import torch

from src.interface import load_interface
from src.models import build_model, load_model_config

INTERFACE = "configs/interfaces/mynet_interface.yaml"


def test_mynet_forward_matches_the_contract():
    interface = load_interface(INTERFACE)
    model = build_model(load_model_config("mynet", interface), interface)
    B, T = 2, interface.window_frames
    H, W = interface.RESIZE.H, interface.RESIZE.W
    batch = {"frames": {ch: {prep: torch.zeros(B, T, H, W)
                             for prep in interface.INPUT_PREPROCESSING}
                        for ch in interface.CHANNELS}}
    out = model(batch)
    assert set(out["predictions"]) == set(interface.TRACES)
    assert all(p.shape == (B, T) for p in out["predictions"].values())
    assert len(model.output_layers()) == len(interface.TRACES)
```

Run it with `uv run pytest tests/test_mynet.py`. If the architecture cannot
yet take the interface's window, the test is where the `temporal_divisor`
refusal will surface first.

## Running it

```bash
uv run python run_experiment.py \
    --datasets neckflix \
    --test-participant-dataset neckflix --test-participant-id 1 \
    --model mynet \
    --interface configs/interfaces/mynet_interface.yaml \
    --training configs/training/mynet_training.yaml \
    --limit-windows 8
```

`--limit-windows N` keeps N evenly spaced windows per split for a wiring
check; drop it for a real fold. Pass `--interface` and `--training`
explicitly: the defaults point at `configs/interface.yaml` and
`configs/training.yaml`, which are not the files in this repository. Run
the paper interface first to check the migration against the paper, then
the standard interface every model is compared on.

What happens, in order (all of it in `run_experiment.py`):

1. `load_interface` reads the interface; `load_model_config` reads your YAML
   and validates it against that interface.
2. `build_model(model_config, interface)` calls your builder.
3. `Trainer(...)` checks the window against `temporal_divisor` /
   `temporal_length`, seeds the readout biases, exempts the readouts from
   weight decay, wraps in DDP if distributed, then `fit` and `test`.
4. `evaluate(records, run_dir, fs)` scores the held-out participant.

Outputs land in `runs/<model>_<dataset>_<participant>/`:

| File | Written by | Contents |
| ------ | ----------- | ---------- |
| `config.yaml` | trainer, before anything else | everything the run ran on in one mapping: the command, git commit, every dataset / interface / model / training config as loaded, the files they came from, the stores on each side of the hold-out, and the resolved device and precision |
| `model.pt` | trainer, every epoch | state dict plus the same compiled config |
| `losses.csv` | trainer, every epoch | per-epoch loss, per trace and component |
| `test_records.pt` | trainer, after `test` | one record per strided window, predictions and labels in physical units |
| `windows.csv`, `rates.csv`, `summary.csv`, plots | evaluation | per-window level scores for the absolute signals; per-window heart rate from every cardiac trace (PPG, ECG, ABP, CVP), from their fused spectra and from their median, each against its own label; the summary of both |

## Migrating an upstream rPPG-Toolbox model

Ten upstream rPPG-Toolbox models are migrated and registered in
`MODEL_CONFIGS` today (see the intro). One upstream model in
`neural_methods/model/` remains unmigrated —
[`PhysHydra.py`](../neural_methods/model/PhysHydra.py) — and it stays there
deliberately, out of scope rather than pending. An upstream model, before
migration, predicts one BVP trace from RGB and carries conventions this
contract drops. Migrating one is editing the module in place, not writing a
new one beside it, and the diff is almost always these items:

1. **Widen the first layer.** Replace the hard-coded `3` on the first conv
   with an `in_channels` constructor argument defaulting to `3`.
2. **One readout, width one, exposed.** Keep the final layer emitting a
   single trace and return it from `output_layers()`. Delete any
   multi-signal head, `ParallelSignals` or `DictModel` wrapper from the
   legacy attempt; the wrapper in `src/models.py` replaces all of them.
3. **Fix the output shape.** Clip models often emit `(B, T)`; return
   `(B, 1, T)` (`rearrange(x, "b t -> b 1 t")`). Per-frame models emit `(N, 1)`.
4. **Drop the preprocessing and the loss.** Upstream models sometimes
   normalise inputs or own a `FrameTransform`; the dataset does that now.
   Any loss lives in the interface's `LOSS` block.
5. **Drop `params`, `get_config`, and dead imports** (`pdb`, `math` for
   nothing). Delete rather than keep for compatibility.
6. **einops for every reshape.** `x.view(B, -1)` becomes
   `rearrange(x, "b c h w -> b (c h w)")`, and so on.
7. **Free the frame and the window.** Derive what the interface can supply,
   and wrap what it cannot in an adaptive stage that is the identity at the
   paper's shape (step 1, "Any frame size, any window length"). Only as an
   interim, declare `temporal_divisor` or `temporal_length` instead.
8. **Decide `per_frame`.** Does the paper feed single frames, with `T` hidden
   in the batch axis, or clips? A temporal shift that must stay adaptive
   within a clip (TS-CAN, EfficientPhys, BigSmall) needs the clip, so it
   folds `(b t)` inside the module instead and takes `per_frame=False`. That
   answer is the builder's `per_frame` flag; the module itself does not need
   to know.

Then steps 2 to 5 above. DeepPhys shows the finished form: compare
`neural_methods/model/DeepPhys.py` against the upstream file to see exactly
how small the diff is.

## Checklist

- [ ] `neural_methods/model/<Name>.py`: `in_channels` argument, published
      sizes as defaults, one of the two `forward` shapes, `output_layers()`,
      einops, no loss.
- [ ] `src/models.py`: import, `<Name>Config` with `validate`,
      `_build_<name>`, one line each in `MODEL_CONFIGS` and `MODEL_BUILDERS`.
- [ ] `configs/models/<name>.yaml`: `NAME` plus one key per config field.
- [ ] `configs/interfaces/<name>_interface.yaml`: the paper's rate, window,
      resize, input preprocessing, single PPG trace, label preprocessing
      and loss. Nothing Neckflix-specific.
- [ ] `configs/training/<name>_training.yaml`: the paper's epochs, batch,
      optimiser, rate, decay, schedule and precision, from the upstream
      config and trainer class.
- [ ] The model builds and runs on the standard interface too; any size the
      paper did not use goes through an adaptive stage, not a refusal.
- [ ] `tests/test_<name>.py`: one build-and-forward test, passing.
- [ ] A smoke run with `--limit-windows` reaches `evaluate` and writes the
      run directory.
- [ ] `README.md`, "Algorithms": add the model to the list on the contract,
      and the exact command that trains and tests it on PURE with only the
      first participant held out (`--datasets pure --test-participant-dataset
      pure --test-participant-id 01`) on its paper interface and paper
      training recipe. A migration is not finished until that command is
      there.
