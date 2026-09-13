"""The training stage: fit a :class:`~src.models.MultiTraceModel`, then run the test set.

One class, ``Trainer``, holding the state that ``fit`` and ``test`` share —
runtime, criterion, optimiser, scheduler, scaler — and the run directory both
write to. Everything it does is stated by the three configs it is handed:

* the **interface** says what the loss is (``LOSS``, per trace) and which
  traces are predicted in raw units (``LABEL_PREPROCESSING``);
* the **training recipe** says how the train set is optimised;
* the **model** says nothing — it is a function from frames to predictions.

There is no validation split: LOSO scores on the held-out participant, and
the last epoch is the model. The test pass returns *records* — one dict per
strided window with predictions, labels, stats, masks and metadata, in
physical units — and writes nothing: ``src.outputs`` turns them into files,
and scoring them is the evaluation package's job, not this one's.

The trainer is also handed the run's compiled ``config`` — one plain mapping
of every config the run executed on, assembled by
``src.experiment.compile_config`` for ``scripts/train.py`` — and writes it to
the run directory as ``config.yaml`` the moment that directory exists, then
carries it inside every checkpoint. What a run ran on is never in doubt, even
when the files it was launched from have since changed, and
``scripts/infer.py`` rebuilds the run from the checkpoint alone. Handed no
config (inference), it writes nothing but the records.

Distributed runs (``src.distributed.Runtime`` with ``world_size > 1``): the
model is wrapped in ``DistributedDataParallel``, the train set is sharded by a
``DistributedSampler`` reshuffled every epoch, the test set by a plain strided
subset (no padding, so no duplicate records), loss sums are reduced across
ranks before averaging so the log is global, and only the main rank prints,
writes and saves. ``BATCH_SIZE`` is per process.

Two absolute-scale guardrails live here because they are trainer-side
machinery, not architecture: each readout's bias starts at its trace's
physiological prior (a raw-mmHg model otherwise spends its first epochs
learning that pressure is ~90, not ~0), and the readouts are exempt from
weight decay (decay on a raw-mmHg readout is a systematic bias dressed up as
regularisation). Both find the readouts through ``model.output_layers()``.
"""

import csv
import time
from pathlib import Path

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from tqdm import tqdm

from neural_methods.loss.PerSignalLoss import PerSignalLoss, weight_losses
from src.config import ConfigError
from src.distributed import Runtime, all_reduce_sum, gather_lists
from src.interface import InterfaceConfig
from src.models import MultiTraceModel
from src.signal_transforms import denormalise_label, signal_prior
from src.training import TrainingConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"

CHECKPOINT_NAME = "model.pt"
LOSS_LOG_NAME = "losses.csv"
CONFIG_NAME = "config.yaml"

#: The recipe's ``PRECISION`` -> autocast dtype (float32 = autocast off).
PRECISION_DTYPES = {
    "float32": None,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

#: The recipe's ``OPTIMIZER`` -> constructor over parameter groups. Adding one
#: is a line here plus its name in ``src.training.OPTIMIZERS``.
OPTIMIZERS = {
    "Adam": lambda groups, cfg: torch.optim.Adam(groups, lr=cfg.LR),
    "AdamW": lambda groups, cfg: torch.optim.AdamW(groups, lr=cfg.LR),
}

#: The recipe's ``SCHEDULER`` -> constructor, stepped once per batch;
#: ``total_steps`` is known at fit. ``Constant`` is the rate the recipe
#: states, every step (what an upstream ``StepLR`` that never fires inside
#: ``EPOCHS`` amounts to).
SCHEDULERS = {
    "OneCycle": lambda optimizer, cfg, total_steps: torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.LR, total_steps=total_steps),
    "Constant": lambda optimizer, cfg, total_steps: torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1.0),
}


# ---------------------------------------------------------------------------
# Construction-time checks and guardrails
# ---------------------------------------------------------------------------
def check_window(model: MultiTraceModel, interface: InterfaceConfig) -> None:
    """Refuse a window the architecture cannot process, naming the fix in seconds.

    A backbone that downsamples time internally declares ``temporal_divisor``;
    one built for exactly one length declares ``temporal_length``. Declared on
    the backbone, read off the first copy — every copy is the same network.
    """
    backbone = next(iter(model.copies.values()))
    name = type(backbone).__name__
    frames, fs = interface.window_frames, interface.FS
    fixed = getattr(backbone, "temporal_length", None)
    divisor = getattr(backbone, "temporal_divisor", 1) or 1
    if fixed and frames != fixed:
        raise ConfigError(
            f"{name} is built for exactly {fixed} frames, but WINDOW_SECONDS "
            f"gives {frames}. Use WINDOW_SECONDS: {fixed / fs:.6f} at FS={fs:g}.")
    if frames % divisor:
        nearest = max(round(frames / divisor), 1) * divisor
        raise ConfigError(
            f"{name} needs a window length divisible by {divisor}, but "
            f"WINDOW_SECONDS gives {frames} frames. Use WINDOW_SECONDS: "
            f"{nearest / fs:.6f} for {nearest} frames at FS={fs:g}.")


def init_output_bias(model: MultiTraceModel, interface: InterfaceConfig) -> None:
    """Start each readout's bias at its trace's prior: the signal's physiological
    level for a trace predicted raw, zero for one predicted z-scored."""
    layers = list(model.output_layers())
    if not layers:
        return
    if len(layers) != len(model.traces):
        raise ConfigError(
            f"{type(model).__name__}.output_layers() returned {len(layers)} "
            f"readouts for {len(model.traces)} traces; one copy per trace means "
            f"one readout per trace.")
    with torch.no_grad():
        for trace, layer in zip(model.traces, layers):
            if layer.bias is None or layer.bias.numel() != 1:
                raise ConfigError(
                    f"{type(model).__name__}: the {trace} readout needs exactly "
                    f"one bias entry, got "
                    f"{None if layer.bias is None else layer.bias.numel()}.")
            raw = interface.LABEL_PREPROCESSING[trace] == "raw"
            layer.bias.fill_(signal_prior(trace) if raw else 0.0)


def parameter_groups(model: MultiTraceModel, weight_decay: float) -> list:
    """Every parameter decays except the readouts', which decay at zero."""
    exempt = {id(p) for layer in model.output_layers() for p in layer.parameters()}
    decayed = [p for p in model.parameters() if id(p) not in exempt]
    undecayed = [p for p in model.parameters() if id(p) in exempt]
    groups = [{"params": decayed, "weight_decay": weight_decay}]
    if undecayed:
        groups.append({"params": undecayed, "weight_decay": 0.0})
    return groups


def to_physical(record: dict, label_preprocessing: dict) -> dict:
    """One record with ``predictions`` and ``labels`` back in physical units.

    Exact, not approximate: the stats that produced the normalisation ride in
    the record. For a ``raw`` signal the inverse is the identity.
    """
    stats = record["label_stats"]
    out = dict(record)
    for key in ("predictions", "labels"):
        out[key] = {sig: denormalise_label(trace, stats[sig], label_preprocessing[sig])
                    for sig, trace in record[key].items()}
    return out


# ---------------------------------------------------------------------------
# Walking the batch dict
# ---------------------------------------------------------------------------
def map_tensors(obj, fn):
    """Apply ``fn`` to every tensor in a nested dict/list, preserving structure;
    the metadata strings pass through."""
    if torch.is_tensor(obj):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: map_tensors(v, fn) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(map_tensors(v, fn) for v in obj)
    return obj


def move_to_device(batch, device, non_blocking: bool = False):
    return map_tensors(batch, lambda t: t.to(device, non_blocking=non_blocking))


def detach_to_cpu(batch):
    return map_tensors(batch, lambda t: t.detach().cpu())


def _index_sample(obj, i, n):
    """One batch element of a collated value: tensors indexed on their batch
    axis, the length-``n`` lists ``default_collate`` makes of the metadata
    strings likewise, and anything else — a 0-dim tensor is batch-level by
    construction — passed through."""
    if torch.is_tensor(obj):
        return obj if obj.ndim == 0 else obj[i]
    if isinstance(obj, dict):
        return {k: _index_sample(v, i, n) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)) and len(obj) == n:
        return obj[i]
    return obj


def iter_samples(out: dict):
    """Per-sample views of a model output, one dict per batch element, with
    the un-collated shapes the dataset produced. Sized off the predictions."""
    n = next(iter(out["predictions"].values())).shape[0]
    for i in range(n):
        yield _index_sample(out, i, n)


# ---------------------------------------------------------------------------
# The trainer
# ---------------------------------------------------------------------------
class Trainer:
    """Fit ``model`` by the recipe, then record its predictions on the test set."""

    def __init__(self, model: MultiTraceModel, interface: InterfaceConfig,
                 training: TrainingConfig, runtime: Runtime, run_dir: Path,
                 config: dict | None = None):
        check_window(model, interface)
        self.interface = interface
        self.training = training
        self.runtime = runtime
        self.config = {} if config is None else config
        self.run_dir = Path(run_dir)
        self.device = runtime.device
        self.dtype = PRECISION_DTYPES[runtime.precision]

        # Guardrails and readouts are found on the bare model; DDP wraps it
        # afterwards, and the checkpoint saves the bare state dict.
        init_output_bias(model, interface)
        self.model = model.to(self.device)
        self.net = model
        if runtime.distributed:
            ids = [self.device] if self.device.type == "cuda" else None
            self.net = DistributedDataParallel(model, device_ids=ids)
        self.criterion = PerSignalLoss(interface.TRACES, interface.LOSS, fs=interface.FS)
        self.optimizer = OPTIMIZERS[training.OPTIMIZER](
            parameter_groups(model, training.WEIGHT_DECAY), training)
        # Loss scaling is only a float16 concern; bfloat16 has float32's range.
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.dtype is torch.float16)

    # -- helpers ------------------------------------------------------------
    def _loader(self, dataset: Dataset, sampler=None, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset, batch_size=self.training.BATCH_SIZE, sampler=sampler,
            shuffle=shuffle and sampler is None,
            num_workers=self.training.NUM_WORKERS,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.training.NUM_WORKERS > 0)

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=self.dtype,
                              enabled=self.dtype is not None)

    def _losses(self, out: dict) -> tuple:
        """``(total, weighted)``: the scalar to backpropagate and the per-trace
        per-component floats for logging, weights from the interface."""
        raw = self.criterion(out["predictions"], out["labels"], out["label_mask"])
        return weight_losses(raw, self.criterion.weights)

    def _progress(self, iterable, desc: str):
        return tqdm(iterable, desc=desc, leave=False, disable=not self.runtime.is_main)

    def checkpoint(self) -> dict:
        """The model plus the compiled run config, so ``scripts/infer.py`` can
        rebuild the run from this file alone."""
        return {"model_state": self.model.state_dict(), "config": self.config}

    def _prepare_run_dir(self) -> None:
        """Main rank only: create the run directory and write ``config.yaml``
        into it, once, before anything else lands there."""
        if not self.runtime.is_main:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / CONFIG_NAME
        if self.config and not path.exists():
            with open(path, "w") as handle:
                yaml.safe_dump(self.config, handle, sort_keys=False)

    # -- fit ----------------------------------------------------------------
    def step(self, batch: dict) -> tuple[float, dict]:
        """One optimiser step on one collated batch: forward under autocast,
        the weighted loss, backward through the scaler, update. Returns the
        total as a float and the per-trace per-component floats. The
        scheduler is ``fit``'s to advance; this is the step alone, so a
        memory probe can take exactly one."""
        batch = move_to_device(batch, self.device, non_blocking=True)
        with self._autocast():
            out = self.net(batch)
            total, weighted = self._losses(out)
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(total).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return float(total.detach()), weighted

    def fit(self, train_dataset: Dataset) -> list:
        """Train for the recipe's epochs; returns the per-epoch loss log."""
        cfg, runtime = self.training, self.runtime
        sampler = DistributedSampler(train_dataset, num_replicas=runtime.world_size,
                                     rank=runtime.rank, shuffle=True) \
            if runtime.distributed else None
        loader = self._loader(train_dataset, sampler=sampler, shuffle=True)
        scheduler = SCHEDULERS[cfg.SCHEDULER](
            self.optimizer, cfg, total_steps=cfg.EPOCHS * len(loader))
        self._prepare_run_dir()
        log = []
        for epoch in range(cfg.EPOCHS):
            if sampler is not None:
                sampler.set_epoch(epoch)
            self.net.train()
            started = time.time()
            sums = {"batches": 0.0}
            progress = self._progress(loader, f"epoch {epoch + 1}/{cfg.EPOCHS}")
            for batch in progress:
                total, weighted = self.step(batch)
                scheduler.step()
                for trace, components in weighted.items():
                    for component, value in components.items():
                        key = f"{trace}/{component}"
                        sums[key] = sums.get(key, 0.0) + value
                sums["total"] = sums.get("total", 0.0) + float(total)
                sums["batches"] += 1
                progress.set_postfix(loss=f"{sums['total'] / sums['batches']:.4g}")
            sums = all_reduce_sum(sums, runtime)      # global means, not rank 0's
            n = max(sums.pop("batches"), 1.0)
            row = {"epoch": epoch, **{k: v / n for k, v in sums.items()},
                   "seconds": time.time() - started}
            log.append(row)
            if runtime.is_main:
                self._print_epoch(row)
                self._write_loss_log(log)
                torch.save(self.checkpoint(), self.run_dir / CHECKPOINT_NAME)
        return log

    def _print_epoch(self, row: dict) -> None:
        per_trace = ", ".join(f"{t}={row.get(f'{t}/total', 0.0):.4g}"
                              for t in self.model.traces)
        print(f"epoch {row['epoch'] + 1}/{self.training.EPOCHS}: "
              f"loss {row['total']:.4g} ({per_trace}) in {row['seconds']:.0f}s")

    def _write_loss_log(self, log: list) -> None:
        columns = list(dict.fromkeys(k for row in log for k in row))
        with open(self.run_dir / LOSS_LOG_NAME, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(log)

    # -- test ---------------------------------------------------------------
    @torch.no_grad()
    def test(self, test_dataset: Dataset) -> list:
        """Every strided window through the model; returns the records.

        A record is the per-sample view of the batch dict minus the frames:
        ``predictions``, ``labels``, ``label_stats``, ``label_mask``,
        ``channel_mask`` and ``metadata``, detached on the CPU and with the
        traces inverted to physical units. Nothing is written here. Under
        DDP each rank runs a strided shard and the main rank returns them
        all; the other ranks return ``[]``.
        """
        runtime = self.runtime
        shard = test_dataset
        if runtime.distributed:
            shard = Subset(test_dataset, range(runtime.rank, len(test_dataset),
                                               runtime.world_size))
        self.net.eval()
        records = []
        for batch in self._progress(self._loader(shard), "test"):
            batch = move_to_device(batch, self.device, non_blocking=True)
            with self._autocast():
                out = self.net(batch)
            out = {k: v for k, v in out.items() if k != "frames"}
            out["predictions"] = {t: p.float() for t, p in out["predictions"].items()}
            for sample in iter_samples(out):
                records.append(to_physical(detach_to_cpu(sample),
                                           self.interface.LABEL_PREPROCESSING))
        return gather_lists(records, runtime)
