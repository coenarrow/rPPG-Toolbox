"""Trainer for the Neckflix batch-dict contract.

One trainer serves every dict-contract architecture, because the only thing
that differs between them is how a clip becomes a ``(B, S, T)`` tensor — and
that lives in the model (see :class:`DictModel`). Everything here is the part
that would otherwise be copy-pasted per model: DDP, AMP, the masked
multi-signal loss, checkpointing, and per-signal evaluation.

What makes it different from the legacy per-model trainers is that predictions
and labels stay *keyed by signal name* from the loader all the way to the
metric report, so a model predicting ABP and CVP together is scored on each
separately, and a window whose recording lacks one of them is simply not
counted for that signal (``label_mask``).
"""

import os
import pickle
from collections import defaultdict
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from config import interface_payload
from dataset.data_loader.neckflix_config import (
    frame_size, label_norms, window_frames,
)
from evaluation.plots import draw as draw_plots
from evaluation.records import from_saved
from evaluation.report import build_frame, digest, write as write_report
from neural_methods.batch import (
    ATTRS, LABEL_MASK, LABEL_STATS, LABELS, METADATA, PREDICTIONS,
    detach_to_cpu, iter_samples, move_to_device,
)
from neural_methods.frame_transforms import FrameTransform
from neural_methods.loss.PerSignalLoss import PerSignalLoss
from neural_methods.signals import signal_prior

NCOLS = 80

#: Weight decay applied to everything except the output readout, which is
#: exempt (see :func:`_parameter_groups`).
WEIGHT_DECAY = 5e-4


# ---------------------------------------------------------------------------
# What a builder gets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """The ``INTERFACE`` block, resolved once into what a builder consumes.

    A ``MODEL`` config block carries ``NAME`` plus genuinely architectural
    hyperparameters and nothing else; every width below is computed from the
    rate, window, channels, traces and resize the interface states, so nothing
    is said in two places. Builders read this rather than digging through the
    config themselves.
    """

    channels: tuple
    traces: tuple
    transform: FrameTransform
    fs: float
    window: int
    resize: tuple
    head_style: str
    label_norms: dict

    @property
    def camera_channels(self) -> int:
        """``len(channels)`` — the width of *one* ``DATA_TYPE`` block.

        This is the first-layer width for an architecture that splits the
        stacked blocks apart itself (DeepPhys and TS-CAN slice ``[:C]`` into the
        motion branch and ``[C:2C]`` into the appearance branch, exactly as
        upstream did with 3). Using the full stacked width there leaves the
        second branch with zero channels.
        """
        return len(self.channels)

    @property
    def in_channels(self) -> int:
        """``len(channels) x DATA_TYPE blocks`` — the width of the whole tensor.

        The first-layer width for an architecture that consumes the stacked
        blocks as one input (PhysMamba). See :attr:`camera_channels` for the
        models that split them.
        """
        return len(self.channels) * self.transform.channel_multiplier

    @property
    def out_signals(self) -> int:
        """One output row per trace — the final readout's width."""
        return len(self.traces)

    @property
    def img_size(self) -> tuple:
        """``(H, W)`` the backbone sees; an error if the config left it open."""
        if self.resize is None:
            raise ValueError(
                "INTERFACE.RESIZE.H/W must be set: this architecture sizes its "
                "dense layers from the frame size, so it cannot be built against "
                "'whatever the cache happens to be'.")
        return self.resize

    @property
    def priors(self) -> list:
        """Per-trace output-bias prior, in the units that trace is predicted in.

        A physiological prior only means something where the model predicts
        physical units; a per-window normalised label is centred already, so its
        prior is 0 and the initialisation is a no-op for it.
        """
        return [signal_prior(sig) if self.label_norms[sig] == 'raw' else 0.0
                for sig in self.traces]


def model_spec(config) -> ModelSpec:
    """Resolve the ``INTERFACE`` block (the model's demand) into a spec."""
    interface = config.INTERFACE
    fps = float(interface.FS)
    data_types = [t for t in interface.DATA_TYPE if t] or ['Standardized']
    size = frame_size(interface)
    return ModelSpec(
        channels=tuple(interface.CHANNELS),
        traces=tuple(interface.TRACES),
        transform=FrameTransform(data_types, size=size),
        fs=fps,
        window=window_frames(interface.WINDOW_SECONDS, fps),
        resize=size,
        head_style=str(config.MODEL.HEAD_STYLE or 'widened'),
        label_norms=label_norms(interface),
    )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _build_physmamba(config, spec):
    from neural_methods.model.PhysMamba import PhysMamba
    return PhysMamba(channels=spec.channels, traces=spec.traces,
                     frame_transform=spec.transform, fs=spec.fs)


def _build_deepphys(config, spec):
    from neural_methods.model.DeepPhys import DeepPhys
    from neural_methods.model.SignalDictWrapper import SignalDictWrapper
    height, width = spec.img_size
    if height != width:
        raise ValueError(f"DeepPhys wants square frames; RESIZE is {height}x{width}")
    # DeepPhys is a two-branch network: it slices the stacked DATA_TYPE blocks
    # into a motion branch and an appearance branch itself, so it needs exactly
    # two of them and each conv is built for one block's width.
    if spec.transform.channel_multiplier != 2:
        raise ValueError(
            "DeepPhys needs exactly two DATA_TYPE entries — the motion branch "
            "takes the first block and the appearance branch the second. Use "
            "DATA_TYPE: ['DiffNormalized', 'Standardized']; got "
            f"{list(spec.transform.data_types)}.")
    backbone = DeepPhys(in_channels=spec.camera_channels, out_signals=spec.out_signals,
                        img_size=height, head_style=spec.head_style)
    return SignalDictWrapper(backbone, channels=spec.channels, traces=spec.traces,
                             input_mode='frames2d', frame_transform=spec.transform,
                             fs=spec.fs)


def _build_physformer(config, spec):
    from neural_methods.model.PhysFormer import PhysFormer
    if spec.head_style != 'widened':
        raise ValueError(
            "PhysFormer implements head style A (a widened readout) only. Its "
            "readout reads a feature whose token grid has already been averaged "
            "away, so per-signal head copies would every one of them see the "
            "identical vector; the style-B idea of a per-signal spatial "
            "weighting would mean moving the pooling into the head — a design "
            f"change, not a builder option. Got HEAD_STYLE {spec.head_style!r}.")
    block = config.MODEL.PHYSFORMER
    height, width = spec.img_size
    return PhysFormer(
        channels=spec.channels, traces=spec.traces, frame_transform=spec.transform,
        fs=spec.fs, image_size=(spec.window, height, width),
        patches=int(block.PATCH_SIZE), dim=int(block.DIM), ff_dim=int(block.FF_DIM),
        num_heads=int(block.NUM_HEADS), num_layers=int(block.NUM_LAYERS),
        theta=float(block.THETA), dropout_rate=float(config.MODEL.DROP_RATE))


#: Architectures that speak the batch-dict contract. Add a builder here to make
#: a model available to ``MODE: train_and_test`` on Neckflix.
MODEL_REGISTRY = {
    'PhysMamba': _build_physmamba,
    'DeepPhys': _build_deepphys,
    'PhysFormer': _build_physformer,
}


# ---------------------------------------------------------------------------
# Construction-time checks and the absolute-scale guardrails
# ---------------------------------------------------------------------------
def check_window(model, spec):
    """Refuse a window the architecture cannot process, naming the fix.

    The error is in seconds, not frames, because seconds is what the config
    says: telling someone "T must be a multiple of 4" when they wrote
    ``WINDOW_SECONDS: 4.3`` leaves them to do the conversion themselves.
    """
    fixed = getattr(model, 'temporal_length', None)
    divisor = getattr(model, 'temporal_divisor', 1) or 1
    if fixed and spec.window != fixed:
        raise ValueError(
            f"{type(model).__name__} is built for exactly {fixed} frames, but "
            f"WINDOW_SECONDS gives {spec.window}. Use "
            f"WINDOW_SECONDS: {fixed / spec.fs:.6f} at FS={spec.fs:g}.")
    if spec.window % divisor:
        nearest = max(round(spec.window / divisor), 1) * divisor
        raise ValueError(
            f"{type(model).__name__} needs a window length divisible by "
            f"{divisor}, but WINDOW_SECONDS gives {spec.window} frames. Use "
            f"WINDOW_SECONDS: {nearest / spec.fs:.6f} for {nearest} frames at "
            f"FS={spec.fs:g}.")


def init_output_bias(model, spec):
    """Start each output row at its signal's physiological prior.

    Without this an absolute-class model begins training predicting ~0 mmHg —
    a ~90 mmHg systematic error the first epochs spend themselves removing.
    Costs one tensor assignment and removes it up front.
    """
    layers = list(model.output_layers())
    priors = spec.priors
    if not layers or not any(priors):
        return
    with torch.no_grad():
        if len(layers) == 1:                       # style A: one widened readout
            bias = layers[0].bias
            if bias is None or bias.numel() != len(priors):
                raise ValueError(
                    f"{type(model).__name__}.output_layers() has "
                    f"{None if bias is None else bias.numel()} bias entries for "
                    f"{len(priors)} traces; a widened readout needs one each.")
            bias.copy_(torch.tensor(priors, dtype=bias.dtype, device=bias.device))
        elif len(layers) == len(priors):            # style B: one head per signal
            for layer, prior in zip(layers, priors):
                layer.bias.fill_(prior)
        else:
            raise ValueError(
                f"{type(model).__name__}.output_layers() returned {len(layers)} "
                f"layers for {len(priors)} traces; expected 1 (widened) or "
                f"{len(priors)} (per-signal).")


def _parameter_groups(model, readout):
    """Split parameters so the output readout is exempt from weight decay.

    Decay on a readout that emits raw mmHg pulls every prediction toward zero —
    a systematic pressure bias dressed up as regularisation. Everything else
    decays as it always did.
    """
    exempt = {id(p) for layer in readout for p in layer.parameters()}
    decayed, undecayed = [], []
    for parameter in model.parameters():
        (undecayed if id(parameter) in exempt else decayed).append(parameter)
    groups = [{'params': decayed, 'weight_decay': WEIGHT_DECAY}]
    if undecayed:
        groups.append({'params': undecayed, 'weight_decay': 0.0})
    return groups


def build_model(config):
    """Construct the configured dict-contract model from the experiment config."""
    name = config.MODEL.NAME
    builder = MODEL_REGISTRY.get(name)
    if builder is None:
        raise ValueError(
            f"Model {name!r} does not speak the Neckflix dict contract yet. "
            f"Available: {', '.join(sorted(MODEL_REGISTRY))}"
        )
    spec = model_spec(config)
    model = builder(config, spec)
    check_window(model, spec)
    init_output_bias(model, spec)
    # Contract v2: the criterion belongs to the model, so the config's
    # TRAIN.LOSS overrides have to reach it here — before .to(device) and
    # before any DDP wrap. Stage keys are the model's own, weighted by the
    # trainer; anything else has to name a trace, and resolve_loss_specs
    # refuses it here if it does not.
    overrides = dict(getattr(config.TRAIN, 'LOSS', None) or {})
    stage_names = set(model.loss_modules())
    signal_overrides = {k: v for k, v in overrides.items()
                        if k not in stage_names} or None
    model.attach_loss(PerSignalLoss(spec.traces, specs=signal_overrides,
                                    fs=spec.fs))
    return model


class MultiSignalTrainer:
    """Train/validate/test any :class:`DictModel` on the Neckflix zarr cache."""

    def __init__(self, config, data_loader, *, rank=0, world_size=1, debug=False):
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        self.debug = debug
        self.config = config
        self.local_rank = int(os.environ.get('LOCAL_RANK', self.rank))
        self.device = self._select_device()
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.RUN.model_dir
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.min_valid_loss = None
        self.best_epoch = 0

        # Everything identity-shaped comes from the one INTERFACE block — the
        # model's demand, which main.py has already reconciled with the
        # checkpoint's copy in only_test mode.
        interface = config.INTERFACE
        self.traces = list(interface.TRACES)
        self.channels = list(interface.CHANNELS)
        # The rate the model is trained and evaluated at, which the loader has
        # already resampled every store to. It flows into the spectral loss
        # term and the HR post-processing alike; neither hardcodes it.
        self.frame_rate = float(interface.FS)
        # Per signal, because an absolute-class signal is predicted in mmHg and
        # a shape-class one in its normalised space; the report has to invert
        # each with its own inverse.
        self.label_norms = label_norms(interface)

        model = build_model(config).to(self.device)
        readout = list(model.output_layers())
        if self.world_size > 1:
            device_ids = [self.local_rank] if self.device.type == 'cuda' else None
            model = DDP(model, device_ids=device_ids,
                        output_device=self.local_rank if device_ids else None)
        self.model = model

        self.criterion = PerSignalLoss(
            self.traces, specs=getattr(config.TRAIN, 'LOSS', None),
            fs=self.frame_rate)
        if self.is_main:
            print(f"Loss per signal:\n{self.criterion.extra_repr()}")

        self.use_amp = bool(getattr(config.TRAIN, 'USE_AMP', False)) and self.device.type == 'cuda'
        self.amp_dtype = torch.float16 if getattr(config.TRAIN, 'AMP_DTYPE', '') == 'float16' \
            else torch.bfloat16
        self.scaler = torch.amp.GradScaler('cuda') if (self.use_amp and self.amp_dtype == torch.float16) else None

        self.optimizer = None
        self.scheduler = None
        self.train_sampler = None
        if config.MODE == "train_and_test":
            if data_loader.get("train") is None:
                raise ValueError("train_and_test needs a train dataloader")
            self.num_train_batches = len(data_loader["train"])
            self.optimizer = optim.Adam(_parameter_groups(self.model, readout),
                                        lr=config.TRAIN.LR)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=max(self.num_train_batches, 1))
            if self.world_size > 1:
                self.train_sampler = data_loader["train"].sampler
        elif config.MODE != "only_test":
            raise ValueError("MultiSignalTrainer initialized in incorrect mode!")

    def _unwrap_model(self):
        """The underlying model, unwrapping DDP if necessary."""
        return self.model.module if isinstance(self.model, DDP) else self.model

    # --- setup helpers ---------------------------------------------------
    def _select_device(self):
        """Honour ``config.DEVICE`` when it asks for CPU; otherwise best available.

        ``DEVICE: cpu`` is how the smoke configs stay runnable on a machine that
        happens to have a GPU, so it has to actually mean CPU.
        """
        requested = str(getattr(self.config, 'DEVICE', '') or '').lower()
        if requested.startswith('cpu'):
            return torch.device('cpu')
        if torch.cuda.is_available():
            # LOCAL_RANK, not the global rank: on a multi-node job rank 5 is
            # local GPU 1 of node 1, not a sixth GPU on this node.
            return torch.device(f'cuda:{self.local_rank}')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def _autocast(self):
        return torch.amp.autocast(self.device.type, dtype=self.amp_dtype,
                                  enabled=self.use_amp)

    # --- training --------------------------------------------------------
    def _loss_for(self, batch):
        """Forward one batch and reduce it to the per-signal composite loss."""
        out = self.model(batch)
        loss, breakdown = self.criterion(out[PREDICTIONS], batch[LABELS],
                                         batch[LABEL_MASK])
        return loss, breakdown, out

    @staticmethod
    def _accumulate(totals, breakdown):
        """Fold one batch's ``{signal: {component: value}}`` into running sums."""
        for signal, terms in breakdown.items():
            for component, value in terms.items():
                totals[(signal, component)].append(value)

    def train(self, data_loader):
        if data_loader.get("train") is None:
            raise ValueError("No data for train")
        if self.world_size > 1:
            dist.barrier()

        mean_training_losses, mean_valid_losses, lrs = [], [], []
        # Per epoch, the mean of every loss component of every signal. Which
        # term dominates is the first thing debugging a multi-signal run needs,
        # and a single scalar curve cannot answer it.
        component_history = []
        for epoch in range(self.max_epoch_num):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            if self.is_main:
                print(f"\n====Training Epoch: {epoch}====")
            self.model.train()
            train_loss = []
            components = defaultdict(list)
            tbar = tqdm(data_loader["train"], ncols=NCOLS) if self.is_main else data_loader["train"]
            for batch in tbar:
                batch = move_to_device(batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    loss, breakdown, _ = self._loss_for(batch)
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                lrs.append(self.scheduler.get_last_lr())
                self.scheduler.step()
                train_loss.append(loss.item())
                self._accumulate(components, breakdown)
                if self.is_main:
                    tbar.set_description(f"Train epoch {epoch}")
                    tbar.set_postfix(loss=loss.item())

            epoch_loss = self._reduce_mean(train_loss)
            mean_training_losses.append(epoch_loss)
            component_history.append(
                {key: float(np.mean(values)) for key, values in components.items()})
            if self.is_main:
                # The progress bar only ever showed the last batch; the epoch
                # mean is what tells you whether training is going anywhere.
                print(f"mean training loss: {epoch_loss:.4f}")
                print("  " + "  ".join(
                    f"{signal}={component_history[-1][(signal, 'total')]:.4f}"
                    for signal in self.traces if (signal, 'total') in component_history[-1]))
            self.save_model(epoch)

            if not self.config.TEST.USE_LAST_EPOCH and data_loader.get("valid") is not None:
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                if self.is_main:
                    print('validation loss: ', valid_loss)
                    if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                        self.min_valid_loss = valid_loss
                        self.best_epoch = epoch
                        print(f"Update best model! Best epoch: {self.best_epoch}")
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        if not self.config.TEST.USE_LAST_EPOCH and self.is_main:
            print(f"best trained epoch: {self.best_epoch}, min_val_loss: {self.min_valid_loss}")
        if self.is_main:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs)
            self.plot_loss_components(component_history)

    def valid(self, data_loader):
        if data_loader.get("valid") is None:
            raise ValueError("No data for valid")
        if self.is_main:
            print("\n ====Validing===")
        self.model.eval()
        valid_loss = []
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=NCOLS) if self.is_main else data_loader["valid"]
            for batch in vbar:
                batch = move_to_device(batch, self.device)
                with self._autocast():
                    loss, _, _ = self._loss_for(batch)
                valid_loss.append(loss.item())
                if self.is_main:
                    vbar.set_description("Validation")
                    vbar.set_postfix(loss=loss.item())
        return self._reduce_mean(valid_loss)

    def _reduce_mean(self, values):
        """Mean of a per-rank list, averaged across ranks when distributed."""
        local = float(np.mean(values)) if values else float('nan')
        if self.world_size > 1:
            tensor = torch.tensor([local], device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            return tensor.cpu().item()
        return local

    # --- testing ---------------------------------------------------------
    def _load_weights_for_test(self):
        model = self._unwrap_model()
        if self.config.MODE == "only_test":
            path = self.config.TEST.MODEL_PATH
            if not os.path.exists(path):
                raise ValueError("Inference model path error! Please check TEST.MODEL_PATH in your yaml.")
            print("Testing uses pretrained model!\n" + path)
        elif self.config.TEST.USE_LAST_EPOCH:
            path = os.path.join(self.model_dir,
                                f"{self.model_file_name}_Epoch{self.max_epoch_num - 1}.pth")
            print("Testing uses last epoch as non-pretrained model!\n" + path)
        else:
            path = os.path.join(self.model_dir,
                                f"{self.model_file_name}_Epoch{self.best_epoch}.pth")
            print("Testing uses best epoch selected using model selection as non-pretrained model!\n" + path)
        model.load_state_dict(load_checkpoint_state(path, map_location=self.device))

    def test(self, data_loader):
        """Run inference and report metrics per predicted signal."""
        if not self.is_main:
            return None
        if data_loader.get("test") is None:
            raise ValueError("No data for test")
        print("\n===Testing===")
        self._load_weights_for_test()
        self.model = self.model.to(self.device)
        self.model.eval()

        windows = []          # per (recording, signal) window records, for saving
        with torch.no_grad():
            for batch in tqdm(data_loader["test"], ncols=NCOLS):
                on_device = move_to_device(batch, self.device)
                with self._autocast():
                    out = self.model(on_device)
                out = detach_to_cpu(out)
                for sample in iter_samples(out):
                    windows.extend(self._score_sample(sample))

        print('')
        run = from_saved(windows, fs=self.frame_rate, traces=self.traces,
                         label_norms=self.label_norms)
        hr_method = ('Peak' if self.config.TEST.EVALUATION_METHOD == "peak detection"
                     else 'FFT')
        frame = build_frame(run, bootstrap=self.config.TEST.REPORT.BOOTSTRAP,
                            hr_method=hr_method)
        summary = digest(frame, run)
        print(summary)
        draw_plots(frame, run, output_dir=self._plot_dir(),
                   filename_id=self._filename_id(),
                   plots=self.config.TEST.REPORT.PLOTS)
        if self.config.RUN.output_dir:
            write_report(frame, summary, self.config.RUN.output_dir,
                         self._filename_id())
            self.save_dict_outputs(windows)
        return frame

    def _score_sample(self, sample):
        """Return one window's saveable records, per signal actually labeled."""
        metadata = sample[METADATA]
        records = []
        for signal in self.traces:
            if not bool(sample[LABEL_MASK][signal]):
                continue
            prediction = sample[PREDICTIONS][signal].float().numpy()
            label = sample[LABELS][signal].float().numpy()
            records.append({
                'signal': signal,
                'recording_id': metadata['recording_id'],
                'camera_id': metadata['camera_id'],
                'start_frame': int(metadata['start_frame']),
                'attrs': dict(metadata.get(ATTRS) or {}),
                'prediction': prediction,
                'label': label,
                'label_stats': {k: float(v) for k, v in sample[LABEL_STATS][signal].items()},
            })
        return records

    def _plot_dir(self):
        """Where the standard plot set is written, next to the loss curves."""
        return os.path.join(self.config.LOG_PATH,
                            self.config.RUN.exp_name, 'plots')

    def plot_losses_and_lrs(self, train_loss, valid_loss, lrs):
        """Train/valid loss and LR curves (contract §6, plot 1).

        Formerly inherited from ``BaseTrainer``; absorbed here when the trainer
        stopped reading the legacy config tree.
        """
        output_dir = self._plot_dir()
        os.makedirs(output_dir, exist_ok=True)
        filename_id = self._filename_id()

        figure = plt.figure(figsize=(10, 6))
        epochs = range(len(train_loss))
        plt.plot(epochs, train_loss, label='Training Loss')
        if valid_loss:
            plt.plot(epochs, valid_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{filename_id} Losses')
        plt.legend()
        figure.savefig(os.path.join(output_dir, f'{filename_id}_losses.pdf'), dpi=300)
        plt.close(figure)

        figure = plt.figure(figsize=(6, 4))
        plt.plot(range(len(lrs)), lrs, label='Learning Rate')
        plt.xlabel('Scheduler Step')
        plt.ylabel('Learning Rate')
        plt.title(f'{filename_id} LR Schedule')
        plt.legend()
        figure.savefig(os.path.join(output_dir, f'{filename_id}_learning_rates.pdf'),
                       bbox_inches='tight', dpi=300)
        plt.close(figure)
        print('Saving plots of losses and learning rates to:', output_dir)

    def plot_loss_components(self, history):
        """Per-signal and per-component training curves (contract §6, plot 1).

        The scalar loss curve says whether training is going anywhere; these say
        *which signal* and *which term* is responsible, which is the first
        question any multi-signal run raises.
        """
        if not history:
            return
        signals = sorted({signal for epoch in history for signal, _ in epoch})
        if not signals:
            return
        epochs = range(len(history))
        figure, axes = plt.subplots(1, len(signals), figsize=(5 * len(signals), 4),
                                    squeeze=False)
        for axis, signal in zip(axes[0], signals):
            components = sorted({component for epoch in history
                                 for sig, component in epoch if sig == signal})
            for component in components:
                values = [epoch.get((signal, component), np.nan) for epoch in history]
                axis.plot(epochs, values, label=component,
                          linewidth=2.0 if component == 'total' else 1.2,
                          linestyle='-' if component == 'total' else '--')
            axis.set_title(f"{signal} ({self.criterion.specs[signal]['type']})",
                           fontsize=10)
            axis.set_xlabel('Epoch')
            axis.set_ylabel('Masked mean loss')
            axis.legend(fontsize=7)
        figure.suptitle(f"{self._filename_id()} — loss components", fontsize=11)
        figure.tight_layout()
        output_dir = self._plot_dir()
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{self._filename_id()}_loss_components.pdf")
        figure.savefig(path, bbox_inches='tight', dpi=200)
        plt.close(figure)
        print('Saving per-signal loss component curves to:', path)

    def _filename_id(self):
        if self.config.MODE == 'train_and_test':
            return self.model_file_name
        root = os.path.basename(self.config.TEST.MODEL_PATH).split(".pth")[0]
        return f"{root}_{self.config.DATA.DATASET}"

    def save_dict_outputs(self, windows):
        """Persist every scored window, keyed by signal and recording.

        Kept flat and self-describing (one record per window, carrying its own
        ``label_stats``) so downstream analysis can invert the normalisation
        without re-reading the cache.
        """
        output_dir = self.config.RUN.output_dir
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, self._filename_id() + '_outputs.pickle')
        payload = {
            'windows': windows,
            'traces': list(self.traces),
            'channels': list(self.channels),
            'fs': self.frame_rate,
            # The same identity the checkpoint records, so a pooled sweep can
            # say out loud when a `runs/` directory holds two models' folds.
            'model_name': self.config.MODEL.NAME,
            # Per signal, so downstream tooling can invert each one correctly
            # without knowing anything about the model that produced it.
            'label_norms': dict(self.label_norms),
        }
        with open(path, 'wb') as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saving outputs to:', path)

    def save_model(self, index):
        if not self.is_main:
            return
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, f"{self.model_file_name}_Epoch{index}.pth")
        # The checkpoint carries the interface it was trained against, so at
        # only_test the loaders can be pointed at what the model actually
        # demands rather than what a config happens to restate.
        torch.save({
            "state_dict": self._unwrap_model().state_dict(),
            "interface": interface_payload(self.config.INTERFACE),
            "model_name": self.config.MODEL.NAME,
        }, path)
        print('Saved Model Path: ', path)


def load_checkpoint_state(path, map_location=None):
    """The ``state_dict`` from a checkpoint, whichever format it is in.

    Checkpoints written since the interface redesign are
    ``{"state_dict", "interface", "model_name"}`` — the interface is the
    model's demand on the data pipeline and the authority at only_test. A bare
    ``state_dict`` (the pre-redesign format) still loads, with a warning that
    the config's INTERFACE block is being trusted blind.
    """
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    import warnings
    warnings.warn(
        f"{os.path.basename(path)} carries no interface metadata (pre-redesign "
        "checkpoint); the config's INTERFACE block is being trusted blind.")
    return payload


def checkpoint_interface(path):
    """The interface payload a checkpoint carries, or ``None`` for a bare one."""
    payload = torch.load(path, map_location='cpu')
    if isinstance(payload, dict) and "interface" in payload:
        return payload["interface"]
    return None
