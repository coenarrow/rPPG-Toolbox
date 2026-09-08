"""The model stage: name a model, load its config, check it against the interface.

A model is named on the command line (``--model deepphys``) and resolves to
``configs/models/<name>.yaml``. That file carries ``NAME`` — the architecture
— plus the switches an experiment may flip for it, and nothing else. Layer
sizes are not config: an architecture is defined once, in its module, at its
published values. Every width is derived from the interface (first layer from
``CHANNELS``, one copy of the network per entry of ``TRACES``), so nothing is
said in two places.

Every key is required, unknown keys are refused, and a switch that names an
interface block (a preprocessing the model reads by name) is checked here
against the interface it will run with.

The built model is a :class:`MultiTraceModel`: one complete copy of the
architecture per trace, speaking the batch dict. A model is a function from
frames to predictions and nothing else — the loss is the trainer's.
"""

from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange

from src.config import ConfigError, build, load_yaml
from neural_methods.batch import FRAMES, PREDICTIONS, split_signals
from neural_methods.frame_transforms import DATA_TYPES
from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.model import (
    PhysFormer as physformer, PhysMamba as physmamba, PhysNet as physnet,
    iBVPNet as ibvpnet,
)
from neural_methods.model.PhysMamba import PhysMamba
from neural_methods.model.TS_CAN import TSCAN
from src.interface import InterfaceConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_CONFIG_DIR = REPO_ROOT / "configs" / "models"


# ---------------------------------------------------------------------------
# Per-architecture config classes
# ---------------------------------------------------------------------------
@dataclass
class DeepPhysConfig:
    NAME: str = ""
    MOTION_INPUT: str = ""        # INPUT_PREPROCESSING block for the motion branch
    APPEARANCE_INPUT: str = ""    # ... and for the appearance branch

    def validate(self, interface: InterfaceConfig, where: str) -> None:
        for key in ("MOTION_INPUT", "APPEARANCE_INPUT"):
            _require_input_block(getattr(self, key), interface, f"{where}: {key}")
        if self.MOTION_INPUT == self.APPEARANCE_INPUT:
            raise ConfigError(
                f"{where}: MOTION_INPUT and APPEARANCE_INPUT are both "
                f"{self.MOTION_INPUT!r}; the two branches read different "
                f"preprocessings of the frame")


@dataclass
class TSCANConfig:
    NAME: str = ""
    MOTION_INPUT: str = ""        # INPUT_PREPROCESSING block for the motion branch
    APPEARANCE_INPUT: str = ""    # ... and for the appearance branch
    FRAME_DEPTH: int = 0          # segment length the temporal shift shifts within

    def validate(self, interface: InterfaceConfig, where: str) -> None:
        for key in ("MOTION_INPUT", "APPEARANCE_INPUT"):
            _require_input_block(getattr(self, key), interface, f"{where}: {key}")
        if self.MOTION_INPUT == self.APPEARANCE_INPUT:
            raise ConfigError(
                f"{where}: MOTION_INPUT and APPEARANCE_INPUT are both "
                f"{self.MOTION_INPUT!r}; the two branches read different "
                f"preprocessings of the frame")
        if self.FRAME_DEPTH <= 0:
            raise ConfigError(
                f"{where}: FRAME_DEPTH must be positive, got {self.FRAME_DEPTH}")


@dataclass
class PhysMambaConfig:
    NAME: str = ""
    INPUT: str = ""               # INPUT_PREPROCESSING block the stem reads

    def validate(self, interface: InterfaceConfig, where: str) -> None:
        _require_input_block(self.INPUT, interface, f"{where}: INPUT")


@dataclass
class PhysFormerConfig:
    NAME: str = ""
    INPUT: str = ""               # INPUT_PREPROCESSING block the stem reads

    def validate(self, interface: InterfaceConfig, where: str) -> None:
        _require_input_block(self.INPUT, interface, f"{where}: INPUT")


@dataclass
class PhysNetConfig:
    NAME: str = ""
    INPUT: str = ""               # INPUT_PREPROCESSING block the stem reads

    def validate(self, interface: InterfaceConfig, where: str) -> None:
        _require_input_block(self.INPUT, interface, f"{where}: INPUT")


@dataclass
class iBVPNetConfig:
    NAME: str = ""
    INPUT: str = ""               # INPUT_PREPROCESSING block the stem reads

    def validate(self, interface: InterfaceConfig, where: str) -> None:
        _require_input_block(self.INPUT, interface, f"{where}: INPUT")


#: ``NAME`` -> the dataclass its file is parsed into.
MODEL_CONFIGS = {
    "DeepPhys": DeepPhysConfig,
    "PhysFormer": PhysFormerConfig,
    "PhysMamba": PhysMambaConfig,
    "PhysNet": PhysNetConfig,
    "iBVPNet": iBVPNetConfig,
    "TSCAN": TSCANConfig,
}


def _require_min_frame(interface: InterfaceConfig, name: str, minimum: int) -> None:
    """A backbone whose stem pools spatially needs a frame it leaves something of.
    Only checkable here when the interface resizes; otherwise the backbone
    refuses at forward time with the same message."""
    if interface.resizes and min(interface.RESIZE.H, interface.RESIZE.W) < minimum:
        raise ConfigError(
            f"{name} pools frames down to nothing below {minimum}x{minimum}; the "
            f"interface RESIZE is {{H: {interface.RESIZE.H}, W: {interface.RESIZE.W}}}")


def _require_input_block(name: str, interface: InterfaceConfig, where: str) -> None:
    if name not in DATA_TYPES:
        raise ConfigError(
            f"{where} must be one of {list(DATA_TYPES)}, got {name!r}")
    if name not in interface.INPUT_PREPROCESSING:
        raise ConfigError(
            f"{where} is {name!r}, which the interface does not produce; its "
            f"INPUT_PREPROCESSING is {interface.INPUT_PREPROCESSING}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def resolve_model_config(name: str) -> Path:
    path = MODEL_CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        known = sorted(p.stem for p in MODEL_CONFIG_DIR.glob("*.yaml"))
        raise ConfigError(
            f"No model config {path.name} in {MODEL_CONFIG_DIR}; known: {known}")
    return path


def load_model_config(name: str, interface: InterfaceConfig):
    """``configs/models/<name>.yaml``, typed by its ``NAME``, checked against ``interface``."""
    path = resolve_model_config(name)
    mapping = load_yaml(str(path))
    if not isinstance(mapping, dict) or not mapping:
        raise ConfigError(f"{path.name}: a model config must be a non-empty mapping")
    arch = mapping.get("NAME")
    if arch not in MODEL_CONFIGS:
        raise ConfigError(
            f"{path.name}: NAME must be one of {sorted(MODEL_CONFIGS)}, got {arch!r}")
    cls = MODEL_CONFIGS[arch]
    missing = sorted(f.name for f in fields(cls) if f.name not in mapping)
    if missing:
        raise ConfigError(
            f"{path.name}: every {arch} key is required; missing {missing}")
    cfg = build(cls, mapping, path.stem)
    cfg.validate(interface, path.name)
    return cfg


# ---------------------------------------------------------------------------
# The multi-trace model
# ---------------------------------------------------------------------------
class MultiTraceModel(nn.Module):
    """S complete copies of a single-trace architecture, dict in, dict out.

    Each trace gets its own untouched copy of the published architecture, its
    first layer widened to the interface's channels. Signals such as ABP and
    CVP come from different regions of the frame, so they share no trunk; the
    cost is parameters, by design.

    A backbone is any ``nn.Module`` with ``forward(x)`` and
    ``output_layers()``. ``per_frame=True`` backbones see one frame at a time,
    ``(N, C_in, H, W) -> (N, 1)``, with T folded into the batch axis on the way
    in and out; clip backbones see ``(B, C_in, T, H, W) -> (B, 1, T)``.

    ``C_in = len(channels) * len(input_blocks)``: the backbone's input is the
    interface's channels stacked in order, once per named preprocessing block,
    blocks concatenated in ``input_blocks`` order. Channel and trace order is
    owned here, never inferred from dict iteration.

    ``forward(batch)`` returns the same dict with ``predictions`` added,
    ``{trace: (B, T)}``. Nothing is dropped in transit and nothing else is
    computed: the loss belongs to the trainer.
    """

    def __init__(self, make_copy, channels, traces, input_blocks, per_frame):
        super().__init__()
        self.channels = tuple(channels)
        self.traces = tuple(traces)
        self.input_blocks = tuple(input_blocks)
        self.per_frame = per_frame
        self.copies = nn.ModuleDict({trace: make_copy() for trace in self.traces})

    @property
    def in_channels(self) -> int:
        return len(self.channels) * len(self.input_blocks)

    def output_layers(self):
        """Each copy's activation-free readout, in traces order."""
        return [layer for trace in self.traces
                for layer in self.copies[trace].output_layers()]

    def prepare_frames(self, batch) -> torch.Tensor:
        """``batch['frames'][ch][block]`` ``(B, T, H, W)`` -> ``(B, C_in, T, H, W)``."""
        frames = batch[FRAMES]
        blocks = [torch.stack([frames[ch][block] for ch in self.channels], dim=1)
                  for block in self.input_blocks]
        return torch.cat(blocks, dim=1)

    def forward_video(self, video: torch.Tensor) -> torch.Tensor:
        """``(B, C_in, T, H, W)`` -> ``(B, S, T)``, traces order."""
        if self.per_frame:
            frames = rearrange(video, "b c t h w -> (b t) c h w")
            outs = [rearrange(self.copies[trace](frames), "(b t) s -> b s t",
                              b=video.shape[0])
                    for trace in self.traces]
        else:
            outs = [self.copies[trace](video) for trace in self.traces]
        return torch.cat(outs, dim=1)

    def forward(self, batch: dict) -> dict:
        predictions = split_signals(self.forward_video(self.prepare_frames(batch)),
                                    self.traces)
        return {**batch, PREDICTIONS: predictions}

    def extra_repr(self) -> str:
        return (f"channels={list(self.channels)}, traces={list(self.traces)}, "
                f"input_blocks={list(self.input_blocks)}, per_frame={self.per_frame}")


# ---------------------------------------------------------------------------
# Builders: (model config, interface) -> MultiTraceModel
# ---------------------------------------------------------------------------
def _build_deepphys(cfg: DeepPhysConfig, interface: InterfaceConfig) -> MultiTraceModel:
    if not interface.resizes or interface.RESIZE.H != interface.RESIZE.W:
        raise ConfigError(
            f"DeepPhys sizes its dense layer from a square frame; the interface "
            f"RESIZE is {{H: {interface.RESIZE.H}, W: {interface.RESIZE.W}}}")
    width = len(interface.CHANNELS)
    size = interface.RESIZE.H
    return MultiTraceModel(
        make_copy=lambda: DeepPhys(in_channels=width, img_size=size),
        channels=interface.CHANNELS, traces=interface.TRACES,
        input_blocks=[cfg.MOTION_INPUT, cfg.APPEARANCE_INPUT], per_frame=True)


def _build_tscan(cfg: TSCANConfig, interface: InterfaceConfig) -> MultiTraceModel:
    if not interface.resizes or interface.RESIZE.H != interface.RESIZE.W:
        raise ConfigError(
            f"TSCAN sizes its dense layer from a square frame; the interface "
            f"RESIZE is {{H: {interface.RESIZE.H}, W: {interface.RESIZE.W}}}")
    width = len(interface.CHANNELS)
    size = interface.RESIZE.H
    return MultiTraceModel(
        make_copy=lambda: TSCAN(in_channels=width, img_size=size, frame_depth=cfg.FRAME_DEPTH),
        channels=interface.CHANNELS, traces=interface.TRACES,
        input_blocks=[cfg.MOTION_INPUT, cfg.APPEARANCE_INPUT], per_frame=False)


def _build_physmamba(cfg: PhysMambaConfig, interface: InterfaceConfig) -> MultiTraceModel:
    _require_min_frame(interface, "PhysMamba", physmamba.MIN_FRAME)
    width = len(interface.CHANNELS)
    return MultiTraceModel(
        make_copy=lambda: physmamba.PhysMamba(in_channels=width),
        channels=interface.CHANNELS, traces=interface.TRACES,
        input_blocks=[cfg.INPUT], per_frame=False)


def _build_physformer(cfg: PhysFormerConfig, interface: InterfaceConfig) -> MultiTraceModel:
    _require_min_frame(interface, "PhysFormer", physformer.MIN_FRAME)
    width = len(interface.CHANNELS)
    return MultiTraceModel(
        make_copy=lambda: physformer.PhysFormer(in_channels=width),
        channels=interface.CHANNELS, traces=interface.TRACES,
        input_blocks=[cfg.INPUT], per_frame=False)


def _build_physnet(cfg: PhysNetConfig, interface: InterfaceConfig) -> MultiTraceModel:
    _require_min_frame(interface, "PhysNet", physnet.MIN_FRAME)
    width = len(interface.CHANNELS)
    return MultiTraceModel(
        make_copy=lambda: physnet.PhysNet(in_channels=width),
        channels=interface.CHANNELS, traces=interface.TRACES,
        input_blocks=[cfg.INPUT], per_frame=False)


def _build_ibvpnet(cfg: iBVPNetConfig, interface: InterfaceConfig) -> MultiTraceModel:
    _require_min_frame(interface, "iBVPNet", ibvpnet.MIN_FRAME)
    width = len(interface.CHANNELS)
    return MultiTraceModel(
        make_copy=lambda: ibvpnet.iBVPNet(in_channels=width),
        channels=interface.CHANNELS, traces=interface.TRACES,
        input_blocks=[cfg.INPUT], per_frame=False)


#: ``NAME`` -> builder. One line per architecture, beside its config class.
MODEL_BUILDERS = {
    "DeepPhys": _build_deepphys,
    "PhysFormer": _build_physformer,
    "PhysMamba": _build_physmamba,
    "PhysNet": _build_physnet,
    "iBVPNet": _build_ibvpnet,
    "TSCAN": _build_tscan,
}


def build_model(cfg, interface: InterfaceConfig) -> MultiTraceModel:
    """The model a loaded config describes, every width taken from ``interface``."""
    return MODEL_BUILDERS[cfg.NAME](cfg, interface)
