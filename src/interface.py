"""The interface stage: the model's demand on the data pipeline.

``configs/interface.yaml`` says what every sample must look like when it
reaches the model — rate, window, channels, traces, resolution and the
preprocessing of frames and labels. Every key is required; there are no
defaults, so a run's interface is exactly what its file says.

The order of operations this describes, for every store of every dataset:

1. resample the store to ``FS`` (decimate, or interpolate if opted in);
2. cut a window of ``window_frames`` frames;
3. preprocess what the store *has*: ``INPUT_PREPROCESSING`` on the frames
   present, ``LABEL_PREPROCESSING`` on the traces present;
4. only then pad: a channel or trace in ``CHANNELS`` / ``TRACES`` the store
   lacks becomes zeros with a False mask.

Padding after preprocessing is what keeps a padded trace exactly zero — a
zero-filled trace pushed through zscore would not be.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path

from neural_methods.loss.PerSignalLoss import normalise_loss_weights
from src.config import ConfigError, build, load_yaml
from src.frame_transforms import FRAME_TRANSFORMS
from src.signal_transforms import (
    LABEL_TRANSFORMS, canonical_signal, validate_channels, validate_traces,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INTERFACE_PATH = REPO_ROOT / "configs" / "interface.yaml"

UPSAMPLING_MODES = ("refuse", "interpolate")
#: How far off a whole frame a duration may land before it is refused.
FRAME_SNAP_TOLERANCE = 0.01


@dataclass
class ResizeConfig:
    H: int = 0
    W: int = 0


@dataclass
class InterfaceConfig:
    FS: float = 0.0
    UPSAMPLING: str = ""
    WINDOW_SECONDS: float = 0.0
    WINDOW_STRIDE: float = 0.0
    CHANNELS: list = field(default_factory=list)
    TRACES: list = field(default_factory=list)
    RESIZE: ResizeConfig = field(default_factory=ResizeConfig)
    INPUT_PREPROCESSING: list = field(default_factory=list)
    LABEL_PREPROCESSING: dict = field(default_factory=dict)
    LOSS: dict = field(default_factory=dict)      # {trace: {component: weight}}

    # Derived, never written in YAML.
    @property
    def window_frames(self) -> int:
        return _frames(self.WINDOW_SECONDS, self.FS)

    @property
    def stride_frames(self) -> int:
        return _frames(self.WINDOW_STRIDE, self.FS)

    @property
    def resizes(self) -> bool:
        return self.RESIZE.H > 0


def _frames(seconds: float, fs: float) -> int:
    return int(round(seconds * fs))


def _snapped(seconds: float, fs: float, key: str) -> None:
    """Refuse a duration that is not a whole number of frames at ``fs``."""
    frames = seconds * fs
    if abs(frames - round(frames)) > FRAME_SNAP_TOLERANCE:
        raise ConfigError(
            f"{key} {seconds} s is {frames:.3f} frames at FS {fs}, not a whole "
            f"number; pick a duration that is (or write it as N / FS)")


def _require_every_key(mapping: dict, where: str) -> None:
    missing = sorted(f.name for f in fields(InterfaceConfig) if f.name not in mapping)
    if missing:
        raise ConfigError(
            f"{where}: every interface key is required; missing {missing}")


def validate_interface(cfg: InterfaceConfig, where: str) -> InterfaceConfig:
    """Every rule the interface carries, applied in place; returns ``cfg``."""
    if cfg.FS <= 0:
        raise ConfigError(f"{where}: FS must be a positive frame rate, got {cfg.FS}")
    if cfg.UPSAMPLING not in UPSAMPLING_MODES:
        raise ConfigError(
            f"{where}: UPSAMPLING must be one of {list(UPSAMPLING_MODES)}, "
            f"got {cfg.UPSAMPLING!r}")
    if cfg.WINDOW_SECONDS <= 0:
        raise ConfigError(
            f"{where}: WINDOW_SECONDS must be a positive duration, got "
            f"{cfg.WINDOW_SECONDS}")
    _snapped(cfg.WINDOW_SECONDS, cfg.FS, f"{where}: WINDOW_SECONDS")
    if cfg.WINDOW_STRIDE * cfg.FS < 1 - FRAME_SNAP_TOLERANCE:
        raise ConfigError(
            f"{where}: WINDOW_STRIDE must be at least one frame "
            f"(1 / FS = {1 / cfg.FS:.4f} s), got {cfg.WINDOW_STRIDE}")
    _snapped(cfg.WINDOW_STRIDE, cfg.FS, f"{where}: WINDOW_STRIDE")

    try:
        cfg.CHANNELS = validate_channels(cfg.CHANNELS)
        cfg.TRACES = validate_traces(cfg.TRACES)
    except ValueError as err:
        raise ConfigError(f"{where}: {err}") from err
    if len(set(cfg.CHANNELS)) != len(cfg.CHANNELS):
        raise ConfigError(f"{where}: CHANNELS repeats a channel: {cfg.CHANNELS}")
    if len(set(cfg.TRACES)) != len(cfg.TRACES):
        raise ConfigError(f"{where}: TRACES repeats a signal: {cfg.TRACES}")

    h, w = cfg.RESIZE.H, cfg.RESIZE.W
    if h < 0 or w < 0 or (h == 0) != (w == 0):
        raise ConfigError(
            f"{where}: RESIZE must be {{H: >0, W: >0}} or {{H: 0, W: 0}} for no "
            f"resize, got {{H: {h}, W: {w}}}")

    if not cfg.INPUT_PREPROCESSING:
        raise ConfigError(
            f"{where}: INPUT_PREPROCESSING must list at least one of "
            f"{list(FRAME_TRANSFORMS)}")
    unknown = [t for t in cfg.INPUT_PREPROCESSING if t not in FRAME_TRANSFORMS]
    if unknown:
        raise ConfigError(
            f"{where}: INPUT_PREPROCESSING has unknown entries {unknown}; "
            f"known: {list(FRAME_TRANSFORMS)}")

    # One entry per trace, no more and no fewer: a listed trace with no rule
    # would need a default, and a rule for an unlisted trace is a typo.
    resolved = {}
    for name, mode in cfg.LABEL_PREPROCESSING.items():
        signal = _canonical(name, f"{where}: LABEL_PREPROCESSING")
        if mode not in LABEL_TRANSFORMS:
            raise ConfigError(
                f"{where}: LABEL_PREPROCESSING.{name} must be one of "
                f"{list(LABEL_TRANSFORMS)}, got {mode!r}")
        resolved[signal] = mode
    _exactly_the_traces(resolved, cfg.TRACES, f"{where}: LABEL_PREPROCESSING")
    cfg.LABEL_PREPROCESSING = {t: resolved[t] for t in cfg.TRACES}

    # The loss is stated outright per trace: which components, at what
    # weight. No presets — the weights are the loss, and they are also where
    # the per-signal scale factors live. The rules are the loss module's.
    try:
        cfg.LOSS = normalise_loss_weights(cfg.TRACES, cfg.LOSS)
    except (ValueError, KeyError) as err:
        raise ConfigError(f"{where}: {err}") from err
    return cfg


def _canonical(name, where: str) -> str:
    try:
        return canonical_signal(name)
    except KeyError as err:
        raise ConfigError(f"{where}: {err}") from err


def _exactly_the_traces(mapping: dict, traces: list, where: str) -> None:
    missing = [t for t in traces if t not in mapping]
    extra = [s for s in mapping if s not in traces]
    if missing or extra:
        raise ConfigError(
            f"{where} must name exactly the TRACES {traces}; missing "
            f"{missing}, not in TRACES {extra}")


def parse_interface(mapping: dict, where: str) -> InterfaceConfig:
    """One interface mapping — a loaded file, or the ``interface`` section a
    run's compiled config carries — typed, every key present, every rule
    checked. ``where`` names the source in errors."""
    if not isinstance(mapping, dict):
        raise ConfigError(f"{where}: the interface must be a mapping")
    _require_every_key(mapping, where)
    cfg = build(InterfaceConfig, mapping, where)
    return validate_interface(cfg, where)


def load_interface(path: Path = DEFAULT_INTERFACE_PATH) -> InterfaceConfig:
    """The interface file, typed, every key present, every rule checked."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"No interface config at {path}")
    return parse_interface(load_yaml(str(path)), path.name)
