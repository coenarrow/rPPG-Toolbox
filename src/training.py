"""The training stage: how the train set is optimised.

``configs/training.yaml`` is the recipe — epochs, batch size, optimiser,
schedule, precision — plus the two machine keys a run needs (device, loader
workers). Every key is required except ``MODEL_FILE_NAME``; there are no
defaults, so a run's recipe is exactly what its file says. Names
(``OPTIMIZER``, ``SCHEDULER``) are admitted only when the trainer implements
them.
"""

from dataclasses import dataclass, fields
from pathlib import Path

from src.config import ConfigError, build, load_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAINING_PATH = REPO_ROOT / "configs" / "training.yaml"

OPTIMIZERS = ("Adam", "AdamW")
SCHEDULERS = ("OneCycle", "Constant")
PRECISIONS = ("float32", "bfloat16", "float16")
DEVICES = ("gpu", "cpu")
OPTIONAL_KEYS = ("MODEL_FILE_NAME",)


@dataclass
class TrainingConfig:
    EPOCHS: int = 0
    BATCH_SIZE: int = 0
    OPTIMIZER: str = ""
    LR: float = 0.0
    WEIGHT_DECAY: float = -1.0
    SCHEDULER: str = ""
    PRECISION: str = ""
    DEVICE: str = ""
    NUM_WORKERS: int = -1
    MODEL_FILE_NAME: str = ""     # optional; "" = derive from the run


def _require_every_key(mapping: dict, where: str) -> None:
    missing = sorted(f.name for f in fields(TrainingConfig)
                     if f.name not in mapping and f.name not in OPTIONAL_KEYS)
    if missing:
        raise ConfigError(
            f"{where}: every training key is required (except "
            f"{list(OPTIONAL_KEYS)}); missing {missing}")


def _one_of(value, allowed, key: str, where: str) -> None:
    if value not in allowed:
        raise ConfigError(
            f"{where}: {key} must be one of {list(allowed)}, got {value!r}")


def validate_training(cfg: TrainingConfig, where: str) -> TrainingConfig:
    """Every rule the recipe carries, applied in place; returns ``cfg``."""
    for key in ("EPOCHS", "BATCH_SIZE"):
        value = getattr(cfg, key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"{where}: {key} must be a positive integer, got {value!r}")
    if not isinstance(cfg.NUM_WORKERS, int) or isinstance(cfg.NUM_WORKERS, bool) \
            or cfg.NUM_WORKERS < 0:
        raise ConfigError(
            f"{where}: NUM_WORKERS must be a non-negative integer, got {cfg.NUM_WORKERS!r}")
    if cfg.LR <= 0:
        raise ConfigError(f"{where}: LR must be positive, got {cfg.LR}")
    if cfg.WEIGHT_DECAY < 0:
        raise ConfigError(
            f"{where}: WEIGHT_DECAY must be non-negative, got {cfg.WEIGHT_DECAY}")
    _one_of(cfg.OPTIMIZER, OPTIMIZERS, "OPTIMIZER", where)
    _one_of(cfg.SCHEDULER, SCHEDULERS, "SCHEDULER", where)
    _one_of(cfg.PRECISION, PRECISIONS, "PRECISION", where)
    _one_of(cfg.DEVICE, DEVICES, "DEVICE", where)
    # A gpu recipe on a box without one falls back to CPU with a warning at
    # runtime (src.distributed.init_runtime); a recipe that *writes* cpu with
    # a reduced precision contradicts itself and is refused here.
    if cfg.PRECISION != "float32" and cfg.DEVICE == "cpu":
        raise ConfigError(
            f"{where}: PRECISION {cfg.PRECISION} needs DEVICE: gpu; on cpu use float32")
    return cfg


def load_training(path: Path = DEFAULT_TRAINING_PATH) -> TrainingConfig:
    """The training file, typed, every required key present, every rule checked."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"No training config at {path}")
    mapping = load_yaml(str(path))
    if not isinstance(mapping, dict):
        raise ConfigError(f"{path}: the training recipe must be a mapping")
    _require_every_key(mapping, path.name)
    cfg = build(TrainingConfig, mapping, path.stem)
    return validate_training(cfg, path.name)
