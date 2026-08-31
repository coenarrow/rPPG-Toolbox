"""Typed experiment configuration for the zarr pipeline.

The schema is the DATA / INTERFACE / MODEL split designed in
docs/plans/2026-08-31-interface-config-redesign.md:

* ``DATA`` states which stores participate (cache path, filters, participants,
  admission thresholds) plus per-split sampling policy (``SPLITS``);
* ``INTERFACE`` states the model's demand on the data pipeline (rate, window,
  channels, traces, resize, ``DATA_TYPE``, per-signal label norm) — it is
  serialized into every checkpoint, and at ``only_test`` the checkpoint's copy
  is the authority;
* ``MODEL`` states the architecture: name, head style, and per-model
  hyperparameter blocks of arbitrary size.

Everything is a plain dataclass and the schema holds only keys a YAML file may
write: unknown keys are refused with the full path, ints coerce to floats
(``STRIDE_SECONDS: 0`` is legal), and the loader resolves ``9e-3``-style
floats (YAML 1.2 semantics), so numbers are numbers everywhere — including
inside the free-form per-signal (``TRAIN.LOSS``) and per-model
(``MODEL.<NAME>``) blocks. Runtime-derived values (experiment name, model and
output dirs) live on ``config.RUN`` (:class:`RunPaths`), assigned by
``main.py``.

``BASE:`` lists include files (paths relative to the config file), deep-merged
in order before the file's own keys; scalar and list values override, mappings
merge. That is what keeps the ``*_SMOKE`` variants to a handful of lines.
"""

import os
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass

import yaml

MODES = ("train_and_test", "only_test", "unsupervised_method")
UPSAMPLING_MODES = ("refuse", "interpolate")
SPLIT_NAMES = ("TRAIN", "VALID", "TEST")

DEFAULT_METRICS = ("MAE", "RMSE", "MAPE", "MACC", "Pearson", "SNR", "BA")


class ConfigError(ValueError):
    """A config file said something the schema cannot accept."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
@dataclass
class ResizeConfig:
    H: int = 0          # 0 = keep the cache's own size
    W: int = 0


@dataclass
class SplitConfig:
    """What genuinely differs between splits — nothing else may.

    ``FILTERS`` / ``PARTICIPANTS`` default to ``None`` meaning "inherit the
    ``DATA`` block's"; stating them (even empty) overrides.
    """

    STRIDE_SECONDS: float = 0.0     # 0 = stride by a whole window (no overlap)
    RANDOM_WINDOWS: bool = False
    FILTERS: dict = None
    PARTICIPANTS: list = None


@dataclass
class DataConfig:
    """Which stores participate. Facts about the data, not about the model."""

    DATASET: str = "Neckflix"
    CACHED_PATH: str = ""
    FILTERS: dict = field(default_factory=dict)
    PARTICIPANTS: list = field(default_factory=list)
    ALLOW_MISSING: bool = True
    MIN_CHANNELS: int = 1
    MIN_LABELS: int = 1
    SPLITS: dict = field(default_factory=dict)   # {TRAIN/VALID/TEST: SplitConfig}

    def split(self, name: str) -> SplitConfig:
        """The resolved policy for one split; unsupervised runs use TEST's."""
        key = str(name).upper()
        if key == "UNSUPERVISED":
            key = "TEST"
        if key not in SPLIT_NAMES:
            raise ConfigError(
                f"Unknown split {name!r}; valid: {list(SPLIT_NAMES)} (the "
                "unsupervised mode uses the TEST split policy)")
        return self.SPLITS.get(key, SplitConfig())


@dataclass
class InterfaceConfig:
    """The model's demand on the data pipeline.

    Serialized into every checkpoint (:func:`interface_payload`); the loader's
    job is to satisfy it — zero-fill + mask what the data cannot provide,
    resample time, resize space — or refuse with an error naming the fix.
    """

    FS: float = 0.0                 # model-facing rate, mandatory
    WINDOW_SECONDS: float = 0.0     # T = WINDOW_SECONDS x FS, mandatory
    CHANNELS: list = field(default_factory=list)
    TRACES: list = field(default_factory=list)
    RESIZE: ResizeConfig = field(default_factory=ResizeConfig)
    DATA_TYPE: list = field(default_factory=lambda: ["Standardized"])
    LABEL_NORM: dict = field(default_factory=dict)
    UPSAMPLING: str = "refuse"      # 'interpolate' opts into linear upsampling


@dataclass
class ModelConfig:
    """Architecture identity plus per-model blocks (``MODEL.<NAME>.<KEY>``).

    Any mapping under an unknown key becomes an attribute namespace, so
    ``config.MODEL.PHYSFORMER.PATCH_SIZE`` works without the schema having to
    enumerate every architecture's hyperparameters (PhysFormer retro item 1).
    """

    NAME: str = ""
    HEAD_STYLE: str = "widened"
    DROP_RATE: float = 0.0


@dataclass
class TrainConfig:
    EPOCHS: int = 1
    BATCH_SIZE: int = 4
    LR: float = 1e-4
    MODEL_FILE_NAME: str = ""
    USE_AMP: bool = True
    AMP_DTYPE: str = "bfloat16"
    LOSS: dict = field(default_factory=dict)    # per-signal registry


@dataclass
class TestConfig:
    """How predictions are scored — shared by every mode."""

    BATCH_SIZE: int = 4
    METRICS: list = field(default_factory=lambda: list(DEFAULT_METRICS))
    USE_LAST_EPOCH: bool = True
    EVALUATION_METHOD: str = "FFT"      # 'FFT' or 'peak detection'
    EVALUATION_WINDOW_SECONDS: float = 0.0  # 0 = score each window whole
    MODEL_PATH: str = ""                # only_test: the checkpoint to load


@dataclass
class ExperimentConfig:
    MODE: str = "train_and_test"
    DEVICE: str = "cuda:0"
    DEBUG: bool = False
    LOG_PATH: str = "runs/exp"
    UNSUPERVISED_METHODS: list = field(default_factory=list)
    DATA: DataConfig = field(default_factory=DataConfig)
    INTERFACE: InterfaceConfig = field(default_factory=InterfaceConfig)
    MODEL: ModelConfig = field(default_factory=ModelConfig)
    TRAIN: TrainConfig = field(default_factory=TrainConfig)
    TEST: TestConfig = field(default_factory=TestConfig)


@dataclass
class RunPaths:
    """Derived by ``main.py`` at startup; never written in YAML.

    Attached to the loaded config as ``config.RUN`` so every consumer that
    already holds the config can reach the run's directories.
    """

    exp_name: str = ""
    model_dir: str = ""     # checkpoints
    output_dir: str = ""    # saved prediction outputs (per mode)


class ModelBlock:
    """Read-only-ish attribute view of one per-model architecture mapping."""

    def __init__(self, mapping: dict, path: str):
        self._path = path
        for key, value in mapping.items():
            setattr(self, str(key),
                    ModelBlock(value, f"{path}.{key}")
                    if isinstance(value, dict) else value)

    def __getattr__(self, name):
        raise AttributeError(
            f"{self._path} has no key {name!r}; it carries "
            f"{sorted(k for k in vars(self) if not k.startswith('_'))}")

    def __repr__(self):
        entries = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        return f"ModelBlock({self._path}: {entries})"


# ---------------------------------------------------------------------------
# Building the schema from a YAML mapping
# ---------------------------------------------------------------------------
def _coerce_scalar(value, target, path):
    if target is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path} must be a number, got {value!r}")
        return float(value)
    if target is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or (isinstance(value, float) and not value.is_integer()):
            raise ConfigError(f"{path} must be an integer, got {value!r}")
        return int(value)
    if target is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path} must be true/false, got {value!r}")
        return value
    if target is str:
        if value is None:
            return ""
        if isinstance(value, (str, int, float)):
            return str(value)
        raise ConfigError(f"{path} must be a string, got {value!r}")
    return value


def _build(cls, mapping, path):
    """One dataclass block from one YAML mapping, refusing unknown keys."""
    if mapping is None:
        mapping = {}
    if not isinstance(mapping, dict):
        raise ConfigError(f"{path} must be a mapping, got {mapping!r}")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(str(k) for k in mapping if k not in known)
    if unknown:
        raise ConfigError(
            f"{path or 'config'} has unknown key(s) {unknown}; "
            f"valid keys: {sorted(known)}")
    kwargs = {}
    for name, f in known.items():
        if name not in mapping:
            continue
        value = mapping[name]
        sub = f"{path}.{name}" if path else name
        if is_dataclass(f.type):
            kwargs[name] = _build(f.type, value, sub)
        elif f.type is dict:
            if value is None:
                value = {}
            if not isinstance(value, dict):
                raise ConfigError(f"{sub} must be a mapping, got {value!r}")
            kwargs[name] = dict(value)
        elif f.type is list:
            if value is None:
                value = []
            if not isinstance(value, list):
                raise ConfigError(f"{sub} must be a list, got {value!r}")
            kwargs[name] = list(value)
        else:
            kwargs[name] = _coerce_scalar(value, f.type, sub)
    return cls(**kwargs)


def _build_model(mapping):
    """``MODEL``: known fields via the schema, mappings become model blocks."""
    if mapping is None:
        mapping = {}
    if not isinstance(mapping, dict):
        raise ConfigError(f"MODEL must be a mapping, got {mapping!r}")
    known = {f.name for f in fields(ModelConfig)}
    plain = {k: v for k, v in mapping.items() if k in known}
    extras = {k: v for k, v in mapping.items() if k not in known}
    for key, value in extras.items():
        if not isinstance(value, dict):
            raise ConfigError(
                f"MODEL.{key} is not a schema key, so it must be a per-model "
                f"architecture block (a mapping); got {value!r}. "
                f"Schema keys: {sorted(known)}")
    model = _build(ModelConfig, plain, "MODEL")
    for key, value in extras.items():
        setattr(model, str(key), ModelBlock(value, f"MODEL.{key}"))
    return model


def _build_splits(mapping):
    if mapping is None:
        mapping = {}
    splits = {}
    for name, value in mapping.items():
        key = str(name).upper()
        if key not in SPLIT_NAMES:
            raise ConfigError(
                f"DATA.SPLITS names unknown split {name!r}; valid: "
                f"{list(SPLIT_NAMES)} (the unsupervised mode uses TEST's policy)")
        split = _build(SplitConfig, value, f"DATA.SPLITS.{key}")
        if split.FILTERS is not None:
            split.FILTERS = dict(split.FILTERS)
        if split.PARTICIPANTS is not None:
            split.PARTICIPANTS = list(split.PARTICIPANTS)
        splits[key] = split
    return splits


def config_from_mapping(mapping: dict) -> ExperimentConfig:
    """A validated :class:`ExperimentConfig` from one merged YAML mapping."""
    if not isinstance(mapping, dict):
        raise ConfigError(f"The config root must be a mapping, got {mapping!r}")
    mapping = dict(mapping)
    model_mapping = mapping.pop("MODEL", None)
    splits_mapping = None
    if isinstance(mapping.get("DATA"), dict):
        data_mapping = dict(mapping["DATA"])
        splits_mapping = data_mapping.pop("SPLITS", None)
        mapping["DATA"] = data_mapping
    if isinstance(mapping.get("INTERFACE"), dict):
        interface_mapping = dict(mapping["INTERFACE"])
        resize = interface_mapping.get("RESIZE")
        if resize is not None and not isinstance(resize, dict):
            # Square shorthand: RESIZE: 128 == RESIZE: {H: 128, W: 128}
            interface_mapping["RESIZE"] = {"H": resize, "W": resize}
        mapping["INTERFACE"] = interface_mapping
    config = _build(ExperimentConfig, mapping, "")
    config.MODEL = _build_model(model_mapping)
    config.DATA.SPLITS = _build_splits(splits_mapping)
    _validate(config)
    return config


def _validate(config: ExperimentConfig) -> None:
    from neural_methods.signals import validate_channels, validate_traces

    if config.MODE not in MODES:
        raise ConfigError(f"MODE must be one of {list(MODES)}, got {config.MODE!r}")
    interface = config.INTERFACE
    if interface.FS <= 0:
        raise ConfigError(
            "INTERFACE.FS must be set to the frame rate the model should see "
            "(e.g. FS: 30). It is the rate WINDOW_SECONDS is converted at and "
            "the rate the loader resamples each store's native fps to.")
    if interface.WINDOW_SECONDS <= 0:
        raise ConfigError(
            "INTERFACE.WINDOW_SECONDS must be a positive duration; the frame "
            f"count is derived from it (T = WINDOW_SECONDS x FS), got "
            f"{interface.WINDOW_SECONDS!r}")
    if interface.UPSAMPLING not in UPSAMPLING_MODES:
        raise ConfigError(
            f"INTERFACE.UPSAMPLING must be one of {list(UPSAMPLING_MODES)}, "
            f"got {interface.UPSAMPLING!r}")
    try:
        interface.CHANNELS = validate_channels(interface.CHANNELS)
        interface.TRACES = validate_traces(interface.TRACES)
    except ValueError as err:
        raise ConfigError(f"INTERFACE: {err}") from err
    # Validate the per-signal registries early, with the config-side names, so
    # a typo fails at load rather than after the datasets are built. The
    # resolved LABEL_NORM is written back so checkpoints serialize the actual
    # per-signal modes, not the omission — a later change to a signal's class
    # default must not reinterpret an existing checkpoint's units.
    from dataset.data_loader.label_transforms import resolve_label_norms
    interface.LABEL_NORM = resolve_label_norms(interface.TRACES,
                                               interface.LABEL_NORM)
    if config.MODE == "train_and_test":
        from neural_methods.loss.PerSignalLoss import resolve_loss_specs
        resolve_loss_specs(interface.TRACES, config.TRAIN.LOSS)


# ---------------------------------------------------------------------------
# YAML loading (BASE includes, deep merge)
# ---------------------------------------------------------------------------
class _ConfigLoader(yaml.SafeLoader):
    """SafeLoader plus YAML 1.2 float resolution, so ``LR: 9e-3`` is a number
    (YAML 1.1 resolves dot-less exponents as strings)."""


_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(\.[0-9]+|[0-9]+(\.[0-9]*)?)([eE][-+]?[0-9]+)?$"),
    list("-+0123456789."))


def _merge(base: dict, override: dict) -> dict:
    """Deep-merge mappings; scalars and lists override, mappings merge."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml_tree(path: str) -> dict:
    with open(path, "r") as handle:
        raw = yaml.load(handle, Loader=_ConfigLoader) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    bases = raw.pop("BASE", []) or []
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for base in bases:
        if not base:
            continue
        merged = _merge(merged,
                        _load_yaml_tree(os.path.join(os.path.dirname(path), base)))
    return _merge(merged, raw)


def load_config(config_file: str) -> ExperimentConfig:
    """Load, merge (``BASE``) and validate one experiment config file."""
    return config_from_mapping(_load_yaml_tree(config_file))


# ---------------------------------------------------------------------------
# The interface as checkpoint metadata
# ---------------------------------------------------------------------------
def interface_payload(interface: InterfaceConfig) -> dict:
    """The interface as a plain dict, the form checkpoints carry it in."""
    return asdict(interface)


def interface_from_payload(payload: dict) -> InterfaceConfig:
    """Rebuild a checkpoint's interface; the same schema validation applies."""
    return _build(InterfaceConfig, payload, "checkpoint interface")


def interface_diff(config_side: InterfaceConfig, checkpoint_side: InterfaceConfig):
    """Human-readable differences, for the only_test adoption notice."""
    left, right = asdict(config_side), asdict(checkpoint_side)
    return [f"INTERFACE.{key}: config={left[key]!r} checkpoint={right[key]!r}"
            for key in left if left[key] != right[key]]
