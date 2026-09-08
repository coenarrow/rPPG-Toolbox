"""The dataset stage: name a dataset, load its config, enumerate its stores.

A dataset is named on the command line (``--datasets neckflix pure``) and each
name resolves to ``configs/datasets/<name>.yaml``, which carries only the cache
location and include/exclude filters over the stores' root attrs. The stage
loads every store in the cache and then drops what the filters exclude; it
knows nothing about channels, traces, windows or rates — that is the
interface's business, and where padding and masking happen.
"""

from dataclasses import dataclass, field
from pathlib import Path

import zarr

from src.config import ConfigError, build, load_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_CONFIG_DIR = REPO_ROOT / "configs" / "datasets"


# ---------------------------------------------------------------------------
# Config: which stores participate, nothing else
# ---------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    CACHED_PATH: str = ""
    FILTERS: dict = field(default_factory=dict)   # {attr: {include, exclude}}


def normalise_filters(filters: dict, where: str) -> dict:
    """``{attr: {"include": [...], "exclude": [...]}}`` with values as strings.

    Every entry must be a mapping with exactly the keys ``include`` and
    ``exclude``, each a list (empty = no constraint). An attr may be nested
    (dotted path). Overlapping include/exclude is a config error.
    """
    out = {}
    for attr, spec in (filters or {}).items():
        if not isinstance(spec, dict) or set(spec) != {"include", "exclude"}:
            raise ConfigError(
                f"{where}: FILTERS.{attr} must be a mapping with exactly the "
                f"keys include and exclude (each a list, [] = no constraint), "
                f"got {spec!r}")
        for key in ("include", "exclude"):
            if spec[key] is not None and not isinstance(spec[key], list):
                raise ConfigError(
                    f"{where}: FILTERS.{attr}.{key} must be a list, got "
                    f"{spec[key]!r}")
        include = [str(v) for v in (spec["include"] or [])]
        exclude = [str(v) for v in (spec["exclude"] or [])]
        overlap = sorted(set(include) & set(exclude))
        if overlap:
            raise ConfigError(
                f"{where}: FILTERS.{attr} both includes and excludes {overlap}")
        out[str(attr)] = {"include": include, "exclude": exclude}
    return out


def resolve_dataset_configs(names: list[str]) -> dict[str, Path]:
    """``{name: path}`` for each dataset name, or a ValueError naming the gap.

    A name resolves to ``configs/datasets/<name>.yaml``; nothing else is
    searched. The name doubles as the label a test participant is qualified
    with later (``--test-participant-dataset``), so it has to be exact.
    """
    resolved = {}
    for name in names:
        path = DATASET_CONFIG_DIR / f"{name}.yaml"
        if not path.is_file():
            available = sorted(p.stem for p in DATASET_CONFIG_DIR.glob("*.yaml"))
            raise ValueError(
                f"No dataset config for {name!r}: expected {path}. "
                f"Create it, or pick one of {available}.")
        resolved[name] = path
    return resolved


def load_dataset_config(path: Path) -> DatasetConfig:
    """One dataset file, typed and with unknown keys refused."""
    cfg = build(DatasetConfig, load_yaml(str(path)), path.stem)
    if not cfg.CACHED_PATH:
        raise ConfigError(f"{path}: CACHED_PATH is required")
    cfg.FILTERS = normalise_filters(cfg.FILTERS, path.stem)
    return cfg


def load_dataset_configs(names: list[str]) -> dict[str, DatasetConfig]:
    """``{name: DatasetConfig}`` for the names given on the command line."""
    return {name: load_dataset_config(path)
            for name, path in resolve_dataset_configs(names).items()}


# ---------------------------------------------------------------------------
# Stores: enumerate every store, then drop what the filters exclude
# ---------------------------------------------------------------------------
def _lookup(attrs: dict, dotted: str):
    """The value at a dotted path, or ``None`` when any step is absent."""
    node = attrs
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def admitted(attrs: dict, filters: dict) -> bool:
    """Include/exclude matching on (possibly nested) root attrs.

    A store lacking the attr, or carrying null, fails an include on it and
    passes an exclude: absence is never a reason to keep under include, and
    never a reason to drop under exclude.
    """
    for attr, spec in filters.items():
        value = _lookup(attrs, attr)
        text = None if value is None else str(value)
        if spec["include"] and text not in spec["include"]:
            return False
        if spec["exclude"] and text in spec["exclude"]:
            return False
    return True


def scan_stores(cfg: DatasetConfig) -> tuple[dict[Path, dict], dict[Path, dict]]:
    """``(kept, dropped)`` as ``{store path: root attrs}``. Metadata only."""
    root = Path(cfg.CACHED_PATH)
    if not root.is_dir():
        raise FileNotFoundError(f"CACHED_PATH {root} is not a directory")
    kept, dropped = {}, {}
    for store in sorted(root.glob("*.zarr")):
        attrs = dict(zarr.open_group(str(store), mode="r").attrs)
        (kept if admitted(attrs, cfg.FILTERS) else dropped)[store] = attrs
    return kept, dropped


def load_stores(configs: dict[str, DatasetConfig]) -> dict[str, dict[Path, dict]]:
    """``{dataset name: {store path: root attrs}}`` for every admitted store."""
    return {name: scan_stores(cfg)[0] for name, cfg in configs.items()}


# ---------------------------------------------------------------------------
# Test participant: held out of its own dataset before anything is concatenated
# ---------------------------------------------------------------------------
@dataclass
class Split:
    """Stores per dataset name for training, and for the held-out participant."""

    train: dict[str, dict[Path, dict]]
    test: dict[str, dict[Path, dict]]


def hold_out_participant(stores: dict[str, dict[Path, dict]],
                         dataset: str, participant: str) -> Split:
    """Drop one participant from one dataset; that participant is the test set.

    Ids are the dataset's own (``'1'`` in Neckflix and ``'01'`` in PURE are
    unrelated people), so the participant is removed from the named dataset
    only. Every other dataset trains whole. Naming a dataset that was not
    loaded, or a participant it does not carry, is an error listing what is.
    """
    if dataset not in stores:
        raise ValueError(
            f"--test-participant-dataset {dataset!r} is not one of the loaded "
            f"datasets {sorted(stores)}")
    participant = str(participant)
    pool = stores[dataset]
    test = {p: a for p, a in pool.items()
            if _lookup(a, "participant") is not None
            and str(_lookup(a, "participant")) == participant}
    if not test:
        present = sorted({str(_lookup(a, "participant")) for a in pool.values()},
                         key=lambda s: (len(s), s))
        raise ValueError(
            f"--test-participant-id {participant!r} matches no admitted store in "
            f"{dataset!r}; its participants are {present}")
    train = {name: dict(pool) for name, pool in stores.items()}
    train[dataset] = {p: a for p, a in pool.items() if p not in test}
    return Split(train=train, test={dataset: test})
