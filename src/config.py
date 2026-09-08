"""YAML-to-dataclass loading shared by every config kind in ``src/``.

This module knows nothing about datasets, models, interfaces or training
recipes: each of those defines its own dataclass and validates it locally.
What lives here is only the machinery they all rely on, so it exists once.

Two guarantees callers depend on:

* ``load_yaml`` resolves ``BASE:`` includes (paths relative to the file) and
  deep-merges them in order before the file's own keys; scalars and lists
  override, mappings merge. Dot-less exponents such as ``LR: 9e-3`` load as
  numbers (YAML 1.2 semantics rather than PyYAML's 1.1 default).
* ``build`` fills a dataclass from a mapping, refusing unknown keys with the
  full dotted path, coercing ints to floats where the field asks for a float,
  and recursing into nested dataclass fields.
"""

import os
import re
from dataclasses import fields, is_dataclass

import yaml


class ConfigError(ValueError):
    """A config file said something the schema cannot accept."""


class _Loader(yaml.SafeLoader):
    """SafeLoader plus YAML 1.2 float resolution."""


_Loader.add_implicit_resolver(
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


def load_yaml(path: str) -> dict:
    """One YAML file as a mapping, with its ``BASE:`` includes merged in."""
    with open(path, "r") as handle:
        raw = yaml.load(handle, Loader=_Loader) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    bases = raw.pop("BASE", []) or []
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for base in bases:
        if not base:
            continue
        merged = _merge(merged, load_yaml(os.path.join(os.path.dirname(path), base)))
    return _merge(merged, raw)


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


def build(cls, mapping, path):
    """One dataclass from one YAML mapping, refusing unknown keys."""
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
            kwargs[name] = build(f.type, value, sub)
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
