"""Per-signal label normalisation and the statistics that generate it.

The dataset normalises each label window at load time and stamps the
generating statistics into the batch. Every mode shares one signature and
returns the same four-key stats dict, so the batch layout is identical
whichever is used and every one is exactly invertible.

The mode is **per signal**, not per dataset (migration contract §3): an
absolute-class signal like ABP or CVP is loaded ``raw`` — physical units,
untouched — because its level is part of what the model has to predict, while
a shape-class signal like PPG or ECG is per-window z-scored, because only its
waveform carries information. :func:`resolve_label_norms` turns a trace list
plus optional config overrides into that mapping.
"""

import torch
from torch import Tensor

from neural_methods.signals import ABSOLUTE, canonical_signal, signal_class

# Canonical stat keys and their order. Consumers iterate or validate against
# this rather than hardcoding key lists.
STAT_NAMES: tuple[str, ...] = ("mean", "std", "min", "max")

# Numerical guard only: a constant trace yields 0/EPS == 0 instead of 0/0 ==
# NaN. That matters because masked losses compute ``values * mask`` and
# NaN * 0 is still NaN, so a NaN could not be masked away after the fact.
EPS = 1e-8


def _stats(trace: Tensor) -> dict[str, Tensor]:
    """Per-window statistics of one ``(T,)`` label trace, as 0-dim tensors."""
    return {
        "mean": trace.mean(),                # ()
        "std": trace.std(correction=1),      # () unbiased
        "min": trace.amin(),                 # ()
        "max": trace.amax(),                 # ()
    }


def _align(stat: Tensor, sig: Tensor) -> Tensor:
    """Right-pad ``stat`` with singleton dims so it broadcasts against ``sig``.

    Handles both the per-sample case (0-dim stat vs ``(T,)`` signal) and the
    collated case (``(B,)`` stat vs ``(B, T)`` signal).
    """
    return stat.reshape(stat.shape + (1,) * (sig.dim() - stat.dim()))


def zscore(trace: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """Z-score one ``(T,)`` trace; returns ``(normed, stats)`` in physical units."""
    stats = _stats(trace)                                             # 4 x ()
    normed = (trace - stats["mean"]) / stats["std"].clamp_min(EPS)    # (T,)
    return normed, stats


def zscore_inverse(sig: Tensor, stats: dict[str, Tensor]) -> Tensor:
    """Map a z-scored signal back to physical units: ``sig * std + mean``."""
    # Must clamp identically to zscore's forward division, or the round-trip
    # is inexact in the 0 < std < EPS band.
    return sig * _align(stats["std"].clamp_min(EPS), sig) + _align(stats["mean"], sig)


def minmax(trace: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """Min-max one ``(T,)`` trace to ``[0, 1]``; same stats dict as ``zscore``."""
    stats = _stats(trace)                                                     # 4 x ()
    span = (stats["max"] - stats["min"]).clamp_min(EPS)                       # ()
    normed = (trace - stats["min"]) / span                                    # (T,)
    return normed, stats


def minmax_inverse(sig: Tensor, stats: dict[str, Tensor]) -> Tensor:
    """Map a min-maxed signal back to physical units: ``sig * (max - min) + min``."""
    # Must clamp identically to minmax's forward division (see zscore_inverse).
    span = (_align(stats["max"], sig) - _align(stats["min"], sig)).clamp_min(EPS)
    return sig * span + _align(stats["min"], sig)


def finite_stats(trace: Tensor) -> dict[str, Tensor]:
    """``_stats`` over the finite entries of ``trace`` only (deviation 4).

    All-NaN/inf input yields all-zero 0-dim stats (the absent-label
    convention); a single finite entry yields ``std == 0`` rather than the
    NaN that ``std(correction=1)`` would produce. On all-finite input this is
    bit-identical to ``_stats``.
    """
    finite = trace[torch.isfinite(trace)]
    if finite.numel() == 0:
        zero = trace.new_zeros(())
        return {name: zero.clone() for name in STAT_NAMES}
    std = finite.std(correction=1) if finite.numel() > 1 else trace.new_zeros(())
    return {
        "mean": finite.mean(),   # ()
        "std": std,              # ()
        "min": finite.amin(),    # ()
        "max": finite.amax(),    # ()
    }


def raw(trace: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """Leave a trace in its physical units; still report the same stats dict.

    The identity mode. Stats are emitted anyway so ``label_stats`` means the
    same thing for every signal — physical-unit descriptive statistics of the
    window — and so downstream code can report a raw signal's window mean or
    span without special-casing it.
    """
    return trace, _stats(trace)


def raw_inverse(sig: Tensor, stats: dict[str, Tensor]) -> Tensor:
    """Inverse of :func:`raw`: the signal is already in physical units."""
    return sig


#: Every label mode, and the exact inverse of each. The keys are the config
#: vocabulary; ``resolve_label_norms`` validates against them.
INVERSES = {"raw": raw_inverse, "zscore": zscore_inverse, "minmax": minmax_inverse}

#: Mode names, in the order they are documented.
NORM_MODES: tuple[str, ...] = ("raw", "zscore", "minmax")


def apply_norm(trace: Tensor, stats: dict[str, Tensor], mode: str) -> Tensor:
    """Forward normalisation using precomputed ``stats`` — no recomputation.

    The dataset normalises with finite-only stats and emits those same stats,
    so the inverse round-trip is exact at finite positions (deviation 4).
    """
    if mode == "raw":
        return trace
    if mode == "zscore":
        return (trace - stats["mean"]) / stats["std"].clamp_min(EPS)
    if mode == "minmax":
        span = (stats["max"] - stats["min"]).clamp_min(EPS)
        return (trace - stats["min"]) / span
    raise ValueError(f"mode must be one of {list(NORM_MODES)}, got {mode!r}")


def default_label_norm(signal: str) -> str:
    """The mode a signal gets when the config says nothing: its class decides.

    Absolute-class signals (ABP, CVP) stay in physical units; everything else
    is per-window z-scored, which is what the pipeline always did.
    """
    return "raw" if signal_class(signal) == ABSOLUTE else "zscore"


def resolve_label_norms(traces, overrides=None) -> dict[str, str]:
    """``{signal: mode}`` for a trace list, applying config overrides.

    ``overrides`` is the config's per-signal mapping (any signal spelling the
    registry accepts); an entry naming a signal that is not being predicted is
    an error rather than a silently ignored typo, since the whole point of the
    key is to control a trace the run actually has.
    """
    resolved = {sig: default_label_norm(sig) for sig in traces}
    for name, mode in dict(overrides or {}).items():
        signal = canonical_signal(name)
        if signal not in resolved:
            raise ValueError(
                f"LABEL_NORM names {name!r}, which is not in TRACES {list(traces)}"
            )
        if mode not in INVERSES:
            raise ValueError(
                f"LABEL_NORM for {signal} is {mode!r}; known: {list(NORM_MODES)}"
            )
        resolved[signal] = mode
    return resolved
