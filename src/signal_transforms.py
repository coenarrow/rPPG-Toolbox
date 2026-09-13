"""The signal registry, the cache vocabularies, and the label normalisation.

One module for everything that is *about a signal*: its canonical name and
aliases, which camera channels and cache groups exist, what class a signal
belongs to and what that implies, and how a window of its label is
normalised for the model and mapped back to physical units afterwards.

**Registry.** ``SIGNALS`` is the table; the accessors below read one column
of it for a canonical name. Two classes are distinguished: an ``absolute``
signal (ABP, CVP, SpO2) carries meaning in its physical units, so it is fed to
the model raw and scored on level as well as shape; a ``shape`` signal (PPG,
ECG, respiration) is per-window normalised and only its waveform matters.

**Label normalisation.** Every mode in ``LABEL_TRANSFORMS`` shares one
signature — ``(trace, stats) -> trace`` — and an exact inverse, both keyed
by the mode's name in the interface's ``LABEL_PREPROCESSING``. The dataset
computes ``finite_stats`` of the window once, normalises with them, and
stamps them into the batch as ``label_stats`` in physical units; the trainer
inverts predictions and labels with the same stats, so the round trip is
exact. The stats broadcast against a per-sample ``(T,)`` trace and a
collated ``(B, T)`` batch alike.
"""

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Vocabularies: channels, cache modalities, cache traces
# ---------------------------------------------------------------------------
CHANNELS = ("R", "G", "B", "I", "D", "Y", "T")

#: Cache contract (``docs/cache-contract.md``): modality group name -> the
#: canonical channels its video planes carry, in stacking order. ``None`` =
#: frame representation not yet pinned (validated loosely).
MODALITY_CHANNELS = {
    "gr":    ("Y",),
    "rgb":   ("R", "G", "B"),
    "ir":    ("I",),
    "depth": ("D",),
    "t":     ("T",),
    "ev":    None,
}

#: Cache trace group name -> canonical signal name, and back.
TRACE_KEYS = {"ecg": "ECG", "abp": "ABP", "cvp": "CVP", "ppg": "PPG", "rr": "RESP"}
TRACE_KEYS_INVERSE = {signal: key for key, signal in TRACE_KEYS.items()}


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
ABSOLUTE, SHAPE = "absolute", "shape"

#: Per signal: its class, its physical unit, the physiological level an
#: absolute-class readout's bias starts at, whether the trace beats with the
#: heart so a heart rate can be read off it (``src/evaluation/rate.py``), and
#: for absolute signals how a report names the per-window max / mean / min.
#: ``beat`` is how the detector reads the trace (``src/evaluation/beats.py``).
SIGNALS = {
    "PPG":  {"class": SHAPE,    "unit": "a.u.", "prior": 0.0,  "cardiac": True,
             "beat": {"polarity": 1, "prominence": 0.3}},
    "ECG":  {"class": SHAPE,    "unit": "uV",   "prior": 0.0,  "cardiac": True,
             "beat": {"polarity": 1, "prominence": 0.3}},
    "ABP":  {"class": ABSOLUTE, "unit": "mmHg", "prior": 90.0, "cardiac": True,
             "beat": {"polarity": 1, "prominence": 0.3},
             "beat_labels": {"max": "systolic", "mean": "MAP", "min": "diastolic"}},
    # CVP has no systole: its waveform is a/c/v waves, and the quantity that
    # matters clinically is the mean. Same machinery as ABP, other words.
    # Its beats are detected with the same detector and a lower prominence;
    # readings.csv's n_ref_beats / n_pred_beats / n_matched say whether
    # that works.
    "CVP":  {"class": ABSOLUTE, "unit": "mmHg", "prior": 8.0,  "cardiac": True,
             "beat": {"polarity": 1, "prominence": 0.2},
             "beat_labels": {"max": "peak", "mean": "mean", "min": "trough"}},
    "RESP": {"class": SHAPE,    "unit": "V",    "prior": 0.0,  "cardiac": False},
    "EDA":  {"class": SHAPE,    "unit": "uS",   "prior": 0.0,  "cardiac": False},
    "SPO2": {"class": ABSOLUTE, "unit": "%",    "prior": 97.0, "cardiac": False,
             "beat_labels": {"max": "max", "mean": "mean", "min": "min"}},
}

_ALIASES = {"BVP": "PPG", "PULSE": "PPG"}


def canonical_signal(name) -> str:
    """Map any config-side spelling to the canonical name (KeyError if unknown)."""
    up = str(name).upper()
    up = _ALIASES.get(up, up)
    if up in SIGNALS:
        return up
    raise KeyError(f"Unknown signal {name!r}; known: {sorted(SIGNALS)}")


def validate_traces(traces) -> list:
    """Canonicalise a config TRACES list; reject empty or unknown."""
    if not traces:
        raise ValueError("TRACES must name at least one signal")
    return [canonical_signal(t) for t in traces]


def validate_channels(channels) -> list:
    """Validate a config CHANNELS list against the canonical slots."""
    if not channels:
        raise ValueError("CHANNELS must name at least one channel")
    bad = [c for c in channels if c not in CHANNELS]
    if bad:
        raise ValueError(f"Unknown channels {bad}; known: {list(CHANNELS)}")
    return list(channels)


def is_absolute(sig) -> bool:
    """True for signals whose physical level is part of the prediction."""
    return SIGNALS[canonical_signal(sig)]["class"] == ABSOLUTE


def is_cardiac(sig) -> bool:
    """True for traces that beat with the heart, so carry a heart rate."""
    return bool(SIGNALS[canonical_signal(sig)]["cardiac"])


def signal_unit(sig) -> str:
    """Physical unit a signal's ``label_stats`` (and raw predictions) are in."""
    return SIGNALS[canonical_signal(sig)]["unit"]


def signal_prior(sig) -> float:
    """Physiological level an absolute-class readout's bias starts at; zero
    for shape-class signals, whose labels are per-window centred anyway."""
    return float(SIGNALS[canonical_signal(sig)]["prior"])


def beat_labels(sig) -> dict:
    """How this signal's per-window max / mean / min are named in a report.
    Shape-class signals fall back to the plain words."""
    entry = SIGNALS[canonical_signal(sig)]
    return dict(entry.get("beat_labels", {"max": "max", "mean": "mean", "min": "min"}))


def beat_config(sig) -> dict:
    """How the beat detector reads this cardiac signal: ``polarity`` (+1 when
    a beat is a peak of the trace, -1 a trough) and ``prominence`` (the
    fraction of the cleaned trace's range a candidate must stand out by).
    KeyError for a signal that does not beat."""
    entry = SIGNALS[canonical_signal(sig)]
    if not entry["cardiac"]:
        raise KeyError(f"{sig} is not cardiac; it has no beats")
    return dict(entry["beat"])


# ---------------------------------------------------------------------------
# Label normalisation: one window of one trace, and its exact inverse
# ---------------------------------------------------------------------------
#: The stats every mode is computed from and stamps into ``label_stats``.
STAT_NAMES = ("mean", "std", "min", "max")

#: Numerical guard only: a constant trace yields 0/EPS == 0 instead of 0/0 ==
#: NaN. That matters because the masked loss computes ``values * mask`` and
#: NaN * 0 is still NaN, so a NaN could not be masked away after the fact.
EPS = 1e-8


def finite_stats(trace: Tensor) -> dict:
    """``STAT_NAMES`` over the finite entries of one ``(T,)`` trace, as 0-dim tensors.

    All-NaN/inf input yields all-zero stats (the absent-label convention); a
    single finite entry yields ``std == 0`` rather than the NaN an unbiased
    std would produce.
    """
    finite = trace[torch.isfinite(trace)]
    if finite.numel() == 0:
        return {name: trace.new_zeros(()) for name in STAT_NAMES}
    std = finite.std(correction=1) if finite.numel() > 1 else trace.new_zeros(())
    return {"mean": finite.mean(), "std": std, "min": finite.amin(), "max": finite.amax()}


def _align(stat: Tensor, sig: Tensor) -> Tensor:
    """Right-pad ``stat`` with singleton dims so a 0-dim stat broadcasts
    against a ``(T,)`` trace and a ``(B,)`` stat against a ``(B, T)`` batch."""
    return stat.reshape(stat.shape + (1,) * (sig.dim() - stat.dim()))


def _raw(trace, stats):
    return trace


def _zscore(trace, stats):
    return (trace - _align(stats["mean"], trace)) / _align(stats["std"], trace).clamp_min(EPS)


def _zscore_inverse(sig, stats):
    # Clamped identically to the forward division, or the round trip is
    # inexact in the 0 < std < EPS band.
    return sig * _align(stats["std"], sig).clamp_min(EPS) + _align(stats["mean"], sig)


def _minmax(trace, stats):
    span = (_align(stats["max"], trace) - _align(stats["min"], trace)).clamp_min(EPS)
    return (trace - _align(stats["min"], trace)) / span


def _minmax_inverse(sig, stats):
    span = (_align(stats["max"], sig) - _align(stats["min"], sig)).clamp_min(EPS)
    return sig * span + _align(stats["min"], sig)


#: ``LABEL_PREPROCESSING`` vocabulary: mode -> ``(forward, inverse)``, each
#: ``(trace, stats) -> trace``. ``raw`` leaves physical units untouched — the
#: mode for absolute-class signals; the other two are per-window.
LABEL_TRANSFORMS = {
    "raw": (_raw, _raw),
    "zscore": (_zscore, _zscore_inverse),
    "minmax": (_minmax, _minmax_inverse),
}


def normalise_label(trace: Tensor, stats: dict, mode: str) -> Tensor:
    """Forward normalisation with precomputed ``stats``, never recomputed."""
    return LABEL_TRANSFORMS[mode][0](trace, stats)


def denormalise_label(sig: Tensor, stats: dict, mode: str) -> Tensor:
    """The exact inverse: a normalised trace back in physical units."""
    return LABEL_TRANSFORMS[mode][1](sig, stats)
