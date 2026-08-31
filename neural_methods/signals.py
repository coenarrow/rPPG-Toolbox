"""Canonical physiological signal and camera-channel registries.

Single source of truth for the multi-signal pipeline (see
docs/architecture.md). Signal names cover exactly what this repo's
dataloaders provide.
"""
import numpy as np

CHANNELS = ('R', 'G', 'B', 'I', 'D')

#: The two signal classes the migration contract distinguishes (§3).
#:
#: ``absolute`` signals carry meaning in their physical units, so they are fed
#: to the model un-normalised and scored on level as well as shape.
#: ``shape`` signals are per-window normalised and only their waveform matters.
#: The class is what decides a signal's default label normalisation and its
#: default loss family; nothing else keys off it.
ABSOLUTE, SHAPE = 'absolute', 'shape'

#: Per signal: the legacy clip range, its class, its physical unit, the
#: physiological prior an absolute-class output bias is initialised to, and
#: ``scale`` — the error magnitude (in that unit) an L1 loss component is
#: divided by, which is where the per-signal scale factor of a multi-signal
#: objective lives (contract §3: no global normalisation constants).
SIGNALS = {
    'PPG':  {'norm': (-3.0, 3.0),        # a.k.a. BVP; standardized units
             'class': SHAPE,    'unit': 'a.u.', 'prior': 0.0,  'scale': 1.0},
    'ECG':  {'norm': (-1500.0, 1500.0),
             'class': SHAPE,    'unit': 'uV',   'prior': 0.0,  'scale': 1.0},
    'ABP':  {'norm': (0.0, 200.0),
             'class': ABSOLUTE, 'unit': 'mmHg', 'prior': 90.0, 'scale': 20.0,
             'beat_labels': {'max': 'systolic', 'mean': 'MAP', 'min': 'diastolic'}},
    'CVP':  {'norm': (-20.0, 30.0),
             'class': ABSOLUTE, 'unit': 'mmHg', 'prior': 8.0,  'scale': 5.0,
             # CVP has no systole: its waveform is a/c/v waves, and the
             # quantity that matters clinically is the mean. The machinery is
             # shared with ABP; only the wording differs.
             'beat_labels': {'max': 'peak', 'mean': 'mean', 'min': 'trough'}},
    'RESP': {'norm': (0.0, 10.0),        # BP4D Resp_Volts scale; override per dataset
             'class': SHAPE,    'unit': 'V',    'prior': 0.0,  'scale': 1.0},
    'EDA':  {'norm': (0.0, 40.0),        # microsiemens; override per dataset
             'class': SHAPE,    'unit': 'uS',   'prior': 0.0,  'scale': 1.0},
    'SPO2': {'norm': (0.0, 100.0),
             'class': ABSOLUTE, 'unit': '%',    'prior': 97.0, 'scale': 3.0,
             'beat_labels': {'max': 'max', 'mean': 'mean', 'min': 'min'}},
}

EVAL_ONLY = ('HR',)

_ALIASES = {'BVP': 'PPG', 'PULSE': 'PPG'}


def canonical_signal(name):
    """Map any loader-side name to the canonical vocabulary (KeyError if unknown)."""
    up = str(name).upper()
    up = _ALIASES.get(up, up)
    if up in SIGNALS or up in EVAL_ONLY:
        return up
    raise KeyError(
        f"Unknown signal {name!r}; known: {sorted(SIGNALS)} + eval-only {list(EVAL_ONLY)}")


def validate_traces(traces):
    """Canonicalize a config TRACES list; reject empty, unknown, or eval-only."""
    if not traces:
        raise ValueError("TRACES must name at least one signal")
    out = []
    for t in traces:
        c = canonical_signal(t)
        if c in EVAL_ONLY:
            raise ValueError(f"{c} is eval-only and cannot be a training trace")
        out.append(c)
    return out


def validate_channels(channels):
    """Validate a config CHANNELS list against the canonical slots."""
    if not channels:
        raise ValueError("CHANNELS must name at least one channel")
    bad = [c for c in channels if c not in CHANNELS]
    if bad:
        raise ValueError(f"Unknown channels {bad}; known: {list(CHANNELS)}")
    return list(channels)


def signal_class(sig) -> str:
    """``'absolute'`` or ``'shape'`` for a canonical signal name."""
    return SIGNALS[canonical_signal(sig)]['class']


def is_absolute(sig) -> bool:
    """True for signals whose physical level is part of the prediction."""
    return signal_class(sig) == ABSOLUTE


def signal_unit(sig) -> str:
    """Physical unit a signal's ``label_stats`` (and raw predictions) are in."""
    return SIGNALS[canonical_signal(sig)]['unit']


def signal_prior(sig) -> float:
    """Physiological prior an absolute-class output bias starts at.

    Zero for shape-class signals, whose labels are per-window centred anyway.
    """
    return float(SIGNALS[canonical_signal(sig)]['prior'])


def signal_scale(sig) -> float:
    """Typical error magnitude, in the signal's own unit.

    An L1 loss component in physical units is weighted by ``1 / scale`` so that
    ABP (errors O(10 mmHg)), CVP (O(1 mmHg)) and a dimensionless CCC term all
    reach the optimiser at the same order of magnitude.
    """
    return float(SIGNALS[canonical_signal(sig)]['scale'])


def norm_range(sig, overrides=None):
    if overrides and sig in overrides:
        lo, hi = overrides[sig]
    else:
        lo, hi = SIGNALS[sig]['norm']
    return float(lo), float(hi)


def normalize_signal(x, sig, overrides=None):
    """Clip to the signal's range then min-max to [-1, 1]."""
    lo, hi = norm_range(sig, overrides)
    clipped = np.clip(x, lo, hi)
    return (clipped - lo) / (hi - lo) * 2.0 - 1.0


def denormalize_signal(x, sig, overrides=None):
    lo, hi = norm_range(sig, overrides)
    return (np.asarray(x) + 1.0) / 2.0 * (hi - lo) + lo


def beat_labels(sig) -> dict:
    """How this signal's per-beat max/mean/min are named in a report.

    Shape-class signals get no beat treatment, so they fall back to the plain
    words rather than borrowing arterial vocabulary.
    """
    entry = SIGNALS[canonical_signal(sig)]
    return dict(entry.get('beat_labels',
                          {'max': 'max', 'mean': 'mean', 'min': 'min'}))


