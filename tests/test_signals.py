import numpy as np
import pytest

from neural_methods import signals as S


def test_registry_contents():
    assert S.CHANNELS == ('R', 'G', 'B', 'I', 'D', 'Y', 'T')
    assert set(S.SIGNALS) == {'PPG', 'ECG', 'ABP', 'CVP', 'RESP', 'EDA', 'SPO2'}
    assert S.EVAL_ONLY == ('HR',)
    for sig, meta in S.SIGNALS.items():
        lo, hi = meta['norm']
        assert lo < hi


def test_canonical_signal_aliases():
    assert S.canonical_signal('BVP') == 'PPG'
    assert S.canonical_signal('bvp') == 'PPG'
    assert S.canonical_signal('Pulse') == 'PPG'
    assert S.canonical_signal('abp') == 'ABP'
    with pytest.raises(KeyError):
        S.canonical_signal('THERMISTOR')


def test_validate_traces():
    assert S.validate_traces(['abp', 'BVP']) == ['ABP', 'PPG']
    with pytest.raises(ValueError):
        S.validate_traces(['HR'])          # eval-only is not a training trace
    with pytest.raises(ValueError):
        S.validate_traces([])


def test_validate_channels():
    assert S.validate_channels(['R', 'G', 'B']) == ['R', 'G', 'B']
    with pytest.raises(ValueError):
        S.validate_channels(['R', 'X'])
    with pytest.raises(ValueError):
        S.validate_channels([])


def test_normalize_roundtrip_and_clip():
    x = np.array([-100.0, 0.0, 100.0, 300.0])   # ABP norm (0, 200)
    n = S.normalize_signal(x, 'ABP')
    assert n.min() >= -1.0 and n.max() <= 1.0
    assert n[0] == -1.0        # clipped low
    assert n[3] == 1.0         # clipped high
    assert n[2] == 0.0         # midpoint of (0,200)
    d = S.denormalize_signal(n, 'ABP')
    assert np.allclose(d, [0.0, 0.0, 100.0, 200.0])


def test_norm_override():
    lo, hi = S.norm_range('ABP', overrides={'ABP': (0.0, 100.0)})
    assert (lo, hi) == (0.0, 100.0)
    lo, hi = S.norm_range('CVP', overrides={'ABP': (0.0, 100.0)})
    assert (lo, hi) == S.SIGNALS['CVP']['norm']


def test_modality_and_trace_vocabularies():
    # Every pinned modality's channels are canonical channel names.
    for modality, channels in S.MODALITY_CHANNELS.items():
        if channels is not None:
            assert S.validate_channels(list(channels)) == list(channels)
    assert S.MODALITY_CHANNELS['rgb'] == ('R', 'G', 'B')
    assert S.MODALITY_CHANNELS['ev'] is None            # not yet pinned
    assert 'Y' in S.CHANNELS and 'T' in S.CHANNELS      # grayscale, thermal
    # Every cache trace key maps to a known canonical signal.
    for cache_key, signal in S.TRACE_KEYS.items():
        assert S.canonical_signal(signal) == signal
    assert S.TRACE_KEYS['rr'] == 'RESP'
