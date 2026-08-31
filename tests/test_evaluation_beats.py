"""The beat layer, against a synthetic arterial waveform of known shape."""
import numpy as np
import pytest

from evaluation.beats import BEAT_STATS, beat_intervals, beat_stats, find_peaks
from neural_methods.signals import beat_labels

FS = 30.0
BEATS_PER_SECOND = 1.2
SECONDS = 10.0
SYSTOLIC, DIASTOLIC = 120.0, 80.0


def synthetic_abp():
    """A clean pulsatile trace peaking at 120 and troughing at 80 mmHg."""
    t = np.arange(int(FS * SECONDS)) / FS
    wave = -np.cos(2 * np.pi * BEATS_PER_SECOND * t)      # starts at a trough
    midpoint, amplitude = (SYSTOLIC + DIASTOLIC) / 2, (SYSTOLIC - DIASTOLIC) / 2
    return midpoint + amplitude * wave


def test_find_peaks_recovers_the_pulse_rate():
    trace = synthetic_abp()
    idx, values = find_peaks(trace, kind="max", width=int(FS * 2 / 3))
    assert len(idx) == pytest.approx(BEATS_PER_SECOND * SECONDS, abs=1)
    assert values.max() == pytest.approx(SYSTOLIC, abs=1.0)


def test_even_width_is_made_odd_before_pooling():
    """The prototype incremented width after using it, so the fix must bite."""
    trace = synthetic_abp()
    assert np.array_equal(find_peaks(trace, width=20)[0],
                          find_peaks(trace, width=21)[0])


def test_beat_stats_recover_systolic_and_diastolic():
    trace = synthetic_abp()
    intervals = beat_intervals(trace, FS)
    assert len(intervals) >= 8
    stats = beat_stats(trace, intervals)
    assert set(stats) == set(BEAT_STATS)
    assert stats["max"].mean() == pytest.approx(SYSTOLIC, abs=2.0)
    assert stats["min"].mean() == pytest.approx(DIASTOLIC, abs=2.0)
    assert stats["mean"].mean() == pytest.approx((SYSTOLIC + DIASTOLIC) / 2, abs=2.0)


def test_beat_labels_are_per_signal():
    assert beat_labels("ABP")["max"] == "systolic"
    assert beat_labels("CVP")["max"] == "peak"
    assert beat_labels("CVP")["mean"] == "mean"
