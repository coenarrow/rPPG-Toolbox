"""One pure function of the evaluation: the beat detector. The chain test in
test_scripts.py covers the rest."""
import numpy as np

from src.evaluation.beats import beat_levels, detect_beats, match_beats


def test_beats_of_a_sine_are_found_levelled_and_matched():
    fs, hz = 50.0, 1.2                       # 72 bpm
    t = np.arange(20 * fs) / fs
    abp = 100 + 20 * np.sin(2 * np.pi * hz * t)
    peaks = detect_beats(abp, fs, "ABP")
    assert abs(peaks.size - 24) <= 1         # 20 s at 1.2 Hz, an edge beat may go
    np.testing.assert_allclose(np.diff(peaks), fs / hz, atol=2)
    levels = beat_levels(abp, peaks, fs, 1)
    np.testing.assert_allclose(levels[:, 0], 120, atol=0.5)     # systolic
    np.testing.assert_allclose(levels[1:-1, 1], 100, atol=1)    # MAP
    np.testing.assert_allclose(levels[:, 2], 80, atol=0.5)      # diastolic
    match = match_beats(peaks, peaks + 3, fs)
    assert (match == np.arange(peaks.size)).all()
    assert (match_beats(peaks, np.array([], dtype=int), fs) == -1).all()
