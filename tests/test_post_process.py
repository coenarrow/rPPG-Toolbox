"""Detrending and MACC: the rewritten hot paths must match the originals exactly.

``_detrend`` moved from a dense O(n^3) inverse to a banded solve and
``_compute_macc`` from a per-lag Python loop to one FFT. Both are supposed to
be identities, not approximations, so both are pinned here against the
straightforward implementations they replaced.
"""
import numpy as np
import pytest
from scipy.sparse import spdiags

from src.evaluation.post_process import (
    _calculate_fft_hr, _compute_macc, _detrend, calculate_metric_per_video,
)
from unsupervised_methods import utils


def detrend_dense(input_signal, lambda_value):
    """The original textbook form: build the full matrix and invert it."""
    n = input_signal.shape[0]
    H = np.identity(n)
    diags_data = np.array([np.ones(n), -2 * np.ones(n), np.ones(n)])
    D = spdiags(diags_data, np.array([0, 1, 2]), (n - 2), n).toarray()
    return np.dot(H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D)),
                  input_signal)


def macc_loop(pred, gt):
    """The original per-lag loop."""
    pred, gt = np.squeeze(np.asarray(pred)), np.squeeze(np.asarray(gt))
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    return max(abs(np.corrcoef(pred, np.roll(gt, lag))[0][1])
               for lag in range(0, len(pred) - 1))


@pytest.mark.parametrize("n", [12, 51, 180, 300])
@pytest.mark.parametrize("lambda_value", [10, 100])
def test_detrend_matches_the_dense_inverse(n, lambda_value):
    rng = np.random.default_rng(n)
    signal = rng.normal(size=n) * 5 + np.linspace(0, 20, n)   # noise on a ramp
    assert np.allclose(_detrend(signal, lambda_value),
                       detrend_dense(signal, lambda_value), atol=1e-9)


def test_detrend_preserves_2d_column_shape():
    """POS used to hand this a (N, 1) column; the shape must survive."""
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(120, 1))
    out = _detrend(signal, 100)
    assert out.shape == (120, 1)
    assert np.allclose(out, detrend_dense(signal, 100), atol=1e-9)


def test_detrend_removes_a_linear_trend():
    t = np.arange(200)
    trend = 0.5 * t + 3.0
    oscillation = np.sin(2 * np.pi * 1.2 * t / 30)
    out = _detrend(trend + oscillation, 100)
    assert abs(out.mean()) < 0.05
    assert np.corrcoef(out, oscillation)[0, 1] > 0.9


def test_detrend_passes_through_signals_too_short_to_difference():
    for n in (0, 1, 2):
        assert _detrend(np.ones(n), 100).shape == (n,)


def test_unsupervised_utils_detrend_is_the_same_function():
    rng = np.random.default_rng(1)
    signal = rng.normal(size=64)
    assert np.array_equal(utils.detrend(signal, 100), _detrend(signal, 100))


@pytest.mark.parametrize("n", [13, 64, 181, 300])
def test_macc_matches_the_per_lag_loop(n):
    rng = np.random.default_rng(n)
    pred = rng.normal(size=n)
    gt = np.roll(pred, 7) + rng.normal(size=n) * 0.2
    assert _compute_macc(pred, gt) == pytest.approx(macc_loop(pred, gt), abs=1e-12)


def test_macc_truncates_to_the_shorter_signal():
    rng = np.random.default_rng(2)
    pred, gt = rng.normal(size=200), rng.normal(size=150)
    assert _compute_macc(pred, gt) == pytest.approx(macc_loop(pred, gt), abs=1e-12)


def test_macc_of_a_constant_signal_is_zero_not_nan():
    rng = np.random.default_rng(3)
    assert _compute_macc(np.ones(50), rng.normal(size=50)) == 0.0
    assert _compute_macc(np.ones(2), np.ones(2)) == 0.0


def test_macc_of_a_shifted_copy_is_one():
    t = np.arange(300) / 30
    wave = np.sin(2 * np.pi * 1.2 * t)
    assert _compute_macc(wave, np.roll(wave, 11)) == pytest.approx(1.0, abs=1e-6)


def test_calculate_metric_per_video_recovers_a_known_rate():
    t = np.arange(600) / 30
    wave = np.sin(2 * np.pi * 1.2 * t)
    gt_hr, pred_hr, snr, macc = calculate_metric_per_video(
        wave, wave, diff_flag=False, fs=30, hr_method='FFT')
    assert gt_hr == pytest.approx(72, abs=2)
    assert pred_hr == pytest.approx(72, abs=2)
    assert macc == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(snr)


def test_fft_hr_of_a_pure_tone():
    t = np.arange(900) / 30
    assert _calculate_fft_hr(np.sin(2 * np.pi * 1.5 * t), fs=30) == pytest.approx(90, abs=1)
