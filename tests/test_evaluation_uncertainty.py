"""Autocorrelation-aware uncertainty: the naive SE is the one that lies."""
import numpy as np
import pytest

from evaluation.uncertainty import mean_se, moving_block_bootstrap


def ar1(rho=0.9, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    series, value = np.empty(n), 0.0
    for i in range(n):
        value = rho * value + rng.standard_normal()
        series[i] = value
    return series


def test_hac_se_exceeds_the_naive_se_on_autocorrelated_data():
    series = ar1()
    _, _, naive = mean_se(series, method="naive")
    _, _, hac = mean_se(series, method="HAC")
    assert hac > 2 * naive


def test_naive_and_hac_agree_on_white_noise():
    series = np.random.default_rng(1).standard_normal(2000)
    _, _, naive = mean_se(series, method="naive")
    _, _, hac = mean_se(series, method="HAC")
    assert hac == pytest.approx(naive, rel=0.5)


def test_bootstrap_is_deterministic_under_a_seed():
    pred, label = ar1(seed=2), ar1(seed=3)
    def pearson(a, b):
        return float(np.corrcoef(a, b)[0, 1])
    first = moving_block_bootstrap(pearson, pred, label, resamples=64, seed=7)
    second = moving_block_bootstrap(pearson, pred, label, resamples=64, seed=7)
    assert first == second and first > 0
