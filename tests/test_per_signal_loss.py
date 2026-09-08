"""The per-signal composite loss: the weights, the masking, the components.

Replaces test_masked_loss.py. The masking assertions are the same ones — that
structure is retained verbatim from ``MaskedMultiSignalLoss`` and is the part a
regression would silently poison (an absent signal contributing NaN instead of
0 poisons the whole batch, not just its own term).
"""
import pytest
import torch

from neural_methods.loss.PerSignalLoss import (
    PerSignalLoss, ccc, mean_l1, negpearson, normalise_loss_weights,
    peak_max_l1, peak_min_l1, soft_peak_stat, weight_losses,
)

ABS = {'CCC': 1.0, 'MEAN': 0.05, 'MAX': 0.05, 'MIN': 0.05}
MSE = {'MSE': 1.0}


def _weigh(criterion, *args):
    """The old one-call `(total, breakdown)`, now the two contract v2 halves."""
    return weight_losses(criterion(*args), criterion.weights)


def _mk(B=4, T=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    preds = {'ABP': torch.randn(B, T, generator=g), 'CVP': torch.randn(B, T, generator=g)}
    labels = {'ABP': torch.randn(B, T, generator=g), 'CVP': torch.randn(B, T, generator=g)}
    return preds, labels


# --- the weights ----------------------------------------------------------
def test_weights_are_stated_outright_in_traces_order():
    weights = normalise_loss_weights(['ECG', 'ABP'],
                                     {'ABP': ABS, 'ECG': {'negpearson': 2.0}})
    assert list(weights) == ['ECG', 'ABP']
    assert weights['ABP'] == {'ccc': 1.0, 'mean': 0.05, 'max': 0.05, 'min': 0.05}
    assert weights['ECG'] == {'negpearson': 2.0}


@pytest.mark.parametrize("bad, match", [
    ({'ABP': ABS}, "missing \\['ECG'\\]"),                    # a trace without a loss
    ({'ABP': ABS, 'ECG': MSE, 'CVP': ABS}, "not in TRACES"),   # a loss without a trace
    ({'ABP': {'HUBER': 1.0}, 'ECG': MSE}, "unknown component"),
    ({'ABP': {'CCC': 0}, 'ECG': MSE}, "positive"),
    ({'ABP': {}, 'ECG': MSE}, "non-empty"),
])
def test_bad_weights_are_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        normalise_loss_weights(['ABP', 'ECG'], bad)


def test_spectral_without_a_rate_is_refused_at_construction():
    with pytest.raises(ValueError, match="spectral"):
        PerSignalLoss(['ECG'], {'ECG': {'SPECTRAL': 1.0}}, fs=None)


# --- components ----------------------------------------------------------
def test_every_component_reduces_per_sample():
    """(B,) out, not a scalar — that is what lets the masking compose."""
    pred, label = torch.randn(5, 40), torch.randn(5, 40)
    for fn in (ccc, mean_l1, negpearson, peak_max_l1, peak_min_l1):
        assert fn(pred, label).shape == (5,), fn.__name__


def test_ccc_is_zero_for_a_perfect_prediction_and_penalises_offset():
    label = torch.sin(torch.linspace(0, 12, 128)).expand(2, 128)
    assert torch.allclose(ccc(label, label), torch.zeros(2), atol=1e-5)
    # Same shape, wrong level: correlation would not notice, CCC does.
    assert (ccc(label + 10.0, label) > 0.9).all()


def test_negpearson_ignores_the_level_ccc_catches():
    label = torch.sin(torch.linspace(0, 12, 128)).expand(2, 128)
    assert torch.allclose(negpearson(label + 10.0, label), torch.zeros(2), atol=1e-5)


def test_soft_peaks_track_a_synthetic_pressure_wave_in_mmHg():
    """Systolic ~120, diastolic ~80: the soft statistics land near the truth."""
    t = torch.linspace(0, 8 * 3.14159, 512)
    wave = (100 + 20 * torch.sin(t)).expand(1, 512)
    scale = (wave.amax(-1) - wave.amin(-1)).reshape(1, 1)
    assert soft_peak_stat(wave, scale, 'max').item() == pytest.approx(120, abs=2.0)
    assert soft_peak_stat(wave, scale, 'min').item() == pytest.approx(80, abs=2.0)


def test_peak_terms_are_differentiable_in_raw_units():
    """The softness scales with the window's range, so mmHg does not saturate it."""
    t = torch.linspace(0, 8 * 3.14159, 256)
    label = (100 + 20 * torch.sin(t)).expand(2, 256)
    # Deliberately off the minimum: |x| has subgradient 0 exactly at 0, so a
    # perfect prediction would show no gradient for reasons that say nothing
    # about saturation.
    pred = (95 + 25 * torch.sin(t)).expand(2, 256).clone().requires_grad_(True)
    loss = peak_max_l1(pred, label).sum() + peak_min_l1(pred, label).sum()
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


# --- masking (retained from MaskedMultiSignalLoss) -----------------------
def test_hand_computed_masked_mse():
    loss_fn = PerSignalLoss(['ABP', 'CVP'], {'ABP': MSE, 'CVP': MSE})
    preds = {'ABP': torch.zeros(2, 4), 'CVP': torch.zeros(2, 4)}
    labels = {'ABP': torch.ones(2, 4), 'CVP': torch.full((2, 4), 2.0)}
    mask = {'ABP': torch.tensor([1.0, 1.0]), 'CVP': torch.tensor([1.0, 0.0])}
    # ABP: per-sample MSE = 1.0, both present -> 1.0
    # CVP: per-sample MSE = 4.0, one present -> 4.0
    total, weighted = _weigh(loss_fn, preds, labels, mask)
    assert torch.isclose(total, torch.tensor((1.0 + 4.0) / 2))
    assert weighted['CVP']['mse'] == pytest.approx(4.0)


def test_fully_masked_signal_contributes_zero_no_nan():
    loss_fn = PerSignalLoss(['ABP', 'CVP'], {'ABP': MSE, 'CVP': MSE})
    preds, labels = _mk()
    mask = {'ABP': torch.ones(4), 'CVP': torch.zeros(4)}
    total, weighted = _weigh(loss_fn, preds, labels, mask)
    assert torch.isfinite(total)
    assert weighted['CVP']['total'] == 0.0
    abp_only = PerSignalLoss(['ABP'], {'ABP': MSE})
    only_abp, _ = _weigh(abp_only, {'ABP': preds['ABP']}, {'ABP': labels['ABP']},
                         {'ABP': mask['ABP']})
    assert torch.isclose(total, only_abp / 2)


def test_absolute_weights_backpropagate_through_every_component():
    loss_fn = PerSignalLoss(['ABP'], {'ABP': ABS}, fs=30)
    pred = torch.randn(3, 64, requires_grad=True) * 10 + 90
    pred.retain_grad()
    labels = {'ABP': torch.randn(3, 64) * 10 + 90}
    raw = loss_fn({'ABP': pred}, labels, {'ABP': torch.ones(3)})
    total, weighted = weight_losses(raw, loss_fn.weights)
    total.backward()
    # The raw dict is components only; the 'total' lives on the weighted side.
    assert set(raw['ABP']) == {'ccc', 'mean', 'max', 'min'}
    assert set(weighted['ABP']) == {'ccc', 'mean', 'max', 'min', 'total'}
    assert torch.isfinite(total) and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_breakdown_keys_every_signal_even_when_absent():
    """The training curves need a row per signal per epoch, present or not."""
    loss_fn = PerSignalLoss(['ABP', 'ECG'], {'ABP': ABS, 'ECG': {'NEGPEARSON': 1.0}})
    preds = {'ABP': torch.randn(2, 32), 'ECG': torch.randn(2, 32)}
    labels = {'ABP': torch.randn(2, 32), 'ECG': torch.randn(2, 32)}
    raw = loss_fn(preds, labels, {'ABP': torch.ones(2), 'ECG': torch.zeros(2)})
    assert set(raw) == {'ABP', 'ECG'}
    assert float(raw['ECG']['negpearson']) == 0.0


def test_raw_is_unweighted_and_weighting_is_separate():
    criterion = PerSignalLoss(["ABP"], {"ABP": {"CCC": 2.0}})
    preds = {"ABP": torch.randn(4, 32, requires_grad=True)}
    labels = {"ABP": torch.randn(4, 32)}
    mask = {"ABP": torch.ones(4, dtype=torch.bool)}
    raw = criterion(preds, labels, mask)
    assert set(raw) == {"ABP"} and "total" not in raw["ABP"]
    assert raw["ABP"]["ccc"].requires_grad
    total, weighted = weight_losses(raw, criterion.weights)
    assert torch.isclose(total, 2.0 * raw["ABP"]["ccc"])
    assert weighted["ABP"]["total"] == float(total.detach())
