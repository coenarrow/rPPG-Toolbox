import pytest
import torch

from src import signal_transforms as S


def test_registry_vocabularies():
    assert S.CHANNELS == ("R", "G", "B", "I", "D", "Y", "T")
    assert set(S.SIGNALS) == {"PPG", "ECG", "ABP", "CVP", "RESP", "EDA", "SPO2"}
    for modality, channels in S.MODALITY_CHANNELS.items():
        if channels is not None:
            assert S.validate_channels(list(channels)) == list(channels)
    for cache_key, signal in S.TRACE_KEYS.items():
        assert S.canonical_signal(signal) == signal
        assert S.TRACE_KEYS_INVERSE[signal] == cache_key


def test_canonical_signal_aliases():
    assert S.canonical_signal("BVP") == "PPG"
    assert S.canonical_signal("Pulse") == "PPG"
    assert S.canonical_signal("abp") == "ABP"
    with pytest.raises(KeyError):
        S.canonical_signal("THERMISTOR")


def test_validators():
    assert S.validate_traces(["abp", "BVP"]) == ["ABP", "PPG"]
    with pytest.raises(ValueError):
        S.validate_traces([])
    assert S.validate_channels(["R", "G", "B"]) == ["R", "G", "B"]
    with pytest.raises(ValueError):
        S.validate_channels(["R", "X"])


def _trace():
    torch.manual_seed(0)
    return 80.0 + 15.0 * torch.randn(64)


@pytest.mark.parametrize("mode", list(S.LABEL_TRANSFORMS))
def test_label_round_trip_exact(mode):
    t = _trace()
    stats = S.finite_stats(t)
    normed = S.normalise_label(t, stats, mode)
    assert torch.allclose(S.denormalise_label(normed, stats, mode), t, atol=1e-3)


def test_inverse_broadcasts_collated_batch():
    t = _trace()
    stats = S.finite_stats(t)
    normed = S.normalise_label(t, stats, "zscore")
    stats_b = {k: torch.stack([v, v]) for k, v in stats.items()}       # (2,)
    out = S.denormalise_label(torch.stack([normed, normed]), stats_b, "zscore")
    assert out.shape == (2, 64)
    assert torch.allclose(out[0], t, atol=1e-3)


def test_constant_trace_never_nan():
    t = torch.full((16,), 7.0)
    for mode in ("zscore", "minmax"):
        assert torch.equal(S.normalise_label(t, S.finite_stats(t), mode), torch.zeros(16))


def test_finite_stats_ignores_nans_and_survives_none():
    t = _trace()
    dirty = t.clone()
    dirty[3], dirty[40] = float("nan"), float("inf")
    keep = torch.isfinite(dirty)
    stats = S.finite_stats(dirty)
    assert torch.equal(stats["mean"], t[keep].mean())
    assert torch.equal(stats["std"], t[keep].std(correction=1))
    empty = S.finite_stats(torch.full((8,), float("nan")))
    assert all(v.item() == 0.0 and v.dim() == 0 for v in empty.values())
    single = torch.full((8,), float("nan"))
    single[2] = 42.0
    assert S.finite_stats(single)["std"].item() == 0.0
