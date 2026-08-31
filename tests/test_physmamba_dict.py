"""PhysMamba on the batch-dict contract: shapes, keys, masks, gradients."""
import pytest
import torch
from torch.utils.data import default_collate

from neural_methods.batch import LABEL_MASK, LABELS, PREDICTIONS
from neural_methods.frame_transforms import FrameTransform
from neural_methods.loss.PerSignalLoss import PerSignalLoss
from neural_methods.model.PhysMamba import PhysMamba
from tests.test_batch_contract import make_sample

# Small enough to stay quick on CPU: PhysMamba halves the spatial size four
# times, so 32x32 is the practical floor.
SIZE = (32, 32)


def build(channels=("R", "G", "B"), traces=("ABP", "CVP"), data_types=("DiffNormalized",)):
    return PhysMamba(channels=channels, traces=traces,
                     frame_transform=FrameTransform(data_types, size=SIZE))


def batch_of(n=2, channels=("R", "G", "B"), signals=("ABP", "CVP"), t=32, hw=(64, 64)):
    return default_collate([make_sample(channels=channels, signals=signals, t=t, hw=hw)
                            for _ in range(n)])


def test_dict_in_dict_out_keeps_everything():
    model = build()
    batch = batch_of()
    out = model(batch)
    assert set(out) == set(batch) | {PREDICTIONS}
    for key in batch:
        assert out[key] is batch[key], f"{key} should pass through untouched"


def test_predictions_are_keyed_by_signal_with_window_length():
    model = build(traces=("ABP", "CVP", "ECG"))
    batch = batch_of(n=2, signals=("ABP", "CVP", "ECG"), t=32)
    predictions = model(batch)[PREDICTIONS]
    assert list(predictions) == ["ABP", "CVP", "ECG"]
    for signal, trace in predictions.items():
        assert trace.shape == (2, 32), signal
        assert torch.isfinite(trace).all()


def test_channel_count_follows_the_configured_channels():
    model = build(channels=("R", "G", "B", "I", "D"))
    assert model.in_channels == 5
    assert model.ConvBlock1[0].in_channels == 5
    batch = batch_of(channels=("R", "G", "B", "I", "D"))
    assert model(batch)[PREDICTIONS]["ABP"].shape == (2, 32)


def test_two_data_type_blocks_double_the_input_width():
    model = build(data_types=("DiffNormalized", "Standardized"))
    assert model.in_channels == 6
    assert model.ConvBlock1[0].in_channels == 6
    assert model(batch_of())[PREDICTIONS]["CVP"].shape == (2, 32)


def test_signal_head_width_follows_the_configured_traces():
    assert build(traces=("ABP",)).ConvBlockLast.out_channels == 1
    assert build(traces=("ABP", "CVP", "ECG")).ConvBlockLast.out_channels == 3


def test_window_length_is_not_baked_into_the_model():
    """One model must serve any CHUNK_LENGTH: the head pools space, not time."""
    model = build()
    for length in (16, 32, 48):
        out = model(batch_of(n=1, t=length))
        assert out[PREDICTIONS]["ABP"].shape == (1, length)


def test_legacy_tensor_path_returns_a_plain_tensor():
    single = build(traces=("PPG",))
    assert single(torch.randn(1, 3, 32, 32, 32)).shape == (1, 32)
    multi = build(traces=("ABP", "CVP"))
    assert multi(torch.randn(1, 3, 32, 32, 32)).shape == (1, 2, 32)


def test_forward_rejects_a_non_batch_object():
    with pytest.raises(TypeError, match="frames"):
        build()({"labels": {}})


def test_gradients_reach_the_stem_from_every_signal():
    model = build(traces=("ABP", "CVP"))
    batch = batch_of(n=1, t=32)
    for signal in ("ABP", "CVP"):
        model.zero_grad(set_to_none=True)
        model(batch)[PREDICTIONS][signal].sum().backward()
        stem_grad = model.ConvBlock1[0].weight.grad
        assert stem_grad is not None and stem_grad.abs().sum() > 0, signal


def test_masked_loss_ignores_absent_labels():
    """A signal absent from every sample must contribute exactly 0, not NaN."""
    model = build(traces=("ABP", "CVP"))
    batch = batch_of(n=2, t=32)
    out = model(batch)
    labels = dict(batch[LABELS])
    masks = {"ABP": torch.tensor([True, True]), "CVP": torch.tensor([False, False])}
    loss, _ = PerSignalLoss(("ABP", "CVP"))(out[PREDICTIONS], labels, masks)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.ConvBlock1[0].weight.grad).all()


def test_absent_channels_are_zero_filled_not_missing():
    """The loader always emits every configured channel; masks say which are real."""
    model = build(channels=("R", "G", "B", "I"))
    batch = batch_of(channels=("R", "G", "B", "I"))
    batch["frames"]["I"] = torch.zeros_like(batch["frames"]["I"])
    batch["channel_mask"]["I"] = torch.tensor([False, False])
    out = model(batch)
    assert torch.isfinite(out[PREDICTIONS]["ABP"]).all()
    assert bool(out["channel_mask"]["I"][0]) is False
