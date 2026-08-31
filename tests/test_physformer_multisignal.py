"""PhysFormer as a multi-channel, multi-signal transformer behind the dict contract.

The fidelity claim (original exactly recoverable at 3-in/1-out) is not testable
from the tree — the pre-migration module lives only at ``git show
cf52990:neural_methods/model/PhysFormer.py``, and the retro records the
numerical check run against it. What is kept green here is the contract
surface: the two widened layers, the *derived* token grid, the declared
temporal constraint, and per-signal readout independence.

Every assertion below is meant to be discriminating — breaking the thing it
names must fail it. ``tests/test_batch_contract.py`` already covers the dict
plumbing, so none of that is re-tested per model.
"""
import pytest
import torch
from torch.utils.data import default_collate

from neural_methods.batch import LABELS, PREDICTIONS
from neural_methods.frame_transforms import FrameTransform
from neural_methods.model.PhysFormer import PhysFormer
from tests.test_batch_contract import make_sample

# Deliberately non-square: the grid is 8x stem pooling then 4x4 patches, so
# 64x32 tokenizes to 2x1. A square size would make a transposed grid invisible,
# and 32x32 (a 1x1 grid) would leave the grid untested altogether.
SIZE = (64, 32)
GRID = (2, 1)

# Small everywhere the published config is large: these are contract tests, not
# a training run.
SMALL = dict(dim=16, ff_dim=24, num_heads=2, num_layers=3)


def build(channels=("R", "G", "B"), traces=("ABP", "CVP"),
          data_types=("DiffNormalized",), size=SIZE, **kwargs):
    return PhysFormer(channels=channels, traces=traces,
                      frame_transform=FrameTransform(data_types, size=size),
                      image_size=(None, *size), **{**SMALL, **kwargs})


def batch_of(n=2, channels=("R", "G", "B"), signals=("ABP", "CVP"), t=8, hw=(40, 40)):
    return default_collate([make_sample(channels=channels, signals=signals, t=t, hw=hw)
                            for _ in range(n)])


def test_predictions_are_keyed_by_signal_with_window_length():
    model = build(traces=("ABP", "CVP", "ECG"))
    predictions = model(batch_of(signals=("ABP", "CVP", "ECG"), t=8))[PREDICTIONS]
    assert list(predictions) == ["ABP", "CVP", "ECG"]
    for signal, trace in predictions.items():
        assert trace.shape == (2, 8), signal
        assert torch.isfinite(trace).all(), signal


def test_only_the_first_layer_and_the_readout_change_width():
    model = build(channels=("R", "G", "B", "I", "D"), traces=("ABP", "CVP", "ECG"),
                  data_types=("DiffNormalized", "Standardized"))
    assert model.in_channels == 10                    # 5 channels x 2 DATA_TYPE blocks
    assert model.backbone.Stem0[0].in_channels == 10
    assert model.backbone.ConvBlockLast.out_channels == 3
    # ...and nothing between them: the second stem conv still takes dim//4.
    assert model.backbone.Stem1[0].in_channels == SMALL["dim"] // 4


def test_the_readout_is_activation_free():
    """Absolute-class signals are predicted in mmHg, so nothing may squash the
    output — which means asserting on the values, not on the layer's type."""
    model = build().eval()
    (readout,) = model.output_layers()
    assert isinstance(readout, torch.nn.Conv1d)
    with torch.no_grad():                             # what init_output_bias does
        readout.bias.copy_(torch.tensor([90.0, 8.0]))
    predictions = model(batch_of())[PREDICTIONS]
    # A tanh/sigmoid anywhere after the readout would cap this at 1.
    assert predictions["ABP"].abs().max() > 50
    assert predictions["ABP"].mean() > predictions["CVP"].mean()


def test_each_signal_gets_an_independent_readout_row():
    """Style A is S independent readouts; identical rows would mean a shared one."""
    model = build(traces=("ABP", "CVP", "ECG")).eval()
    predictions = model(batch_of(signals=("ABP", "CVP", "ECG")))[PREDICTIONS]
    assert (predictions["ABP"] - predictions["CVP"]).abs().max() > 0
    assert (predictions["CVP"] - predictions["ECG"]).abs().max() > 0


def test_the_token_grid_is_derived_and_a_mismatched_frame_size_is_refused():
    """The original hardcoded a 4x4 grid; a wrong one reshapes rather than raising."""
    model = build()
    assert model.backbone.grid == GRID
    # Straight at the backbone: the frame transform would otherwise resize this away.
    with pytest.raises(ValueError, match="token grid"):
        model.backbone(torch.randn(1, 3, 8, 128, 128), 2.0)
    with pytest.raises(ValueError, match=r"RESIZE\.H .*multiple of 32"):
        build(size=(20, 32))


def test_window_length_is_declared_not_silently_truncated():
    model = build().eval()
    assert model.temporal_divisor == 4
    for length in (4, 8, 12):
        assert model(batch_of(n=1, t=length))[PREDICTIONS]["ABP"].shape == (1, length)
    with pytest.raises(ValueError, match="multiple of 4"):
        model(batch_of(n=1, t=6))


def test_gradients_reach_the_stem_from_every_signal():
    model = build(traces=("ABP", "CVP"))
    for signal in ("ABP", "CVP"):
        model.zero_grad(set_to_none=True)
        model(batch_of(n=1))[PREDICTIONS][signal].sum().backward()
        stem_grad = model.backbone.Stem0[0].weight.grad
        assert stem_grad is not None and stem_grad.abs().sum() > 0, signal


def test_absent_labels_and_channels_stay_finite():
    """A signal no sample carries must not poison the loss; a channel the batch
    lacks is zero-filled by ``stack_frames``, not an error."""
    from neural_methods.loss.PerSignalLoss import PerSignalLoss

    model = build(channels=("R", "G", "B", "I"), traces=("ABP", "CVP"))
    batch = batch_of(channels=("R", "G", "B"))        # no I plane at all
    out = model(batch)
    loss, breakdown = PerSignalLoss(("ABP", "CVP"), fs=30)(
        out[PREDICTIONS], batch[LABELS],
        {"ABP": torch.tensor([True, True]), "CVP": torch.tensor([False, False])})
    assert torch.isfinite(loss)
    assert breakdown["CVP"]["total"] == 0.0            # absent, not NaN
    loss.backward()
    assert torch.isfinite(model.backbone.Stem0[0].weight.grad).all()
