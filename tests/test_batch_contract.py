"""The batch-dict contract: packing, splitting, traversal, frame transforms."""
import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from neural_methods import batch as bt
from neural_methods.frame_transforms import (
    FrameTransform, apply_data_types, diff_normalized, resize_video, standardized,
)


def make_sample(channels=("R", "G", "B"), signals=("ABP", "CVP"), t=6, hw=(4, 5)):
    """One dataset-shaped (un-collated) sample."""
    h, w = hw
    return {
        bt.FRAMES: {c: torch.rand(1, t, h, w) * 255 for c in channels},
        bt.LABELS: {s: torch.randn(t) for s in signals},
        bt.LABEL_STATS: {s: {k: torch.tensor(1.0) for k in ("mean", "std", "min", "max")}
                         for s in signals},
        bt.CHANNEL_MASK: {c: torch.tensor(True) for c in channels},
        bt.LABEL_MASK: {s: torch.tensor(s == "ABP") for s in signals},
        bt.METADATA: {bt.RECORDING_ID: "P001_S01_R1_0_D", bt.CAMERA_ID: "1",
                      bt.START_FRAME: 12, bt.ATTRS: {}},
    }


# --- recognition ---------------------------------------------------------
def test_is_batch_dict_only_for_frames_dicts():
    assert bt.is_batch_dict(make_sample())
    assert not bt.is_batch_dict((torch.zeros(1), torch.zeros(1)))
    assert not bt.is_batch_dict({"labels": {}})


def test_require_batch_dict_names_the_contract():
    with pytest.raises(TypeError, match="frames"):
        bt.require_batch_dict((torch.zeros(1),))


# --- stack / unstack -----------------------------------------------------
def test_stack_frames_orders_by_channel_list():
    batch = default_collate([make_sample(), make_sample()])
    video = bt.stack_frames(batch[bt.FRAMES], ["B", "R"])
    assert video.shape == (2, 2, 6, 4, 5)
    assert torch.equal(video[:, 0], batch[bt.FRAMES]["B"][:, 0])
    assert torch.equal(video[:, 1], batch[bt.FRAMES]["R"][:, 0])


def test_stack_frames_round_trips():
    batch = default_collate([make_sample()])
    channels = ["R", "G", "B"]
    restored = bt.unstack_frames(bt.stack_frames(batch[bt.FRAMES], channels), channels)
    for channel in channels:
        assert torch.equal(restored[channel], batch[bt.FRAMES][channel])


def test_stack_frames_zero_fills_a_channel_the_batch_lacks():
    """The model is the authority: an RGB+IR checkpoint runs on RGB-only data."""
    batch = default_collate([make_sample(channels=("R", "G"))])
    video = bt.stack_frames(batch[bt.FRAMES], ["R", "G", "B"])
    assert video.shape == (1, 3, 6, 4, 5)
    assert torch.equal(video[:, 0], batch[bt.FRAMES]["R"][:, 0])
    assert torch.equal(video[:, 2], torch.zeros_like(video[:, 2]))


def test_stack_frames_refuses_a_batch_with_none_of_the_channels():
    """Zero-filling everything would be a silently black clip, not degradation."""
    batch = default_collate([make_sample(channels=("R", "G"))])
    with pytest.raises(KeyError, match="none of the model's channels"):
        bt.stack_frames(batch[bt.FRAMES], ["I", "D"])


# --- split / stack signals ----------------------------------------------
def test_split_signals_keys_and_slices():
    raw = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    out = bt.split_signals(raw, ["ABP", "CVP", "ECG"])
    assert list(out) == ["ABP", "CVP", "ECG"]
    assert torch.equal(out["CVP"], raw[:, 1, :])


def test_split_signals_lifts_single_signal_output():
    out = bt.split_signals(torch.randn(3, 8), ["ABP"])
    assert out["ABP"].shape == (3, 8)


def test_split_signals_rejects_arity_mismatch():
    with pytest.raises(ValueError, match="2 signal"):
        bt.split_signals(torch.randn(1, 2, 5), ["ABP"])


def test_stack_signals_inverts_split():
    raw = torch.randn(2, 3, 7)
    traces = ["ABP", "CVP", "ECG"]
    assert torch.equal(bt.stack_signals(bt.split_signals(raw, traces), traces), raw)


# --- traversal -----------------------------------------------------------
def test_iter_samples_undoes_collation():
    samples = [make_sample(), make_sample()]
    batch = default_collate(samples)
    assert bt.batch_size(batch) == 2
    recovered = list(bt.iter_samples(batch))
    assert len(recovered) == 2
    assert torch.equal(recovered[1][bt.FRAMES]["R"], samples[1][bt.FRAMES]["R"])
    assert recovered[0][bt.METADATA][bt.RECORDING_ID] == "P001_S01_R1_0_D"
    assert bool(recovered[0][bt.LABEL_MASK]["ABP"]) is True
    assert bool(recovered[0][bt.LABEL_MASK]["CVP"]) is False


def test_map_tensors_leaves_metadata_strings_alone():
    batch = default_collate([make_sample()])
    doubled = bt.map_tensors(batch, lambda t: t * 2)
    assert doubled[bt.METADATA][bt.RECORDING_ID] == batch[bt.METADATA][bt.RECORDING_ID]
    assert torch.equal(doubled[bt.LABELS]["ABP"], batch[bt.LABELS]["ABP"] * 2)


def test_move_to_device_cpu_is_a_no_op_copy():
    batch = default_collate([make_sample()])
    moved = bt.move_to_device(batch, torch.device("cpu"))
    assert torch.equal(moved[bt.LABELS]["ABP"], batch[bt.LABELS]["ABP"])


def test_frames_to_rgb_video_is_channels_last_raw():
    sample = make_sample(t=5, hw=(3, 4))
    video = bt.frames_to_rgb_video(sample[bt.FRAMES])
    assert video.shape == (5, 3, 4, 3)
    assert video.dtype == np.float64
    np.testing.assert_allclose(video[..., 1], sample[bt.FRAMES]["G"][0].numpy(), rtol=1e-6)


def test_sample_name_is_stable():
    sample = make_sample()
    assert bt.sample_name(sample[bt.METADATA]) == "P001_S01_R1_0_D_cam1"


# --- frame transforms ----------------------------------------------------
def test_standardized_is_zero_mean_unit_std_per_sample():
    video = torch.rand(3, 2, 4, 5, 6) * 100 + 50
    out = standardized(video)
    flat = out.reshape(3, -1)
    assert torch.allclose(flat.mean(dim=1), torch.zeros(3), atol=1e-5)
    assert torch.allclose(flat.std(dim=1), torch.ones(3), atol=1e-3)


def test_diff_normalized_matches_baseloader_semantics():
    """Same formula, same trailing zero frame, as BaseLoader.diff_normalize_data."""
    video = torch.rand(1, 3, 5, 4, 4) * 200 + 10
    out = diff_normalized(video)
    assert out.shape == video.shape
    assert torch.equal(out[:, :, -1], torch.zeros_like(out[:, :, -1]))
    later, earlier = video[:, :, 1:], video[:, :, :-1]
    expected = (later - earlier) / (later + earlier + 1e-7)
    expected = expected / expected.std()
    assert torch.allclose(out[:, :, :-1], expected, atol=1e-5)


def test_diff_normalized_survives_all_zero_input():
    out = diff_normalized(torch.zeros(1, 1, 4, 2, 2))
    assert torch.isfinite(out).all()


def test_apply_data_types_concatenates_blocks():
    video = torch.rand(2, 3, 4, 5, 5) * 255
    out = apply_data_types(video, ["DiffNormalized", "Standardized"])
    assert out.shape == (2, 6, 4, 5, 5)
    assert torch.allclose(out[:, :3], diff_normalized(video))
    assert torch.allclose(out[:, 3:], standardized(video))


def test_apply_data_types_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown DATA_TYPE"):
        apply_data_types(torch.rand(1, 1, 2, 2, 2), ["Sharpened"])


def test_resize_video_changes_only_spatial_dims():
    video = torch.rand(2, 3, 4, 16, 16)
    out = resize_video(video, (8, 6))
    assert out.shape == (2, 3, 4, 8, 6)
    assert torch.equal(resize_video(video, (16, 16)), video)


def test_frame_transform_channel_multiplier_drives_in_channels():
    single = FrameTransform(("Standardized",))
    double = FrameTransform(("DiffNormalized", "Standardized"), size=(8, 8))
    assert single.channel_multiplier == 1
    assert double.channel_multiplier == 2
    assert double(torch.rand(1, 3, 4, 16, 16)).shape == (1, 6, 4, 8, 8)


def test_loader_keys_match_what_the_dataset_emits():
    """LOADER_KEYS is the documented contract; keep it honest."""
    assert set(make_sample()) == set(bt.LOADER_KEYS)
    assert bt.PREDICTIONS not in bt.LOADER_KEYS


def test_metadata_subkeys_match_the_constants():
    assert set(make_sample()[bt.METADATA]) == {
        bt.RECORDING_ID, bt.CAMERA_ID, bt.START_FRAME, bt.ATTRS}


def test_metadata_carries_store_attrs_as_strings():
    """Grouping attributes ride in metadata, not in the recording id's spelling."""
    sample = {
        "frames": {"G": torch.zeros(1, 4, 2, 2)},
        "labels": {"ABP": torch.zeros(4)},
        "label_stats": {"ABP": {"mean": torch.zeros(())}},
        "channel_mask": {"G": torch.tensor(True)},
        "label_mask": {"ABP": torch.tensor(True)},
        bt.METADATA: {
            "recording_id": "P015_S01_R3_0_D",
            "camera_id": "1",
            "start_frame": 0,
            bt.ATTRS: {"participant": "015", "posture": "0", "light": "D"},
        },
    }
    batch = default_collate([sample, sample])
    first = next(bt.iter_samples(batch))
    assert first[bt.METADATA][bt.ATTRS] == {"participant": "015", "posture": "0",
                                            "light": "D"}
