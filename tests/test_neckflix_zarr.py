import pytest
import torch

from dataset.data_loader.NeckflixLoader import NeckflixDataset
from dataset.data_loader.zarr_dataset import BaseZarrDataset
from tests.zarr_fixtures import base_cfg, make_store, make_unreadable_store


# --------------------------------------------------------------------------
# Task 5: class shape + config validation
# --------------------------------------------------------------------------
def test_neckflix_channel_map():
    ds = object.__new__(NeckflixDataset)   # property needs no __init__
    assert ds.channel_map == {
        "R": ("rgb", 0), "G": ("rgb", 1), "B": ("rgb", 2),
        "I": ("ir", 0), "D": ("depth", 0),
    }


def test_is_torch_dataset_subclass():
    assert issubclass(NeckflixDataset, BaseZarrDataset)
    assert issubclass(BaseZarrDataset, torch.utils.data.Dataset)


def test_bad_label_norm_raises(tmp_path):
    with pytest.raises(ValueError, match="LABEL_NORM"):
        NeckflixDataset(base_cfg(tmp_path, label_norms={"ABP": "fixed"}))


def test_filter_overlap_raises_upfront_even_with_empty_cache(tmp_path):
    # Deviation 6: validation is unconditional, before any store is scanned.
    cfg = base_cfg(
        tmp_path,
        filters={"participant": {"include": ["030"], "exclude": ["030"]}},
    )
    with pytest.raises(ValueError, match="Overlapping include/exclude"):
        NeckflixDataset(cfg)


def test_all_unknown_channels_raise(tmp_path):
    with pytest.raises(ValueError, match="None of the demanded channels"):
        NeckflixDataset(base_cfg(tmp_path, channels=["X", "Y"]))


def test_unknown_channel_is_delivered_as_zeros_with_a_false_mask(tmp_path):
    """Demand-driven delivery: a channel this dataset can never provide is
    zeros + channel_mask=False, so a wider pretrained checkpoint still runs."""
    make_store(tmp_path, "P030_S01_R1_0_D", streams=("rgb",), num_frames=12)
    with pytest.warns(UserWarning, match="not provided by"):
        ds = NeckflixDataset(base_cfg(
            tmp_path, channels=["R", "G", "B", "X"], window_size=4))
    sample = ds[0]
    assert not bool(sample["channel_mask"]["X"])
    assert torch.all(sample["frames"]["X"] == 0)
    assert sample["frames"]["X"].shape == sample["frames"]["R"].shape
    assert bool(sample["channel_mask"]["R"])


def test_a_slower_store_is_refused_by_default(tmp_path):
    """Naive upsampling duplicates frames and darkens DiffNormalized; the
    refusal names the opt-in."""
    make_store(tmp_path, "P030_S01_R1_0_D", streams=("rgb",), num_frames=12,
               fps=15.0)
    with pytest.raises(ValueError, match="UPSAMPLING"):
        NeckflixDataset(base_cfg(tmp_path, channels=["R", "G", "B"],
                                 window_size=8))


def test_opted_in_upsampling_interpolates_rather_than_duplicates(tmp_path):
    import numpy as np

    make_store(tmp_path, "P030_S01_R1_0_D", streams=("rgb",), traces=("abp",),
               num_frames=12, fps=15.0,
               trace_values={("1", "rgb", "abp"): 100.0 + np.arange(12.0)})
    ds = NeckflixDataset(base_cfg(
        tmp_path, channels=["R", "G", "B"], labels=["ABP"], window_size=8,
        upsampling="interpolate", label_norms={"ABP": "raw"}))
    sample = ds[0]
    abp = sample["labels"]["ABP"].numpy()
    assert len(abp) == 8
    # 8 target frames at 30 fps span 4 native frames at 15 fps: the ramp label
    # comes back in linearly interpolated half-steps, never as repeats.
    assert np.allclose(np.diff(abp)[:-1], 0.5)
    frames = sample["frames"]["R"]
    assert not torch.equal(frames[:, 0], frames[:, 1])


# --------------------------------------------------------------------------
# End-to-end smoke: data loads through a real DataLoader
# --------------------------------------------------------------------------
def test_dataloader_end_to_end_smoke(tmp_path):
    from torch.utils.data import DataLoader

    make_store(tmp_path, "P030_S01_R1_0_D", num_frames=12)
    make_store(tmp_path, "P031_S01_R1_45_D", num_frames=12)
    ds = NeckflixDataset(base_cfg(tmp_path, window_size=4))
    assert len(ds) == 6                                  # 2 recordings x 3 windows

    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
    assert set(batch) == {"frames", "labels", "label_stats",
                          "channel_mask", "label_mask", "metadata"}
    for ch in ("R", "G", "B", "I", "D"):
        assert batch["frames"][ch].shape == (4, 1, 4, 8, 8)
        assert batch["frames"][ch].dtype == torch.float32
        assert batch["channel_mask"][ch].dtype == torch.bool
    for sig in ("ABP", "CVP"):
        assert batch["labels"][sig].shape == (4, 4)
        assert torch.isfinite(batch["labels"][sig]).all()
        assert batch["label_stats"][sig]["mean"].shape == (4,)
        assert batch["label_mask"][sig].all()
    assert batch["metadata"]["recording_id"][0] == "P030_S01_R1_0_D"
    assert batch["metadata"]["start_frame"].dtype == torch.int64


def test_dataset_keys_are_exactly_the_declared_contract(tmp_path):
    """The dataset and neural_methods.batch must not drift apart."""
    from neural_methods.batch import (
        ATTRS, CAMERA_ID, LOADER_KEYS, METADATA, RECORDING_ID, START_FRAME,
    )

    make_store(tmp_path, "P030_S01_R1_0_D", num_frames=12)
    item = NeckflixDataset(base_cfg(tmp_path, window_size=4))[0]
    assert set(item) == set(LOADER_KEYS)
    assert set(item[METADATA]) == {RECORDING_ID, CAMERA_ID, START_FRAME, ATTRS}


def test_model_consumes_the_dataset_output_unchanged(tmp_path):
    """Loader -> collate -> model, with no adapter in between."""
    from torch.utils.data import DataLoader

    from neural_methods.batch import PREDICTIONS
    from neural_methods.frame_transforms import FrameTransform
    from neural_methods.model.PhysMamba import PhysMamba

    make_store(tmp_path, "P030_S01_R1_0_D", num_frames=64, hw=(32, 32))
    dataset = NeckflixDataset(base_cfg(tmp_path, window_size=16, channels=["R", "G", "B"]))
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    model = PhysMamba(channels=("R", "G", "B"), traces=("ABP", "CVP"),
                      frame_transform=FrameTransform(("DiffNormalized",), size=(32, 32)))
    out = model(batch)
    assert set(out) == set(batch) | {PREDICTIONS}
    assert out[PREDICTIONS]["ABP"].shape == (2, 16)
    assert torch.isfinite(out[PREDICTIONS]["CVP"]).all()
