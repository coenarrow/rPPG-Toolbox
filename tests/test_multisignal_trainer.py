"""The dict-contract trainer, end to end over a synthetic zarr cache.

Small stores, tiny frames, one epoch: this is a wiring test, not a learning
test. What it pins down is that the batch dict survives the whole round trip —
loader, collate, model, masked loss, per-signal metrics, saved outputs — and
that a recording missing a trace is scored on the traces it does have.
"""
import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from config import RunPaths, load_config
from dataset.data_loader.NeckflixLoader import NeckflixDataset
from dataset.data_loader.neckflix_config import zarr_config
from neural_methods.batch import PREDICTIONS, move_to_device
from neural_methods.trainer.MultiSignalTrainer import (
    MODEL_REGISTRY, MultiSignalTrainer, build_model,
)
from tests.zarr_fixtures import make_store

SMOKE_CONFIG = "configs/neckflix/NECKFLIX_PHYSMAMBA_SMOKE.yaml"
WINDOW = 16
FRAME_SIZE = 32
FS = 30        # the synthetic stores' native rate


@pytest.fixture
def cache(tmp_path):
    """Three recordings; the third carries no ABP, exercising label_mask."""
    make_store(tmp_path, name="P001_S01_R1_0_D", streams=("rgb",),
               traces=("abp", "cvp"), num_frames=40, hw=(FRAME_SIZE, FRAME_SIZE))
    make_store(tmp_path, name="P002_S01_R1_0_D", streams=("rgb",),
               traces=("abp", "cvp"), num_frames=40, hw=(FRAME_SIZE, FRAME_SIZE))
    make_store(tmp_path, name="P003_S01_R1_45_D", streams=("rgb",),
               traces=("cvp",), num_frames=40, hw=(FRAME_SIZE, FRAME_SIZE))
    return tmp_path


@pytest.fixture
def config(cache):
    cfg = load_config(SMOKE_CONFIG)
    cfg.DATA.CACHED_PATH = str(cache)
    cfg.INTERFACE.WINDOW_SECONDS = WINDOW / FS  # WINDOW frames at the fixture rate
    cfg.INTERFACE.CHANNELS = ["R", "G", "B"]
    cfg.INTERFACE.TRACES = ["ABP", "CVP"]
    cfg.INTERFACE.RESIZE.H = FRAME_SIZE
    cfg.INTERFACE.RESIZE.W = FRAME_SIZE
    for split in cfg.DATA.SPLITS.values():
        split.STRIDE_SECONDS = WINDOW / FS
    # TRACES is narrowed above, so the per-signal registries have to be
    # narrowed with it: naming a signal the run does not predict is an error,
    # not a no-op (LABEL_NORM is resolved to all traces at load).
    cfg.TRAIN.LOSS.pop("ECG", None)
    cfg.INTERFACE.LABEL_NORM.pop("ECG", None)
    cfg.TRAIN.EPOCHS = 1
    cfg.TRAIN.BATCH_SIZE = 2
    cfg.TEST.BATCH_SIZE = 2
    cfg.TEST.USE_LAST_EPOCH = True
    cfg.TEST.REPORT.PLOTS = []                        # keep unit tests off the filesystem
    cfg.LOG_PATH = str(cache / "logs")
    cfg.RUN = RunPaths(exp_name="test_exp",
                       model_dir=str(cache / "models"),
                       output_dir=str(cache / "outputs"))
    return cfg


def loaders_for(config, *, train_exclude=("P003",), test_include=("P003",)):
    def loader(split, batch_size, **kwargs):
        dataset = NeckflixDataset(zarr_config(config, split, random_windows=False,
                                              **kwargs))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          num_workers=0, drop_last=False)
    return {
        "train": loader("train", config.TRAIN.BATCH_SIZE,
                        exclude_participants=train_exclude),
        "valid": None,
        "test": loader("test", config.TEST.BATCH_SIZE,
                       include_participants=test_include),
    }


# --- registry / construction --------------------------------------------
def test_registry_holds_the_dict_contract_models():
    assert {"PhysMamba", "DeepPhys"} <= set(MODEL_REGISTRY)


def test_build_model_reads_the_interface(config):
    model = build_model(config)
    assert model.channels == ("R", "G", "B")
    assert model.traces == ("ABP", "CVP")
    assert model.frame_transform.size == (FRAME_SIZE, FRAME_SIZE)
    assert model.frame_transform.data_types == ("DiffNormalized",)


def test_build_model_rejects_an_unregistered_model(config):
    config.MODEL.NAME = "RhythmFormer"
    with pytest.raises(ValueError, match="does not speak the Neckflix dict contract"):
        build_model(config)


# --- end to end ----------------------------------------------------------
def test_train_then_test_round_trip(config, cache, capsys):
    loaders = loaders_for(config)
    assert len(loaders["train"].dataset) > 0
    assert len(loaders["test"].dataset) > 0

    trainer = MultiSignalTrainer(config, loaders, rank=0, world_size=1, debug=False)
    trainer.train(loaders)
    checkpoint = cache / "models" / "neckflix_physmamba_smoke_Epoch0.pth"
    assert checkpoint.exists()
    # The checkpoint carries the interface it was trained against.
    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["interface"]["CHANNELS"] == ["R", "G", "B"]
    assert payload["model_name"] == "PhysMamba"

    frame = trainer.test(loaders)
    # P003 has no ABP, so only CVP is scored on the held-out split.
    assert set(frame["signal"]) == {"CVP"}
    window_mae = frame[(frame["level"] == "window") & (frame["metric"] == "mae")]
    assert not window_mae.empty
    assert np.isfinite(window_mae["value"]).all()          # finite

    printed = capsys.readouterr().out
    assert "--- CVP (mmHg) ---" in printed
    assert "no windows carried this label" in printed      # the ABP row


def test_saved_outputs_carry_signal_keyed_windows(config, cache):
    loaders = loaders_for(config, train_exclude=("P003",), test_include=("P001",))
    trainer = MultiSignalTrainer(config, loaders, rank=0, world_size=1, debug=False)
    trainer.train(loaders)
    trainer.test(loaders)

    path = cache / "outputs" / "neckflix_physmamba_smoke_outputs.pickle"
    payload = pickle.loads(path.read_bytes())
    assert payload["traces"] == ["ABP", "CVP"]
    assert payload["channels"] == ["R", "G", "B"]
    assert payload["label_norms"] == {"ABP": "raw", "CVP": "raw"}
    signals = {record["signal"] for record in payload["windows"]}
    assert signals == {"ABP", "CVP"}
    record = payload["windows"][0]
    assert record["recording_id"].startswith("P001")
    assert len(record["prediction"]) == WINDOW == len(record["label"])
    assert set(record["label_stats"]) == {"mean", "std", "min", "max"}


def test_one_training_step_moves_the_weights(config):
    loaders = loaders_for(config)
    trainer = MultiSignalTrainer(config, loaders, rank=0, world_size=1, debug=False)
    before = trainer.model.ConvBlock1[0].weight.detach().clone()
    trainer.train(loaders)
    after = trainer.model.ConvBlock1[0].weight.detach()
    assert not torch.allclose(before, after)


def test_model_output_still_carries_the_loader_keys(config):
    loaders = loaders_for(config)
    trainer = MultiSignalTrainer(config, loaders, rank=0, world_size=1, debug=False)
    batch = move_to_device(next(iter(loaders["train"])), trainer.device)
    out = trainer.model(batch)
    assert set(out) == set(batch) | {PREDICTIONS}
    assert set(out[PREDICTIONS]) == {"ABP", "CVP"}
    assert out["metadata"]["recording_id"] == batch["metadata"]["recording_id"]


def test_validation_split_drives_best_epoch_selection(config, cache):
    """USE_LAST_EPOCH False + a held-out valid split exercises valid()."""
    config.TEST.USE_LAST_EPOCH = False
    config.TRAIN.EPOCHS = 2

    def loader(split, batch_size, **kwargs):
        dataset = NeckflixDataset(zarr_config(config, split, random_windows=False,
                                              **kwargs))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          num_workers=0, drop_last=False)

    loaders = {
        "train": loader("train", 2, exclude_participants=("P002", "P003")),
        "valid": loader("valid", 2, include_participants=("P002",)),
        "test": loader("test", 2, include_participants=("P003",)),
    }
    trainer = MultiSignalTrainer(config, loaders, rank=0, world_size=1, debug=False)
    trainer.train(loaders)

    assert trainer.min_valid_loss is not None
    assert np.isfinite(trainer.min_valid_loss)
    assert trainer.best_epoch in (0, 1)
    # test() then loads the *best* epoch, not the last one.
    assert (cache / "models"
            / f"neckflix_physmamba_smoke_Epoch{trainer.best_epoch}.pth").exists()
    assert trainer.test(loaders) is not None


def test_valid_without_a_valid_loader_is_an_explicit_error(config):
    loaders = loaders_for(config)
    trainer = MultiSignalTrainer(config, loaders, rank=0, world_size=1, debug=False)
    with pytest.raises(ValueError, match="No data for valid"):
        trainer.valid({"valid": None})


def test_only_test_mode_needs_no_train_loader(config, cache):
    """Train once to produce a checkpoint, then reload it through only_test."""
    loaders = loaders_for(config, test_include=("P001",))
    MultiSignalTrainer(config, loaders, rank=0, world_size=1, debug=False).train(loaders)

    config.MODE = "only_test"
    config.TEST.MODEL_PATH = str(
        cache / "models" / "neckflix_physmamba_smoke_Epoch0.pth")
    trainer = MultiSignalTrainer(config, {"test": loaders["test"]},
                                 rank=0, world_size=1, debug=False)
    assert trainer.test({"test": loaders["test"]}) is not None


def test_unknown_label_norm_is_rejected_at_construction(config):
    """One resolver validates the key, and both the loader and the trainer use it —
    the trainer needs it to pick the matching inverse for the physical report."""
    loaders = loaders_for(config)                      # built while the norm is valid
    config.INTERFACE.LABEL_NORM = {"ABP": "robust"}
    with pytest.raises(ValueError, match="LABEL_NORM for ABP"):
        MultiSignalTrainer(config, loaders, rank=0, world_size=1, debug=False)
    with pytest.raises(ValueError, match="LABEL_NORM for ABP"):
        NeckflixDataset(zarr_config(config, "test"))


def test_epoch_mean_loss_is_printed(config, capsys):
    """The progress bar shows only the last batch; the mean is what matters."""
    loaders = loaders_for(config)
    trainer = MultiSignalTrainer(config, loaders, rank=0, world_size=1, debug=False)
    trainer.train(loaders)
    printed = capsys.readouterr().out
    assert "mean training loss:" in printed
