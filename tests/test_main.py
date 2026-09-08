"""Entry-point wiring: split construction, naming, and the misconfiguration guards."""
import argparse

import pytest

import main
from config import load_config

from tests.zarr_fixtures import make_store

PHYSMAMBA_CONFIG = "configs/neckflix/NECKFLIX_PHYSMAMBA_SMOKE.yaml"
UNSUPERVISED_CONFIG = "configs/neckflix/NECKFLIX_UNSUPERVISED.yaml"


@pytest.fixture
def cache(tmp_path):
    for name in ("P001_S01_R1_0_D", "P002_S01_R1_0_D", "P003_S01_R1_45_D"):
        make_store(tmp_path, name=name, streams=("rgb",), traces=("abp", "cvp"),
                   num_frames=48, hw=(16, 16))
    return tmp_path


def _config(config_file, cache, **test_overrides):
    config = load_config(config_file)
    config.DATA.CACHED_PATH = str(cache)
    config.INTERFACE.WINDOW_SECONDS = 16 / 30      # 16 frames at the fixture rate
    config.INTERFACE.CHANNELS = ["R", "G", "B"]
    config.INTERFACE.TRACES = ["ABP", "CVP"]
    for split in config.DATA.SPLITS.values():
        split.STRIDE_SECONDS = 16 / 30
    # TRACES is narrowed above, so the per-signal registries have to be
    # narrowed with it: naming a signal the run does not predict is an error,
    # not a no-op (LABEL_NORM is resolved to all traces at load).
    config.INTERFACE.LOSS.pop("ECG", None)
    config.INTERFACE.LABEL_NORM.pop("ECG", None)
    for key, value in test_overrides.items():
        setattr(config.TEST, key, value)
    return config


def _args(**overrides):
    defaults = dict(config_file=PHYSMAMBA_CONFIG, test_participants=None,
                    valid_participants=None, limit_windows=0)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# --- splits --------------------------------------------------------------
def test_loso_splits_are_disjoint_by_participant(cache):
    config = _config(PHYSMAMBA_CONFIG, cache, USE_LAST_EPOCH=True)
    loaders = main.build_data_loaders(
        config, _args(test_participants=["P003"]), rank=0, world_size=1, is_main=True)

    def participants(loader):
        dataset = loader.dataset
        return {rec.split("_")[0] for rec, _ in dataset.samples}

    assert participants(loaders["test"]) == {"P003"}
    assert participants(loaders["train"]) == {"P001", "P002"}
    assert loaders["valid"] is None


def test_valid_participants_are_held_out_of_training(cache):
    config = _config(PHYSMAMBA_CONFIG, cache, USE_LAST_EPOCH=False)
    loaders = main.build_data_loaders(
        config, _args(test_participants=["P003"], valid_participants=["P002"]),
        rank=0, world_size=1, is_main=True)
    train = {rec.split("_")[0] for rec, _ in loaders["train"].dataset.samples}
    valid = {rec.split("_")[0] for rec, _ in loaders["valid"].dataset.samples}
    assert train == {"P001"}
    assert valid == {"P002"}


def test_model_selection_without_a_valid_split_is_refused(cache):
    """USE_LAST_EPOCH False with no valid split would silently test epoch 0."""
    config = _config(PHYSMAMBA_CONFIG, cache, USE_LAST_EPOCH=False)
    with pytest.raises(ValueError, match="no --valid_participants were given"):
        main.build_data_loaders(config, _args(test_participants=["P003"]),
                                         rank=0, world_size=1, is_main=True)


def test_empty_split_names_what_to_check(cache):
    config = _config(PHYSMAMBA_CONFIG, cache, USE_LAST_EPOCH=True)
    with pytest.raises(ValueError, match="dataset is empty"):
        main.build_data_loaders(config, _args(test_participants=["P999"]),
                                         rank=0, world_size=1, is_main=True)


def test_limit_windows_subsamples_evenly(cache):
    config = _config(PHYSMAMBA_CONFIG, cache, USE_LAST_EPOCH=True)
    full = main.build_data_loaders(
        config, _args(test_participants=["P003"]), rank=0, world_size=1, is_main=True)
    limited = main.build_data_loaders(
        config, _args(test_participants=["P003"], limit_windows=2),
        rank=0, world_size=1, is_main=True)
    assert len(limited["test"].dataset) == 2 < len(full["test"].dataset)
    # spans the split rather than taking a prefix
    assert limited["test"].dataset.indices[-1] == len(full["test"].dataset) - 1


def test_unsupervised_mode_builds_only_its_own_loader(cache):
    config = _config(UNSUPERVISED_CONFIG, cache)
    loaders = main.build_data_loaders(
        config, _args(config_file=UNSUPERVISED_CONFIG), rank=0, world_size=1,
        is_main=True)
    assert set(loaders) == {"unsupervised"}
    assert len(loaders["unsupervised"].dataset) > 0


# --- naming ---------------------------------------------------------------
def test_experiment_name_records_what_varies(cache):
    config = _config(PHYSMAMBA_CONFIG, cache, USE_LAST_EPOCH=True)
    named = main.apply_experiment_naming(
        config, _args(test_participants=["P015"]))
    name = named.RUN.exp_name
    assert "TRACES-ABP-CVP" in name
    assert "CHANNELS-RGB" in name
    assert "tested_on_015" in name.replace("\\", "/")
    assert named.RUN.model_dir.endswith("PreTrainedModels")
    assert named.RUN.output_dir.endswith("saved_test_outputs")


def test_unsupervised_mode_gets_its_own_output_dir(cache):
    config = _config(UNSUPERVISED_CONFIG, cache)
    named = main.apply_experiment_naming(config, _args())
    assert named.RUN.output_dir.endswith("saved_outputs")


# --- unsupervised dispatch -------------------------------------------------
def test_unknown_unsupervised_method_is_rejected(cache):
    config = _config(UNSUPERVISED_CONFIG, cache)
    config.UNSUPERVISED_METHODS = ["POS", "MAGIC"]
    with pytest.raises(ValueError, match="Not supported unsupervised method"):
        main.run_unsupervised(config, {"unsupervised": []})


def test_unsupervised_is_a_no_op_off_rank_zero(cache):
    config = _config(UNSUPERVISED_CONFIG, cache)
    assert main.run_unsupervised(config, {"unsupervised": []},
                                          is_main=False) is None
