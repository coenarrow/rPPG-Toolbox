"""Experiment config -> zarr-loader plain-dict translation, and the LOSO wiring."""
import pytest

from config import load_config
from dataset.data_loader.NeckflixLoader import NeckflixDataset
from dataset.data_loader.neckflix_config import (
    build_filters, frame_size, normalise_participant, participant_filter,
    window_frames, zarr_config,
)
from tests.zarr_fixtures import make_store

UNSUPERVISED_CONFIG = "configs/neckflix/NECKFLIX_UNSUPERVISED.yaml"
PHYSMAMBA_CONFIG = "configs/neckflix/NECKFLIX_PHYSMAMBA.yaml"


# --- participant ids -----------------------------------------------------
@pytest.mark.parametrize("given,expected", [
    ("P015", "015"), ("015", "015"), (15, "015"), ("p007", "007"),
    (" P030 ", "030"), ("control", "control"),
])
def test_normalise_participant(given, expected):
    """The CLI says P015; the store's root attr says 015."""
    assert normalise_participant(given) == expected


def test_participant_filter_normalises_both_sides():
    spec = participant_filter(include=["P015"], exclude=[7, "P030"])
    assert spec == {"include": ["015"], "exclude": ["007", "030"]}


# --- config translation --------------------------------------------------
def test_unsupervised_config_translates_to_the_loader_contract():
    cfg = zarr_config(load_config(UNSUPERVISED_CONFIG), "unsupervised")
    assert cfg["channels"] == ["R", "G", "B"]
    assert cfg["labels"] == ["ABP", "CVP", "ECG"]
    assert cfg["target_fps"] == 30
    assert cfg["window_size"] == 300           # 10 s at 30 fps
    assert cfg["window_stride"] == 300
    assert cfg["random_windows"] is False
    assert cfg["upsampling"] == "refuse"
    # Per signal, by class: pressures stay in mmHg, ECG is z-scored.
    assert cfg["label_norms"] == {"ABP": "raw", "CVP": "raw", "ECG": "zscore"}
    assert cfg["allow_missing"] is True
    assert cfg["filters"]["posture"] == {"include": ["0", "45", "90"], "exclude": []}


def test_stride_defaults_to_a_whole_window():
    config = load_config(UNSUPERVISED_CONFIG)
    assert zarr_config(config, "test")["window_stride"] == 300


def test_physmamba_train_split_uses_overlapping_windows():
    config = load_config(PHYSMAMBA_CONFIG)
    train = zarr_config(config, "train")
    test = zarr_config(config, "test")
    assert train["window_size"] == 128 and train["window_stride"] == 64
    assert test["window_stride"] == 128, "evaluation windows should not overlap"


# --- the physical-time contract ------------------------------------------
def test_window_frames_snaps_a_decimal_spelling_of_an_exact_fraction():
    """64/15 s at 30 fps is 128 frames, however many decimals it is written to."""
    assert window_frames(4.266667, 30) == 128
    assert window_frames(5.333333, 30) == 160
    assert window_frames(6.0, 30) == 180


def test_window_frames_refuses_a_genuinely_ambiguous_duration():
    """4.27 x 30 = 128.1: a real mistake, not a rounding artefact."""
    with pytest.raises(ValueError, match="4.266667"):
        window_frames(4.27, 30)


def test_window_frames_needs_a_rate():
    with pytest.raises(ValueError, match="FS"):
        window_frames(4.0, 0)


def test_label_norm_can_be_overridden_per_signal():
    config = load_config(PHYSMAMBA_CONFIG)
    config.INTERFACE.LABEL_NORM = {"ABP": "zscore"}
    assert zarr_config(config, "train")["label_norms"]["ABP"] == "zscore"


def test_label_norm_for_an_unpredicted_signal_is_refused():
    config = load_config(PHYSMAMBA_CONFIG)
    config.INTERFACE.LABEL_NORM = {"PPG": "zscore"}
    with pytest.raises(ValueError, match="not in TRACES"):
        zarr_config(config, "train")


def test_loso_filters_are_disjoint():
    config = load_config(PHYSMAMBA_CONFIG)
    train = zarr_config(config, "train", exclude_participants=["P015"])
    held_out = zarr_config(config, "test", include_participants=["P015"])
    assert train["filters"]["participant"] == {"include": [], "exclude": ["015"]}
    assert held_out["filters"]["participant"] == {"include": ["015"], "exclude": []}


def test_random_windows_override_wins_over_the_config():
    config = load_config(PHYSMAMBA_CONFIG)
    config.DATA.split("train").RANDOM_WINDOWS = True
    assert zarr_config(config, "train")["random_windows"] is True
    assert zarr_config(config, "train", random_windows=False)["random_windows"] is False


def test_empty_config_lists_mean_no_filter():
    config = load_config(UNSUPERVISED_CONFIG)
    config.DATA.FILTERS = {"posture": []}
    assert build_filters(config.DATA) == {}


def test_configured_attribute_lists_become_include_filters():
    """FILTERS keys are store attrs verbatim — no fixed list, any attr works."""
    config = load_config(UNSUPERVISED_CONFIG)
    config.DATA.FILTERS = {"perspective": [1], "light": ["D"],
                           "site": ["A"]}   # an attr no fixed list ever knew
    filters = build_filters(config.DATA)
    assert filters["perspective"]["include"] == [1]
    assert filters["light"]["include"] == ["D"]
    assert filters["site"]["include"] == ["A"]


def test_a_split_can_override_the_filters():
    config = load_config(UNSUPERVISED_CONFIG)
    config.DATA.SPLITS["TEST"] = type(config.DATA.split("test"))(
        FILTERS={"posture": ["45"]})
    assert zarr_config(config, "test")["filters"]["posture"]["include"] == ["45"]
    # An unstated split still inherits DATA's filters.
    assert zarr_config(config, "train")["filters"]["posture"]["include"] == \
        ["0", "45", "90"]


def test_participant_key_in_filters_is_refused():
    """Participants go through PARTICIPANTS/CLI so their ids get normalised."""
    config = load_config(UNSUPERVISED_CONFIG)
    config.DATA.FILTERS = {"participant": ["P015"]}
    with pytest.raises(ValueError, match="PARTICIPANTS"):
        build_filters(config.DATA)


def test_explicit_participants_list_is_used_when_no_cli_argument():
    config = load_config(UNSUPERVISED_CONFIG)
    config.DATA.PARTICIPANTS = ["P002"]
    filters = build_filters(config.DATA)
    assert filters["participant"]["include"] == ["002"]


def test_frame_size_reads_the_resize_block():
    config = load_config(PHYSMAMBA_CONFIG)
    assert frame_size(config.INTERFACE) == (128, 128)
    config.INTERFACE.RESIZE.H = 0
    assert frame_size(config.INTERFACE) is None


# --- against a real (synthetic) cache ------------------------------------
def test_translated_config_drives_a_loso_split(tmp_path):
    for name in ("P015_S01_R1_0_D", "P016_S01_R1_0_D", "P017_S01_R1_45_N"):
        make_store(tmp_path, name=name, streams=("rgb",), traces=("abp", "cvp"),
                   num_frames=16, hw=(6, 6))
    config = load_config(UNSUPERVISED_CONFIG)
    config.DATA.CACHED_PATH = str(tmp_path)
    config.INTERFACE.WINDOW_SECONDS = 8 / 30
    config.INTERFACE.TRACES = ["ABP", "CVP"]
    config.INTERFACE.LABEL_NORM = {}    # narrowed with TRACES

    held_out = NeckflixDataset(zarr_config(config, "test",
                                           include_participants=["P015"]))
    rest = NeckflixDataset(zarr_config(config, "test",
                                       exclude_participants=["P015"]))
    assert held_out.attribute_values("participant") == ["015"]
    assert rest.attribute_values("participant") == ["016", "017"]
    assert len(held_out) and len(rest)


def test_posture_filter_from_the_config_reaches_the_loader(tmp_path):
    make_store(tmp_path, name="P020_S01_R1_0_D", streams=("rgb",),
               traces=("abp",), num_frames=16, hw=(6, 6))
    make_store(tmp_path, name="P020_S01_R2_45_D", streams=("rgb",),
               traces=("abp",), num_frames=16, hw=(6, 6))
    config = load_config(UNSUPERVISED_CONFIG)
    config.DATA.CACHED_PATH = str(tmp_path)
    config.INTERFACE.WINDOW_SECONDS = 8 / 30
    config.INTERFACE.TRACES = ["ABP"]
    config.INTERFACE.LABEL_NORM = {}    # narrowed with TRACES
    config.DATA.FILTERS = {"posture": ["45"]}
    dataset = NeckflixDataset(zarr_config(config, "test"))
    assert {rec for rec, _ in dataset.samples} == {"P020_S01_R2_45_D"}
