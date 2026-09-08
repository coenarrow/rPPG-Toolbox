"""Smoke test for the contract-v2 cache validator (one file, several cases)."""
import numpy as np
import zarr

from tests.zarr_fixtures import make_v2_store
from tools.validate_cache import main, validate_store


def test_conformant_store_passes(tmp_path):
    path = make_v2_store(tmp_path, traces=("abp", "cvp", "ecg"))
    assert validate_store(path) == []


def _messages(path):
    # str(Violation) is "where: message" — the offending name lives in `where`,
    # so joining only the messages would throw away what each case asserts on.
    return " | ".join(str(v) for v in validate_store(path))


def test_violations_are_itemised(tmp_path):
    # Missing participant.
    path = make_v2_store(tmp_path, name="P001_S01_R1_0_D")
    root = zarr.open_group(str(path), mode="a")
    del root.attrs["participant"]
    assert "participant" in _messages(path)

    # Unknown modality name.
    path = make_v2_store(tmp_path, name="P002_S01_R1_0_D")
    root = zarr.open_group(str(path), mode="a")
    bad = root["1"].create_group("sonar")
    bad.create_group("video")["data"] = np.zeros((1, 4, 2, 2), np.uint8)
    assert "sonar" in _messages(path)

    # Trace set differs between modalities.
    path = make_v2_store(tmp_path, name="P003_S01_R1_0_D")
    root = zarr.open_group(str(path), mode="a")
    del root["1"]["ir"]["cvp"]
    assert "trace" in _messages(path).lower()

    # Missing units attr.
    path = make_v2_store(tmp_path, name="P004_S01_R1_0_D")
    root = zarr.open_group(str(path), mode="a")
    del root["1"]["rgb"]["abp"].attrs["units"]
    assert "units" in _messages(path)

    # Trace length disagrees with the video within one modality.
    path = make_v2_store(tmp_path, name="P005_S01_R1_0_D")
    root = zarr.open_group(str(path), mode="a")
    root["1"]["rgb"]["abp"]["data"] = np.zeros(7, np.float64)
    assert "abp" in _messages(path)

    # First-frame misalignment beyond 1/fps (fps=30 -> 33_333 us budget).
    path = make_v2_store(tmp_path, name="P006_S01_R1_0_D",
                         first_frame_offsets_us={"ir": 50_000.0})
    assert "align" in _messages(path).lower()

    # Missing perspective fps.
    path = make_v2_store(tmp_path, name="P007_S01_R1_0_D")
    root = zarr.open_group(str(path), mode="a")
    del root["1"].attrs["fps"]
    assert "fps" in _messages(path)

    # A trace written as a bare array rather than <trace>/data. A group-only
    # walk cannot see it, so without the array-child check the store — which
    # the reader would choke on — validates clean.
    path = make_v2_store(tmp_path, name="P008_S01_R1_0_D")
    root = zarr.open_group(str(path), mode="a")
    root["1"]["rgb"]["ecg"] = np.zeros(12, np.float64)
    assert "array child" in _messages(path)


def test_a_single_store_path_is_validated_not_swept(tmp_path, capsys):
    # A .zarr store is a directory too, so the cache sweep must not glob inside
    # one: the documented "store.zarr" form used to find nothing and exit 1.
    path = make_v2_store(tmp_path, traces=("abp", "cvp", "ecg"))
    assert main([str(path)]) == 0
    assert "PASS" in capsys.readouterr().out


def test_malformed_nodes_are_itemised_not_raised(tmp_path):
    # The validator sweeps a whole cache directory, so raising on one malformed
    # store would leave every store sorted after it unchecked.
    path = make_v2_store(tmp_path, name="P010_S01_R1_0_D")
    root = zarr.open_group(str(path), mode="a")
    del root["1"]["rgb"]["video"]
    root["1"]["rgb"]["video"] = np.zeros((3, 12, 8, 8), np.uint8)   # the v1 shape
    assert "video/data" in _messages(path)

    path = make_v2_store(tmp_path, name="P011_S01_R1_0_D")
    root = zarr.open_group(str(path), mode="a")
    del root["1"]["rgb"]["abp"]["data"]
    root["1"]["rgb"]["abp"].create_group("data")                    # a group, not an array
    assert "abp" in _messages(path)


def test_video_dtype_is_unconstrained(tmp_path):
    # 2026-09-02 amendment (docs/cache-contract.md): the contract says nothing
    # about frame dtype. A 16-bit IR/depth sensor writes uint16 and a float
    # store is equally fine; a validator that demands uint8 fails every real
    # Neckflix store.
    path = make_v2_store(tmp_path, name="P012_S01_R1_0_D")
    root = zarr.open_group(str(path), mode="a")
    for modality, dtype in (("ir", np.uint16), ("depth", np.float32)):
        video = root["1"][modality]["video"]
        frames = video["data"][:].astype(dtype)
        del video["data"]
        video["data"] = frames
    assert validate_store(path) == []


def test_participant_must_be_a_string(tmp_path):
    # Any identifier, any format -- but a string: the split machinery matches
    # it exactly, and an int 13 never equals a configured "013".
    path = make_v2_store(tmp_path, name="P013_S01_R1_0_D")
    root = zarr.open_group(str(path), mode="a")
    root.attrs["participant"] = 13
    assert "participant" in _messages(path)


def test_perspective_fps_may_be_null(tmp_path):
    # An event camera has no frame rate: the key is still required, its value
    # may be null, and with no rate there is no 1/fps budget to judge
    # first-frame alignment against, so that check is skipped.
    path = make_v2_store(tmp_path, name="P014_S01_R1_0_D",
                         perspectives=("1", "2"),
                         first_frame_offsets_us={"ir": 50_000.0})
    root = zarr.open_group(str(path), mode="a")
    root["1"].attrs["fps"] = None
    root["2"].attrs["fps"] = float("nan")     # the other spelling of "none"
    assert validate_store(path) == []


def test_perspective_fps_is_otherwise_a_positive_number(tmp_path):
    path = make_v2_store(tmp_path, name="P015_S01_R1_0_D")
    root = zarr.open_group(str(path), mode="a")
    root["1"].attrs["fps"] = "30"
    assert "fps" in _messages(path)
