"""Smoke test for the contract-v2 cache validator (one file, several cases)."""
import numpy as np
import zarr

from tests.zarr_fixtures import make_v2_store
from tools.validate_cache import validate_store


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
