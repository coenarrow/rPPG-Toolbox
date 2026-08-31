import numpy as np
import zarr

from tests.zarr_fixtures import TOOL_VERSION, base_cfg, make_store


def test_store_schema_matches_preprocessor_output(tmp_path):
    path = make_store(tmp_path, "P030_S01_R1_0_D", num_frames=10, hw=(6, 6))
    root = zarr.open_group(str(path), mode="r")
    attrs = dict(root.attrs)
    assert attrs["participant"] == "030"          # unprefixed (spec: value format)
    assert attrs["posture"] == "0"
    assert attrs["recording"] == "P030_S01_R1_0_D"
    assert attrs["complete"] is True
    assert attrs["tool_version"] == TOOL_VERSION
    rgb = root["1"]["rgb"]["video"]["frames"]
    assert rgb.shape == (3, 10, 6, 6) and rgb.dtype == np.uint8
    ir = root["1"]["ir"]["video"]["frames"]
    assert ir.shape == (1, 10, 6, 6) and ir.dtype == np.uint16
    assert root["1"]["rgb"]["video"].attrs["num_frames"] == 10
    abp = root["1"]["rgb"]["abp"]["data"]
    assert abp.shape == (10,) and abp.dtype == np.float64


def test_attr_override_and_removal(tmp_path):
    path = make_store(tmp_path, "P031_S01_R1_45_D",
                      attrs={"complete": None, "tool_version": "0.9.0"})
    attrs = dict(zarr.open_group(str(path), mode="r").attrs)
    assert "complete" not in attrs
    assert attrs["tool_version"] == "0.9.0"
    assert attrs["posture"] == "45"


def test_per_stream_num_frames_and_short_trace(tmp_path):
    path = make_store(tmp_path, "P032_S01_R1_0_D",
                      num_frames={"rgb": 12, "ir": 8, "depth": 12},
                      trace_lengths={("1", "rgb", "abp"): 7})
    root = zarr.open_group(str(path), mode="r")
    assert root["1"]["ir"]["video"].attrs["num_frames"] == 8
    assert root["1"]["rgb"]["abp"]["data"].shape == (7,)
    assert root["1"]["rgb"]["cvp"]["data"].shape == (12,)


def test_events_and_extra_groups(tmp_path):
    path = make_store(tmp_path, "P033_S01_R1_0_D",
                      events=True, extra_groups=(("1", "notes"),))
    root = zarr.open_group(str(path), mode="r")
    assert "events" in root
    assert "video" not in root["events"]
    assert "video" not in root["1"]["notes"]


def test_base_cfg_shape(tmp_path):
    cfg = base_cfg(tmp_path, window_size=8, labels=["ABP"])
    assert cfg["cache_dir"] == str(tmp_path)
    assert cfg["window_size"] == 8
    assert cfg["labels"] == ["ABP"]
    # The physical window follows the frame count at the fixture rate, and the
    # norm map covers exactly the configured labels.
    assert cfg["window_seconds"] == 8 / 30
    assert cfg["label_norms"] == {"ABP": "zscore"}
