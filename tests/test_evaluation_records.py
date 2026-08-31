"""The records layer: one physical-unit inversion, two equivalent sources."""
import pickle

import numpy as np

from evaluation.records import RunRecords, from_saved, load, to_physical


def _raw_window(signal="ABP", start=0):
    return {
        "signal": signal,
        "recording_id": "P015_S01_R3_0_D",
        "camera_id": "1",
        "start_frame": start,
        "prediction": np.linspace(80.0, 120.0, 16, dtype=np.float32),
        "label": np.linspace(78.0, 118.0, 16, dtype=np.float32),
        "label_stats": {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0},
        "attrs": {"participant": "015", "posture": "0"},
    }


def test_raw_mode_inverts_by_identity():
    values = np.array([80.0, 120.0])
    stats = {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0}
    assert np.allclose(to_physical(values, stats, "raw"), values)


def test_zscore_mode_uses_the_window_stats():
    values = np.array([-1.0, 0.0, 1.0])
    stats = {"mean": 90.0, "std": 10.0, "min": 80.0, "max": 100.0}
    assert np.allclose(to_physical(values, stats, "zscore"), [80.0, 90.0, 100.0])


def test_memory_and_pickle_sources_agree(tmp_path):
    """The trainer's in-memory records and its own pickle must load identically."""
    raw = [_raw_window(start=0), _raw_window(start=16)]
    meta = dict(fs=30.0, traces=("ABP",), label_norms={"ABP": "raw"})
    in_memory = from_saved(raw, **meta)

    path = tmp_path / "run_outputs.pickle"
    with open(path, "wb") as handle:
        pickle.dump({"windows": raw, "channels": ["G"], **meta,
                     "traces": ["ABP"]}, handle)
    from_disk = load(tmp_path)

    assert isinstance(in_memory, RunRecords) and isinstance(from_disk, RunRecords)
    assert len(in_memory.windows) == len(from_disk.windows) == 2
    assert in_memory.signals() == from_disk.signals() == ["ABP"]
    for a, b in zip(in_memory.windows, from_disk.windows):
        assert np.allclose(a.prediction, b.prediction)
        assert a.attrs == b.attrs == {"participant": "015", "posture": "0"}
