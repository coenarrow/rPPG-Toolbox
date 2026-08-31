"""The tidy frame is the interface between layers; its schema is the contract."""
import numpy as np
import pytest

from evaluation.records import from_saved
from evaluation.report import FRAME_COLUMNS, build_frame, digest

FS = 30.0


def _run(n_windows=6, signal="ABP"):
    t = np.arange(120) / FS
    wave = 100.0 + 20.0 * -np.cos(2 * np.pi * 1.2 * t)
    raw = [{
        "signal": signal,
        "recording_id": "P015_S01_R3_0_D",
        "camera_id": "1",
        "start_frame": i * 120,
        "prediction": (wave + 2.0).astype(np.float32),
        "label": wave.astype(np.float32),
        "label_stats": {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0},
        "attrs": {"participant": "015", "posture": "0"},
    } for i in range(n_windows)]
    return from_saved(raw, fs=FS, traces=(signal,), label_norms={signal: "raw"})


def test_frame_has_the_contract_columns_and_level_vocabulary():
    frame = build_frame(_run())
    assert tuple(frame.columns[:len(FRAME_COLUMNS)]) == FRAME_COLUMNS
    assert set(frame["level"]) <= {"beat", "window", "section", "recording",
                                   "participant", "cohort"}
    assert frame["value"].dtype == np.float64
    assert "participant" in frame.columns


def test_a_constant_offset_shows_up_as_beat_level_bias():
    frame = build_frame(_run())
    beats = frame[(frame["level"] == "beat") & (frame["metric"] == "beat_error")]
    assert not beats.empty
    assert beats["value"].mean() == pytest.approx(2.0, abs=0.5)


def test_single_participant_is_declined_not_graded():
    frame = build_frame(_run())
    verdict = frame[frame["metric"] == "iso81060_3_passes"]
    assert verdict.empty or verdict["value"].isna().all()


def test_digest_names_the_threshold_provenance():
    text = digest(build_frame(_run()), _run())
    assert "UNVERIFIED" in text
