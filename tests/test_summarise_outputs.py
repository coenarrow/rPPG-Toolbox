"""The offline summariser over a saved test pickle.

The pickle is self-describing on purpose: every window carries the stats that
normalised it, so the physical-unit numbers can be reproduced without the zarr
cache. These tests pin that.
"""
import pickle

import numpy as np
import pytest

from tools.summarise_neckflix_outputs import find_pickle, summarise, window_table

FS = 30


def _window(signal, recording, start, *, mean=90.0, std=12.0, offset=0.0, n=64):
    t = np.arange(n) / FS
    label = np.sin(2 * np.pi * 1.2 * t)
    return {
        "signal": signal,
        "recording_id": recording,
        "camera_id": "1",
        "start_frame": start,
        "prediction": label + offset,
        "label": label,
        "label_stats": {"mean": mean, "std": std,
                        "min": mean - 2 * std, "max": mean + 2 * std},
    }


def _payload(windows, label_norm="zscore"):
    return {"windows": windows, "traces": ["ABP", "CVP"], "channels": ["R", "G", "B"],
            "fs": FS,
            "label_norms": {w["signal"]: label_norm for w in windows}}


def test_window_table_has_one_row_per_record():
    payload = _payload([_window("ABP", "P015_S01_R1_0_D", 0),
                        _window("ABP", "P015_S01_R1_0_D", 64),
                        _window("CVP", "P016_S01_R1_0_D", 0)])
    table = window_table(payload)
    assert len(table) == 3
    assert set(table["signal"]) == {"ABP", "CVP"}
    assert set(table["participant"]) == {"P015", "P016"}


def test_perfect_prediction_scores_zero_error():
    table = window_table(_payload([_window("ABP", "P015_S01_R1_0_D", 0)]))
    row = table.iloc[0]
    assert row["mae"] == pytest.approx(0.0, abs=1e-6)
    assert row["mae_physical"] == pytest.approx(0.0, abs=1e-4)
    assert row["pearson"] == pytest.approx(1.0, abs=1e-6)
    assert row["hr_error"] == pytest.approx(0.0, abs=1e-6)


def test_physical_error_scales_by_the_windows_own_std():
    """A constant normalised offset becomes offset x std in mmHg."""
    table = window_table(_payload([_window("ABP", "P015_S01_R1_0_D", 0,
                                           std=12.0, offset=0.25)]))
    row = table.iloc[0]
    assert row["mae"] == pytest.approx(0.25)
    assert row["mae_physical"] == pytest.approx(0.25 * 12.0, rel=1e-4)


def test_minmax_normalisation_uses_the_matching_inverse():
    payload = _payload([_window("ABP", "P015_S01_R1_0_D", 0, mean=90.0, std=12.0,
                                offset=0.25)], label_norm="minmax")
    # minmax_inverse scales by (max - min) = 4 * std = 48.
    assert window_table(payload).iloc[0]["mae_physical"] == pytest.approx(0.25 * 48.0,
                                                                         rel=1e-4)


def test_summarise_groups_and_counts():
    payload = _payload([_window("ABP", "P015_S01_R1_0_D", 0),
                        _window("ABP", "P016_S01_R1_0_D", 0),
                        _window("CVP", "P015_S01_R1_0_D", 0)])
    summary = summarise(window_table(payload), by=("signal",))
    assert list(summary.index) == ["ABP", "CVP"]
    assert summary.loc["ABP", "windows"] == 2
    assert "mae_physical" in summary.columns and "hr_mae" in summary.columns


def test_summarise_can_group_by_participant():
    payload = _payload([_window("ABP", "P015_S01_R1_0_D", 0),
                        _window("ABP", "P016_S01_R1_0_D", 0)])
    summary = summarise(window_table(payload), by=("signal", "participant"))
    assert list(summary.index) == [("ABP", "P015"), ("ABP", "P016")]


def test_short_windows_are_tabulated_without_hr_columns():
    payload = _payload([_window("ABP", "P015_S01_R1_0_D", 0, n=4)])
    table = window_table(payload)
    assert len(table) == 1
    assert "hr_error" not in table.columns
    assert np.isfinite(table.iloc[0]["mae"])


def test_find_pickle_accepts_a_run_directory(tmp_path):
    path = tmp_path / "run" / "saved_test_outputs" / "model_outputs.pickle"
    path.parent.mkdir(parents=True)
    path.write_bytes(pickle.dumps(_payload([_window("ABP", "P015_S01_R1_0_D", 0)])))
    assert find_pickle(tmp_path) == path
    assert find_pickle(path) == path


def test_find_pickle_refuses_an_ambiguous_directory(tmp_path):
    for name in ("a", "b"):
        p = tmp_path / name / "x_outputs.pickle"
        p.parent.mkdir(parents=True)
        p.write_bytes(pickle.dumps(_payload([])))
    with pytest.raises(SystemExit, match="Several output pickles"):
        find_pickle(tmp_path)


def test_find_pickle_reports_an_empty_directory(tmp_path):
    with pytest.raises(SystemExit, match="No .*_outputs.pickle"):
        find_pickle(tmp_path)
