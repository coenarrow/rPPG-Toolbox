"""The one smoke test for the per-window evaluation: levels and heart rate."""

import numpy as np
import torch

from src.trainer import to_physical

#: Every synthetic trace beats at 1.2 Hz.
BEAT_BPM = 72.0


def _record(participant, recording, start, abp_mask, cvp_mask, seed):
    """One synthetic 6 s window at 30 fps with ABP, CVP and ECG."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(180, dtype=torch.float32) / 30
    abp = 100 + 20 * torch.sin(2 * np.pi * 1.2 * t)
    cvp = 6 + 3 * torch.sin(2 * np.pi * 1.2 * t)
    ecg = 5 * torch.sin(2 * np.pi * 1.2 * t) + 0.5 * torch.randn(180, generator=g)

    def noise():
        return 2 * torch.randn(180, generator=g)

    def stats(x):
        return {"mean": x.mean(), "std": x.std(), "min": x.amin(), "max": x.amax()}

    return {
        "predictions": {"ABP": abp + noise(), "CVP": cvp + noise(), "ECG": ecg + noise()},
        "labels": {"ABP": abp, "CVP": cvp, "ECG": ecg},
        "label_stats": {s: stats(x) for s, x in (("ABP", abp), ("CVP", cvp), ("ECG", ecg))},
        "channel_mask": {"R": torch.tensor(True)},
        "label_mask": {"ABP": torch.tensor(abp_mask), "CVP": torch.tensor(cvp_mask),
                       "ECG": torch.tensor(True)},
        "metadata": {"dataset": "neckflix", "participant": participant,
                     "recording": recording, "perspective": "1",
                     "start_frame": torch.tensor(start)},
    }


def test_to_physical_inverts_zscore_and_leaves_raw():
    record = _record("1", "r", 0, True, True, 0)
    stats = record["label_stats"]["ECG"]
    z = (record["labels"]["ECG"] - stats["mean"]) / stats["std"]
    normed = {**record, "labels": {**record["labels"], "ECG": z}}
    physical = to_physical(normed, {"ABP": "raw", "CVP": "raw", "ECG": "zscore"})
    torch.testing.assert_close(physical["labels"]["ECG"], record["labels"]["ECG"],
                               rtol=1e-4, atol=1e-3)
    assert physical["labels"]["ABP"] is record["labels"]["ABP"]
    assert physical["label_stats"] is record["label_stats"]


from src.evaluation.evaluate import SCALARS, digest, summary_table, window_table  # noqa: E402


def _records():
    return [_record("1", "P001_R1", 0, True, True, 0),
            _record("1", "P001_R1", 30, True, True, 1),
            _record("1", "P001_R2", 0, True, False, 2),
            _record("2", "P002_R1", 0, True, False, 3),
            _record("2", "P002_R1", 30, True, True, 4),
            _record("2", "P002_R2", 0, True, False, 5)]


def test_window_table_scores_absolute_signals_where_masked():
    windows = window_table(_records())
    assert (windows.signal == "ABP").sum() == 6
    assert (windows.signal == "CVP").sum() == 3
    assert "ECG" not in set(windows.signal)
    np.testing.assert_allclose(windows.err_max, windows.pred_max - windows.ref_max)
    assert list(windows.columns[:6]) == ["dataset", "participant", "recording",
                                         "perspective", "start_frame", "signal"]
    assert windows.start_frame.dtype.kind == "i"


def test_summary_has_bias_per_signal_per_scalar():
    summary = summary_table(window_table(_records()))
    bias = summary[summary.metric == "bias"]
    assert set(zip(bias.signal, bias.statistic)) == {
        (s, st) for s in ("ABP", "CVP") for st in SCALARS}
    waveform = summary[(summary.statistic == "waveform") & (summary.signal == "ABP")]
    assert set(waveform.metric) == {"mae", "rmse", "pearson", "ccc"}
    assert (waveform.n == 6).all()
    text = digest(summary, {"RESP": "shape-class and not cardiac; nothing to score yet"})
    assert "systolic" in text and "[RESP] skipped" in text


import pandas as pd  # noqa: E402

from src.evaluation.evaluate import HR, evaluate, main as evaluate_main, rate_table  # noqa: E402
from src.evaluation.plots import draw  # noqa: E402
from src.evaluation.rate import FUSED, MEDIAN  # noqa: E402
from src.evaluation.records import RECORDS_NAME  # noqa: E402


def test_rate_table_reads_the_beat_off_every_cardiac_trace_and_the_fusion():
    rates = rate_table(_records(), 30.0)
    # Three windows carry all of ABP, CVP, ECG and so fuse; three carry two.
    assert set(rates.source) == {"ABP", "CVP", "ECG", FUSED, MEDIAN}
    assert (rates.source == FUSED).sum() == 6 and (rates.source == "CVP").sum() == 3
    # 180 frames zero-padded to 256 bins puts the 1.2 Hz beat within one bin.
    np.testing.assert_allclose(rates.ref_hr, BEAT_BPM, atol=4)
    assert rates.loc[rates.source == FUSED, "macc"].isna().all()
    assert rates.loc[rates.source == "ABP", "snr"].notna().all()


def test_draw_writes_bland_altman_and_waveform_figures(tmp_path):
    records = _records()
    draw(window_table(records), rate_table(records, 30.0), records, 30.0, tmp_path)
    for name in ("ABP_max_bland_altman.pdf", "ABP_mean_bland_altman.pdf",
                 "ABP_min_bland_altman.pdf", "ABP_waveforms.pdf",
                 "CVP_max_bland_altman.pdf", "CVP_waveforms.pdf",
                 "HR_ECG_bland_altman.pdf", f"HR_{FUSED}_bland_altman.pdf"):
        assert (tmp_path / name).is_file()


def test_evaluate_writes_tables_digest_and_figures(tmp_path):
    windows, summary = evaluate(_records(), tmp_path, fs=30.0)
    assert len(windows) == 9 and not summary.empty
    for name in ("windows.csv", "rates.csv", "summary.csv", "digest.txt",
                 "ABP_max_bland_altman.pdf"):
        assert (tmp_path / name).is_file()
    hr = summary[summary.statistic == HR]
    assert set(hr.signal) == {"ABP", "CVP", "ECG", FUSED, MEDIAN}
    assert {"mae", "mape", "bias", "snr", "macc"} <= set(hr.metric)
    text = (tmp_path / "digest.txt").read_text(encoding="utf-8")
    assert "Heart rate" in text and FUSED in text and "[ECG] skipped" not in text


def test_cli_pools_runs_with_one_frame_rate(tmp_path):
    records = _records()
    for name, chunk in (("fold_a", records[:3]), ("fold_b", records[3:])):
        (tmp_path / name).mkdir()
        torch.save({"fs": 30.0, "windows": chunk}, tmp_path / name / RECORDS_NAME)
    out = tmp_path / "pooled"
    evaluate_main([str(tmp_path / "fold_a"), str(tmp_path / "fold_b"), "--out", str(out)])
    assert (out / "windows.csv").is_file()
    assert len(pd.read_csv(out / "windows.csv")) == 9
