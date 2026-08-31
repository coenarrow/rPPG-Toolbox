"""The shared HR-metric report: values, degenerate inputs, per-signal scoping."""
import numpy as np
import pytest

from evaluation.metrics_report import report_hr_metrics


class _Cfg(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _config(tmp_path):
    return _Cfg(MODE="only_test",
                LOG_PATH=str(tmp_path), RUN=_Cfg(exp_name="exp"))


def _report(tmp_path, gt, pred, metrics=("MAE", "RMSE", "MAPE", "Pearson", "SNR", "MACC"),
            **kwargs):
    lines = []
    result = report_hr_metrics(
        gt, pred, np.zeros(len(gt)), np.full(len(gt), 0.5),
        metrics=list(metrics), config=_config(tmp_path), filename_id="unit",
        printer=lines.append, **kwargs)
    return result, "\n".join(lines)


def test_values_match_the_definitions(tmp_path):
    gt = np.array([60.0, 70.0, 80.0, 90.0])
    pred = np.array([62.0, 68.0, 83.0, 88.0])
    result, _ = _report(tmp_path, gt, pred)
    errors = pred - gt
    assert result["n"] == 4
    assert result["MAE"] == pytest.approx(np.mean(np.abs(errors)))
    assert result["RMSE"] == pytest.approx(np.sqrt(np.mean(errors ** 2)))
    assert result["MAPE"] == pytest.approx(np.mean(np.abs(errors / gt)) * 100)
    assert result["Pearson"] == pytest.approx(np.corrcoef(pred, gt)[0][1])
    assert result["MACC"] == pytest.approx(0.5)


def test_empty_group_reports_nothing_rather_than_crashing(tmp_path):
    result, printed = _report(tmp_path, [], [])
    assert result == {"n": 0}
    assert "no evaluable windows" in printed


def test_pearson_is_declared_undefined_for_a_constant_series(tmp_path):
    """A short split, or a model predicting one rate, must not print nan +/- nan."""
    gt = np.array([70.0, 70.0, 70.0, 70.0])
    result, printed = _report(tmp_path, gt, np.array([65.0, 71.0, 68.0, 74.0]),
                              metrics=("Pearson",))
    assert np.isnan(result["Pearson"])
    assert "ground-truth HR is constant" in printed
    assert "nan +/- nan" not in printed


def test_pearson_is_declared_undefined_for_too_few_windows(tmp_path):
    result, printed = _report(tmp_path, [70.0, 75.0], [71.0, 74.0], metrics=("Pearson",))
    assert np.isnan(result["Pearson"])
    assert "needs >= 3 windows" in printed


def test_mape_ignores_a_zero_ground_truth(tmp_path):
    result, _ = _report(tmp_path, [0.0, 60.0, 90.0], [10.0, 66.0, 81.0],
                        metrics=("MAPE",))
    assert result["MAPE"] == pytest.approx(np.mean([6 / 60, 9 / 90]) * 100)


def test_scope_labels_every_line(tmp_path):
    _, printed = _report(tmp_path, [60.0, 70.0, 80.0], [61.0, 72.0, 79.0],
                         metrics=("MAE",), scope="CVP")
    assert printed.startswith("[CVP] ")


def test_unknown_metric_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported metric"):
        _report(tmp_path, [60.0, 70.0], [61.0, 72.0], metrics=("Sharpe",))


def test_bland_altman_writes_scoped_plots_even_for_degenerate_data(tmp_path):
    """gaussian_kde raises on a singular covariance; the plot still has to land."""
    gt = np.array([70.0, 70.0, 70.0, 70.0])
    _report(tmp_path, gt, gt.copy(), metrics=("BA",), scope="ABP")
    written = {p.name for p in (tmp_path / "exp" / "bland_altman_plots").iterdir()}
    assert written == {"unit_ABP_FFT_BlandAltman_ScatterPlot.pdf",
                       "unit_ABP_FFT_BlandAltman_DifferencePlot.pdf"}
