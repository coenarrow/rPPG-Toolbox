"""Heart-rate agreement — what the upstream toolbox called *the* evaluation.

Now one family among several. The engine is unchanged
(``post_process.calculate_metric_per_video``); what changed is that it is
computed per signal and lands in the same tidy frame as everything else,
rather than being the whole report.
"""

import numpy as np

from evaluation.post_process import calculate_metric_per_video
from evaluation.uncertainty import mean_se

#: Shortest window ``filtfilt`` can pad. Below this the rate family declines to
#: report and the other families carry on.
MIN_HR_WINDOW = 9


def rate_metrics(prediction, label, *, fs, hr_method="FFT") -> dict:
    """One window's rate agreement, or ``{}`` when the window is too short."""
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(label, dtype=np.float64)
    if pred.size < MIN_HR_WINDOW:
        return {}
    gt_hr, pred_hr, snr, macc = calculate_metric_per_video(
        pred, ref, diff_flag=False, fs=fs, hr_method=hr_method)
    nan = float("nan")
    return {
        "gt_hr": (float(gt_hr), nan),
        "pred_hr": (float(pred_hr), nan),
        "hr_error": (float(pred_hr - gt_hr), nan),
        "snr": (float(snr), nan),
        "macc": (float(macc), nan),
    }


def aggregate_rate(rows) -> dict:
    """Reduce many windows' ``rate_metrics`` to the reported summary.

    Windows are treated as independent units here — they are separate
    measurement occasions, not consecutive samples of one series — so the
    standard errors are naive rather than HAC.
    """
    if not rows:
        return {}
    gt = np.array([row["gt_hr"][0] for row in rows])
    pred = np.array([row["pred_hr"][0] for row in rows])
    snr = np.array([row["snr"][0] for row in rows])
    macc = np.array([row["macc"][0] for row in rows])
    error = pred - gt

    mae, _, mae_se = mean_se(np.abs(error), method="naive")
    mean_square, _, square_se = mean_se(error ** 2, method="naive")
    rmse = float(np.sqrt(mean_square))
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.abs(error / gt)
    relative = relative[np.isfinite(relative)]
    mape, _, mape_se = mean_se(relative * 100, method="naive")

    # Undefined for fewer than three points or a constant series — which a
    # short split, or a model predicting one rate, produces. Say nan rather
    # than raising.
    if pred.size >= 3 and pred.std() > 0 and gt.std() > 0:
        r = float(np.corrcoef(pred, gt)[0, 1])
        r_se = float(np.sqrt(max(1 - r ** 2, 0.0) / (pred.size - 2)))
    else:
        r = r_se = float("nan")

    snr_mean, _, snr_se = mean_se(snr, method="naive")
    macc_mean, _, macc_se = mean_se(macc, method="naive")
    return {
        "mae": (float(mae), float(mae_se)),
        "rmse": (rmse, float(square_se / (2 * rmse)) if rmse > 0 else float("nan")),
        "mape": (float(mape), float(mape_se)),
        "pearson": (r, r_se),
        "snr": (float(snr_mean), float(snr_se)),
        "macc": (float(macc_mean), float(macc_se)),
    }
