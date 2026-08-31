"""Shape and error agreement for one pair of traces, in physical units."""

import numpy as np

from evaluation.post_process import _compute_macc
from evaluation.uncertainty import mean_se, moving_block_bootstrap

_NAN = (float("nan"), float("nan"))


def _pearson(a, b) -> float:
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _ccc(a, b) -> float:
    """Lin's concordance: correlation penalised by disagreement in level."""
    if a.size < 2:
        return float("nan")
    r = _pearson(a, b)
    if not np.isfinite(r):
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    denominator = va + vb + (a.mean() - b.mean()) ** 2
    return float(2 * r * np.sqrt(va * vb) / denominator) if denominator > 0 else float("nan")


def waveform_metrics(prediction, label, *, fs, bootstrap=0) -> dict:
    """``{metric: (value, se)}`` for one prediction/label pair.

    Standard errors are HAC where the samples are autocorrelated, and bootstrap
    for Pearson and CCC, which have no usable closed form here. The bootstrap
    is the one expensive computation in the report, so it is opt-in.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(label, dtype=np.float64)
    if pred.size == 0 or pred.size != ref.size:
        return {name: _NAN for name in ("mae", "rmse", "pearson", "ccc", "macc")}

    error = pred - ref
    mae, _, mae_se = mean_se(np.abs(error), method="HAC", fs=fs)
    mean_square, _, square_se = mean_se(error ** 2, method="HAC", fs=fs)
    rmse = float(np.sqrt(mean_square))
    # Delta method: se(sqrt(x)) = se(x) / (2 sqrt(x)).
    rmse_se = square_se / (2 * rmse) if rmse > 0 else float("nan")

    r, ccc = _pearson(pred, ref), _ccc(pred, ref)
    if bootstrap:
        r_se = moving_block_bootstrap(_pearson, pred, ref, resamples=bootstrap)
        ccc_se = moving_block_bootstrap(_ccc, pred, ref, resamples=bootstrap)
    else:
        r_se = ccc_se = float("nan")

    return {
        "mae": (float(mae), float(mae_se)),
        "rmse": (rmse, float(rmse_se)),
        "pearson": (r, r_se),
        "ccc": (ccc, ccc_se),
        "macc": (float(_compute_macc(pred, ref)), float("nan")),
    }
