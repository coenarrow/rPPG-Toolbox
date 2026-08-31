"""Standard errors that respect autocorrelation.

Consecutive samples of a physiological signal are nowhere near independent, so
``std / sqrt(n)`` over a 150-sample window is optimistic by a large factor.
Lifted from the PhysHydra-era analysis: a HAC (Newey-West) estimator with
Andrews (1991) automatic bandwidth selection, and a moving-block bootstrap for
the statistics with no usable closed form.

Applies to the DL-side metrics only. Clinical numbers use the standards' own
prescribed aggregation — see ``evaluation/metrics/standards.py``.
"""

import numpy as np

#: Andrews (1991) optimal-bandwidth constants and exponents, per kernel.
_ANDREWS = {
    "Bartlett": (1.1447, 1 / 3),
    "QS": (1.3221, 1 / 5),
    "Parzen": (2.6614, 1 / 5),
}


def _andrews_bandwidth(centred, n_samples, kernel) -> int:
    """Optimal maximum lag from a fitted AR(1) coefficient."""
    lag0 = float(centred @ centred) / n_samples
    lag1 = float(centred[1:] @ centred[:-1]) / n_samples
    rho = lag1 / lag0 if lag0 > 0 else 0.0
    rho = float(np.clip(rho, -0.97, 0.97))     # keep the formulas finite
    # Exactly the forms the PhysHydra analysis used — do not "simplify" them.
    if kernel == "Bartlett":
        alpha = 4 * rho ** 2 / ((1 - rho ** 2) ** 2 * (1 + rho ** 2))
    elif kernel == "QS":
        alpha = 4 * rho ** 2 / (1 - rho ** 2) ** 2
    elif kernel == "Parzen":
        alpha = 4 * rho ** 2 / (1 - rho ** 2)
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}")
    constant, exponent = _ANDREWS[kernel]
    return int(min(max(1, constant * (alpha * n_samples) ** exponent),
                   n_samples - 1))


def _kernel_weight(kernel, lag, max_lag) -> float:
    x = lag / (max_lag + 1.0)
    if kernel == "Bartlett":
        return 1.0 - x
    if kernel == "Parzen":
        return 2.0 * (1.0 - x) ** 3 if x > 0.5 else 1.0 - 6 * x ** 2 + 6 * x ** 3
    if kernel == "QS":
        z = 6.0 * np.pi * x / 5.0
        return 25.0 / (12.0 * np.pi ** 2 * x ** 2) * (np.sin(z) / z - np.cos(z))
    raise ValueError(f"Unknown kernel: {kernel!r}")


def mean_se(signal, method="HAC", kernel="QS", bandwidth="auto", fs=30):
    """``(mean, sd, se)`` for a 1-D series.

    ``method='naive'`` assumes i.i.d. samples and is right for a series whose
    elements are already independent units (one value per subject, say).
    ``method='HAC'`` is right within a recording.
    """
    values = np.asarray(signal, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    n_samples = values.size
    if n_samples == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if n_samples == 1:
        return mean, float("nan"), float("nan")
    sd = float(values.std(ddof=1))
    if method == "naive":
        return mean, sd, sd / np.sqrt(n_samples)
    if method != "HAC":
        raise ValueError(f"Unknown method: {method!r}. Use 'naive' or 'HAC'")

    centred = values - mean
    max_lag = (_andrews_bandwidth(centred, n_samples, kernel)
               if bandwidth == "auto"
               else int(min(bandwidth * fs, n_samples - 1)))
    variance = float(centred @ centred) / n_samples
    for lag in range(1, max_lag + 1):
        gamma = float(centred[lag:] @ centred[:-lag]) / n_samples
        variance += 2.0 * _kernel_weight(kernel, lag, max_lag) * gamma
    variance = max(variance, 0.0)               # kernels can undershoot
    return mean, sd, float(np.sqrt(variance / n_samples))


def moving_block_bootstrap(stat_fn, pred, label, resamples=500, seed=0) -> float:
    """Standard error of ``stat_fn(pred, label)`` under block resampling.

    Blocks of length ``n ** (1/3)`` keep the local autocorrelation intact,
    which an i.i.d. bootstrap would destroy. Seeded, so a report is
    reproducible.
    """
    pred = np.asarray(pred, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    n_samples = pred.size
    if n_samples < 4:
        return float("nan")
    rng = np.random.default_rng(seed)
    block = max(2, int(round(n_samples ** (1 / 3))))
    n_blocks = int(np.ceil(n_samples / block))
    starts = rng.integers(0, n_samples - block + 1, size=(resamples, n_blocks))
    offsets = np.arange(block)
    values = np.empty(resamples, dtype=np.float64)
    for i, row in enumerate(starts):
        index = (row[:, None] + offsets[None, :]).ravel()[:n_samples]
        values[i] = stat_fn(pred[index], label[index])
    values = values[np.isfinite(values)]
    return float(values.std(ddof=1)) if values.size > 1 else float("nan")
