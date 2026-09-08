"""Per-window evaluation, with no config.

What is scored follows from the signal registry. Every absolute-class
signal (ABP, CVP) at least one window carries gets the window max / mean /
min of prediction and label — systolic / MAP / diastolic for ABP, peak /
mean / trough for CVP — and the sample-wise fidelity of the two traces.
Every cardiac trace (PPG, ECG, ABP, CVP) gets a heart rate, from itself and
from all of them fused (``src/evaluation/rate.py``). The **window tables are
the primary output**; the summary is derived from them and nothing else,
so pooling LOSO folds later is a groupby over the same columns.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation.rate import RATE_METRICS, window_rates
from src.evaluation.records import load_records
from neural_methods.signals import beat_labels, is_absolute, is_cardiac, signal_unit

SCALARS = ("max", "mean", "min")
WAVEFORM_METRICS = ("mae", "rmse", "pearson", "ccc")
META = ("dataset", "participant", "recording", "perspective", "start_frame")
WINDOW_COLUMNS = (*META, "signal",
                  *(f"{part}_{s}" for s in SCALARS for part in ("ref", "pred", "err")),
                  *WAVEFORM_METRICS)
RATE_COLUMNS = (*META, "source", *RATE_METRICS)
SUMMARY_COLUMNS = ("signal", "statistic", "metric", "value", "n")
#: The ``statistic`` the heart-rate rows of the summary carry.
HR = "hr"
_REDUCE = {"max": np.max, "mean": np.mean, "min": np.min}
_NAN = float("nan")


# ---------------------------------------------------------------------------
# Agreement between two sequences
# ---------------------------------------------------------------------------
def pearson(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return _NAN
    return float(np.corrcoef(a, b)[0, 1])


def ccc(a, b) -> float:
    """Lin's concordance: correlation penalised by disagreement in level."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    r = pearson(a, b)
    if not np.isfinite(r):
        return _NAN
    va, vb = a.var(ddof=1), b.var(ddof=1)
    denominator = va + vb + (a.mean() - b.mean()) ** 2
    return float(2 * r * np.sqrt(va * vb) / denominator) if denominator > 0 else _NAN


def _array(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


# ---------------------------------------------------------------------------
# Which signals, and the window table
# ---------------------------------------------------------------------------
def scored_signals(records) -> tuple[list, dict]:
    """``(absolute signals to score, {signal: why nothing is scored})``.

    Cardiac shape-class signals are not skipped: they have no level to
    score but they carry a heart rate, and the rate table takes them.
    """
    signals, skipped = [], {}
    if not records:
        return signals, skipped
    for sig in records[0]["predictions"]:
        if not any(bool(r["label_mask"][sig]) for r in records):
            skipped[sig] = "no window carried this label"
        elif is_absolute(sig):
            signals.append(sig)
        elif not is_cardiac(sig):
            skipped[sig] = "shape-class and not cardiac; nothing to score yet"
    return signals, skipped


def _metadata(record) -> dict:
    meta = {}
    for key in META:
        value = record["metadata"][key]
        meta[key] = int(value) if key == "start_frame" else str(value)
    return meta


def window_table(records) -> pd.DataFrame:
    """One row per window per scored signal, physical units throughout."""
    signals, _ = scored_signals(records)
    rows = []
    for record in records:
        meta = _metadata(record)
        for sig in signals:
            if not bool(record["label_mask"][sig]):
                continue
            pred, ref = _array(record["predictions"][sig]), _array(record["labels"][sig])
            row = {**meta, "signal": sig}
            for s, reduce_fn in _REDUCE.items():
                row[f"ref_{s}"] = float(reduce_fn(ref))
                row[f"pred_{s}"] = float(reduce_fn(pred))
                row[f"err_{s}"] = row[f"pred_{s}"] - row[f"ref_{s}"]
            error = pred - ref
            row["mae"] = float(np.abs(error).mean())
            row["rmse"] = float(np.sqrt((error ** 2).mean()))
            row["pearson"] = pearson(pred, ref)
            row["ccc"] = ccc(pred, ref)
            rows.append(row)
    return pd.DataFrame(rows, columns=list(WINDOW_COLUMNS))


def rate_table(records, fs: float) -> pd.DataFrame:
    """One row per window per heart-rate source: each cardiac trace the
    window carries, then the fused spectrum and the median of the traces."""
    rows = []
    for record in records:
        meta = _metadata(record)
        rows.extend({**meta, **row} for row in window_rates(record, fs))
    return pd.DataFrame(rows, columns=list(RATE_COLUMNS))


# ---------------------------------------------------------------------------
# The summary, derived from the window tables only
# ---------------------------------------------------------------------------
def _scalar_summary(ref, pred, err) -> dict:
    n = err.size
    bias = float(err.mean()) if n else _NAN
    sd = float(err.std(ddof=1)) if n > 1 else _NAN
    return {
        "mae": float(np.abs(err).mean()) if n else _NAN,
        "rmse": float(np.sqrt((err ** 2).mean())) if n else _NAN,
        "pearson": pearson(pred, ref),
        "bias": bias,
        "sd": sd,
        "loa_low": bias - 1.96 * sd,
        "loa_high": bias + 1.96 * sd,
    }


def _finite_mean(values) -> tuple[float, int]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return (float(finite.mean()) if finite.size else _NAN), int(finite.size)


def summary_table(windows: pd.DataFrame, rates: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per signal and scalar, the level agreement; per signal, the waveform
    fidelity; per heart-rate source (``statistic`` ``hr``), the rate
    agreement plus MAPE, SNR and MACC as the upstream toolbox reported them."""
    rows = []
    for sig, group in windows.groupby("signal", sort=False):
        for s in SCALARS:
            ref, pred, err = (group[f"{part}_{s}"].to_numpy(dtype=np.float64)
                              for part in ("ref", "pred", "err"))
            rows.extend({"signal": sig, "statistic": s, "metric": metric,
                         "value": value, "n": int(err.size)}
                        for metric, value in _scalar_summary(ref, pred, err).items())
        for metric in WAVEFORM_METRICS:
            value, n = _finite_mean(group[metric])
            rows.append({"signal": sig, "statistic": "waveform", "metric": metric,
                         "value": value, "n": n})
    if rates is not None:
        rows.extend(rate_summary(rates))
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def rate_summary(rates: pd.DataFrame) -> list:
    """The ``hr`` rows of the summary: per source, the rate agreement plus
    MAPE, SNR and MACC as the upstream toolbox reported them. Also what the
    unsupervised methods report, so their numbers are these numbers."""
    rows = []
    for source, group in rates.groupby("source", sort=False):
        ref, pred, err = (group[column].to_numpy(dtype=np.float64)
                          for column in ("ref_hr", "pred_hr", "err_hr"))
        summary = _scalar_summary(ref, pred, err)
        with np.errstate(divide="ignore", invalid="ignore"):
            summary["mape"], _ = _finite_mean(np.abs(err / ref) * 100)
        rows.extend({"signal": source, "statistic": HR, "metric": metric,
                     "value": value, "n": int(err.size)}
                    for metric, value in summary.items())
        for metric in ("snr", "macc"):
            value, n = _finite_mean(group[metric])
            rows.append({"signal": source, "statistic": HR, "metric": metric,
                         "value": value, "n": n})
    return rows


# ---------------------------------------------------------------------------
# The digest
# ---------------------------------------------------------------------------
def digest(summary: pd.DataFrame, skipped: dict) -> str:
    """The fixed, readable summary; the CSVs are where everything else lives."""
    levels, rates = summary[summary["statistic"] != HR], summary[summary["statistic"] == HR]
    lines = ["=== Evaluation: absolute-class signals, per window ==="]
    for sig in levels["signal"].unique():
        unit, labels = signal_unit(sig), beat_labels(sig)
        rows = levels[levels["signal"] == sig]

        def value(statistic, metric):
            hit = rows[(rows["statistic"] == statistic) & (rows["metric"] == metric)]
            return float(hit["value"].iloc[0]) if not hit.empty else _NAN

        lines.append(f"--- {sig} ({unit}) ---")
        for s in SCALARS:
            n = int(rows[rows["statistic"] == s]["n"].iloc[0])
            lines.append(
                f"  {labels[s]:>9}: MAE {value(s, 'mae'):.2f}  RMSE {value(s, 'rmse'):.2f}  "
                f"r {value(s, 'pearson'):.3f}  bias {value(s, 'bias'):+.2f} "
                f"+/- {value(s, 'sd'):.2f} {unit}  (n = {n})")
        lines.append(
            f"   waveform: MAE {value('waveform', 'mae'):.2f}  "
            f"RMSE {value('waveform', 'rmse'):.2f} {unit}  "
            f"r {value('waveform', 'pearson'):.3f}  CCC {value('waveform', 'ccc'):.3f}")
    if not rates.empty:
        lines.append("=== Heart rate (bpm), per window, each source against its own label ===")
        for source in rates["signal"].unique():
            rows = rates[rates["signal"] == source]

            def value(metric):
                hit = rows[rows["metric"] == metric]
                return float(hit["value"].iloc[0]) if not hit.empty else _NAN

            n = int(rows[rows["metric"] == "mae"]["n"].iloc[0])
            lines.append(
                f"  {source:>6}: MAE {value('mae'):.2f}  RMSE {value('rmse'):.2f}  "
                f"MAPE {value('mape'):.1f}%  r {value('pearson'):.3f}  "
                f"bias {value('bias'):+.2f} +/- {value('sd'):.2f}  "
                f"SNR {value('snr'):.1f} dB  MACC {value('macc'):.3f}  (n = {n})")
    for sig, reason in skipped.items():
        lines.append(f"[{sig}] skipped: {reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry points: one function, two callers
# ---------------------------------------------------------------------------
def evaluate(records, out_dir, fs: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every absolute-class signal and every heart-rate source in
    ``records``; write the tables, the plots and the digest into ``out_dir``;
    return ``(windows, summary)``."""
    from src.evaluation.plots import draw      # plots import this module's constants

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, skipped = scored_signals(records)
    windows = window_table(records)
    rates = rate_table(records, fs)
    summary = summary_table(windows, rates)
    windows.to_csv(out_dir / "windows.csv", index=False)
    rates.to_csv(out_dir / "rates.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    text = digest(summary, skipped)
    (out_dir / "digest.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    draw(windows, rates, records, fs, out_dir)
    return windows, summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate one run's test records, or pool several runs' "
                    "(a LOSO sweep) into one report.")
    parser.add_argument("run_dirs", nargs="+", metavar="RUN_DIR",
                        help="run directories holding test_records.pt")
    parser.add_argument("--out", metavar="DIR",
                        help="where the tables, digest and plots go (default: "
                             "the run directory; required when pooling)")
    args = parser.parse_args(argv)
    if args.out is None and len(args.run_dirs) > 1:
        parser.error("--out is required when pooling several runs")
    try:
        records, fs = load_records(args.run_dirs)
    except (FileNotFoundError, ValueError) as err:
        parser.error(str(err))
    evaluate(records, args.out or args.run_dirs[0], fs)


if __name__ == "__main__":
    main()
