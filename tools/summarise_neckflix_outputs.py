"""Turn a saved Neckflix test pickle into a per-window table and a summary.

``MultiSignalTrainer`` writes one self-describing record per scored window,
carrying the prediction, the label, and the ``label_stats`` that normalised
them — so everything here can be recomputed offline, including the inverse back
to physical units, without touching the zarr cache.

    uv run python tools/summarise_neckflix_outputs.py <run_dir_or_pickle>
    uv run python tools/summarise_neckflix_outputs.py <pickle> --csv windows.csv

Grouping defaults to the signal; ``--by`` adds columns (``participant``,
``recording_id``, ``camera_id``) so a LOSO sweep can be broken down per subject.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from dataset.data_loader.label_transforms import INVERSES as _INVERSES  # noqa: E402
from evaluation.post_process import calculate_metric_per_video  # noqa: E402

#: Shortest window the HR post-processing can filter (filtfilt padlen).
MIN_HR_WINDOW = 9


def find_pickle(target: Path) -> Path:
    """Accept a pickle, or any run directory containing exactly one."""
    if target.is_file():
        return target
    candidates = sorted(target.rglob("*_outputs.pickle"))
    if not candidates:
        raise SystemExit(f"No *_outputs.pickle under {target}")
    if len(candidates) > 1:
        listing = "\n  ".join(str(c) for c in candidates)
        raise SystemExit(f"Several output pickles under {target}; name one:\n  {listing}")
    return candidates[0]


def _to_physical(values, stats, label_norm):
    """Invert the per-window normalisation using the stats that produced it."""
    import torch

    inverse = _INVERSES[label_norm]
    tensor_stats = {k: torch.tensor(float(v)) for k, v in stats.items()}
    return inverse(torch.as_tensor(np.asarray(values), dtype=torch.float32),
                   tensor_stats).numpy()


def _safe_pearson(prediction, label):
    p = prediction - prediction.mean()
    l = label - label.mean()
    denominator = np.sqrt((p ** 2).sum() * (l ** 2).sum())
    return float((p * l).sum() / denominator) if denominator > 0 else np.nan


def window_table(payload, hr_method="FFT") -> pd.DataFrame:
    """One row per scored window, with per-window metrics recomputed."""
    fs = payload["fs"]
    # Per signal: a raw-mode signal is already in mmHg and inverts by identity,
    # a z-scored one inverts with the window's own stats.
    norms = payload["label_norms"]
    rows = []
    for record in payload["windows"]:
        prediction = np.asarray(record["prediction"], dtype=np.float64)
        label = np.asarray(record["label"], dtype=np.float64)
        norm = norms[record["signal"]]
        physical_pred = _to_physical(prediction, record["label_stats"], norm)
        physical_label = _to_physical(label, record["label_stats"], norm)
        error = physical_pred - physical_label
        row = {
            "signal": record["signal"],
            "recording_id": record["recording_id"],
            "participant": str(record["recording_id"]).split("_")[0],
            "camera_id": record["camera_id"],
            "start_frame": record["start_frame"],
            "pearson": _safe_pearson(prediction, label),
            "mae": float(np.mean(np.abs(prediction - label))),
            "rmse": float(np.sqrt(np.mean((prediction - label) ** 2))),
            "mae_physical": float(np.mean(np.abs(error))),
            "rmse_physical": float(np.sqrt(np.mean(error ** 2))),
            "label_mean": float(record["label_stats"]["mean"]),
            "label_std": float(record["label_stats"]["std"]),
        }
        if len(prediction) >= MIN_HR_WINDOW:
            gt_hr, pred_hr, snr, macc = calculate_metric_per_video(
                prediction, label, diff_flag=False, fs=fs, hr_method=hr_method)
            row.update(gt_hr=gt_hr, pred_hr=pred_hr, hr_error=pred_hr - gt_hr,
                       snr=snr, macc=macc)
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(table: pd.DataFrame, by=("signal",)) -> pd.DataFrame:
    columns = [c for c in ("pearson", "mae", "rmse", "mae_physical", "rmse_physical",
                           "macc", "snr") if c in table]
    summary = table.groupby(list(by))[columns].mean()
    summary.insert(0, "windows", table.groupby(list(by)).size())
    if "hr_error" in table:
        summary["hr_mae"] = table.groupby(list(by))["hr_error"].apply(
            lambda e: e.abs().mean())
        summary["hr_rmse"] = table.groupby(list(by))["hr_error"].apply(
            lambda e: np.sqrt((e ** 2).mean()))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path,
                        help="a *_outputs.pickle, or a run directory containing one")
    parser.add_argument("--by", nargs="+", default=["signal"],
                        help="grouping columns (default: signal)")
    parser.add_argument("--hr_method", default="FFT", choices=("FFT", "Peak"))
    parser.add_argument("--csv", type=Path, default=None,
                        help="also write the per-window table here")
    args = parser.parse_args()

    path = find_pickle(args.target)
    payload = pickle.loads(path.read_bytes())
    print(f"{path}\n  traces={payload['traces']}  channels={payload['channels']}  "
          f"fs={payload['fs']}  label_norms={payload['label_norms']}  "
          f"windows={len(payload['windows'])}\n")

    table = window_table(payload, hr_method=args.hr_method)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summarise(table, by=args.by).round(4))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv, index=False)
        print(f"\nPer-window table written to {args.csv}")


if __name__ == "__main__":
    main()
