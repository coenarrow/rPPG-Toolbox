"""Every figure the evaluation draws, once, never per model.

Per scored signal: a Bland-Altman pair per window scalar — reference
against predicted and difference against mean — and one waveform overlay of
a few windows. Per heart-rate source (each cardiac trace, the fused
spectrum, the median): the same Bland-Altman pair in bpm. Titles use the
registry's per-signal wording, so ABP reads systolic / MAP / diastolic and
CVP reads peak / mean / trough.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # write files; never open a window on a cluster
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.evaluation.evaluate import SCALARS
from neural_methods.signals import beat_labels, signal_unit

#: Windows overlaid per signal: enough to see whether a prediction tracks at
#: all across recordings, few enough to read.
OVERLAYS_PER_SIGNAL = 4


def _save(figure, path: Path) -> None:
    figure.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(figure)
    print(f"Saved {path.name} to {path.parent}.")


def _evenly_spaced(items, count):
    """``count`` items spread across the list, not its head — a test split is
    ordered by recording, so the first N windows all come from one."""
    if len(items) <= count:
        return list(items)
    picks = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in sorted(set(picks.tolist()))]


def _array(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def draw(windows, rates, records, fs: float, out_dir) -> None:
    """Every figure for every scored signal in ``windows`` and every
    heart-rate source in ``rates``, into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sig in windows["signal"].unique():
        rows = windows[windows["signal"] == sig]
        for statistic in SCALARS:
            bland_altman(rows[f"ref_{statistic}"], rows[f"pred_{statistic}"],
                         sig, beat_labels(sig)[statistic], signal_unit(sig),
                         out_dir / f"{sig}_{statistic}_bland_altman.pdf")
        waveforms([r for r in records if bool(r["label_mask"][sig])], sig, fs, out_dir)
    for source in rates["source"].unique():
        rows = rates[rates["source"] == source]
        bland_altman(rows["ref_hr"], rows["pred_hr"], source, "heart rate", "bpm",
                     out_dir / f"HR_{source}_bland_altman.pdf")


def bland_altman(ref, pred, sig: str, label: str, unit: str, path: Path) -> None:
    """Reference vs predicted, and difference vs mean with bias and limits."""
    ref, pred = np.asarray(ref, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if ref.size == 0:
        return
    diff, mean = pred - ref, (pred + ref) / 2
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1)) if diff.size > 1 else float("nan")

    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4.5))
    left.scatter(ref, pred, s=12, alpha=0.6, edgecolors="none")
    low, high = min(ref.min(), pred.min()), max(ref.max(), pred.max())
    left.plot([low, high], [low, high], linestyle="--", linewidth=1, color="0.4")
    left.set_xlabel(f"reference {label} ({unit})")
    left.set_ylabel(f"predicted {label} ({unit})")
    left.set_title("agreement", fontsize=10)

    right.scatter(mean, diff, s=12, alpha=0.6, edgecolors="none")
    right.axhline(bias, linestyle="--", color="0.3", label=f"bias {bias:+.2f} {unit}")
    if np.isfinite(sd):
        for limit in (bias + 1.96 * sd, bias - 1.96 * sd):
            right.axhline(limit, linestyle=":", color="0.5")
        right.set_title(f"Bland-Altman, limits +/- {1.96 * sd:.2f} {unit}", fontsize=10)
    else:
        right.set_title("Bland-Altman", fontsize=10)
    right.axhline(0.0, color="black", linewidth=0.8)
    right.set_xlabel(f"mean of reference and predicted ({unit})")
    right.set_ylabel(f"predicted - reference ({unit})")
    right.legend(fontsize=8, loc="upper right")

    figure.suptitle(f"{sig} {label} ({ref.size} windows)", fontsize=11)
    figure.tight_layout()
    _save(figure, path)


def waveforms(records, sig: str, fs: float, out_dir: Path) -> None:
    """Label and prediction overlaid for a few windows, in the signal's unit."""
    chosen = _evenly_spaced(records, OVERLAYS_PER_SIGNAL)
    if not chosen:
        return
    unit = signal_unit(sig)
    figure, axes = plt.subplots(len(chosen), 1, figsize=(9, 2.2 * len(chosen)),
                                squeeze=False)
    for axis, record in zip(axes[:, 0], chosen):
        label, pred = _array(record["labels"][sig]), _array(record["predictions"][sig])
        time = np.arange(label.size) / fs
        axis.plot(time, label, label="label", linewidth=1.2)
        axis.plot(time, pred, label="prediction", linewidth=1.0, alpha=0.85)
        meta = record["metadata"]
        axis.set_title(f"{meta['recording']} perspective {meta['perspective']} "
                       f"@ frame {int(meta['start_frame'])}", fontsize=8)
        axis.set_ylabel(unit, fontsize=8)
        axis.tick_params(labelsize=7)
    axes[-1, 0].set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    figure.suptitle(f"{sig} waveforms", fontsize=10)
    figure.tight_layout()
    _save(figure, out_dir / f"{sig}_waveforms.pdf")
