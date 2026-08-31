"""Shared aggregation and printing of window-level HR metrics.

``evaluation.metrics`` (supervised) and ``unsupervised_methods`` (traditional)
both reduce a pile of per-window ``(ground-truth HR, predicted HR, SNR, MACC)``
tuples to the same handful of numbers and print them the same way. That
reduction lives here once, so a new evaluation path — like the per-signal
Neckflix one, which reports the same table for ABP, CVP and ECG separately —
gets it for free and stays comparable.
"""

import os

import matplotlib
matplotlib.use("Agg")            # write files; never open a window on a cluster
import matplotlib.pyplot as plt
import numpy as np

from dataset.data_loader.label_transforms import INVERSES
from evaluation.BlandAltmanPy import BlandAltman
from neural_methods.signals import is_absolute, signal_unit

#: Metric names understood by :func:`report_hr_metrics`.
SUPPORTED_METRICS = ("MAE", "RMSE", "MAPE", "Pearson", "SNR", "MACC", "BA")


def _standard_error(values, count):
    return float(np.std(values) / np.sqrt(count)) if count else float("nan")


def report_hr_metrics(gt_hr, pred_hr, snr, macc, *, metrics, config, filename_id,
                      hr_method="FFT", scope="", printer=print):
    """Compute, print and return the configured HR metrics for one group.

    ``scope`` labels the group in the printed lines (e.g. the signal a
    Neckflix run derived its reference HR from); it is empty for the
    single-group datasets. Returns ``{metric: value}`` plus ``n`` so callers can
    tabulate results without re-parsing stdout.
    """
    gt_hr = np.asarray(gt_hr, dtype=np.float64)
    pred_hr = np.asarray(pred_hr, dtype=np.float64)
    snr = np.asarray(snr, dtype=np.float64)
    macc = np.asarray(macc, dtype=np.float64)

    tag = f"[{scope}] " if scope else ""
    n = len(pred_hr)
    results = {"n": n}
    if n == 0:
        printer(f"{tag}no evaluable windows — nothing to report")
        return results

    errors = pred_hr - gt_hr
    for metric in metrics:
        if metric == "MAE":
            value = float(np.mean(np.abs(errors)))
            printer(f"{tag}{hr_method} MAE: {value} +/- {_standard_error(np.abs(errors), n)}")
        elif metric == "RMSE":
            # Standard error is taken on the squared errors, then rooted, so an
            # unusual error distribution cannot distort it.
            squared = np.square(errors)
            value = float(np.sqrt(np.mean(squared)))
            printer(f"{tag}{hr_method} RMSE: {value} +/- {float(np.sqrt(np.std(squared) / np.sqrt(n)))}")
        elif metric == "MAPE":
            with np.errstate(divide="ignore", invalid="ignore"):
                relative = np.abs(errors / gt_hr)
            relative = relative[np.isfinite(relative)]
            value = float(np.mean(relative) * 100) if relative.size else float("nan")
            printer(f"{tag}{hr_method} MAPE: {value} +/- {_standard_error(relative, relative.size) * 100}")
        elif metric == "Pearson":
            # Correlation is undefined for < 3 points or a constant series (which
            # a short evaluation split, or a model predicting one rate, produces).
            # Say so rather than printing "nan +/- nan".
            if n < 3:
                value = float("nan")
                printer(f"{tag}{hr_method} Pearson: undefined, needs >= 3 windows (got {n})")
            elif np.std(pred_hr) == 0 or np.std(gt_hr) == 0:
                value = float("nan")
                which = "predicted" if np.std(pred_hr) == 0 else "ground-truth"
                printer(f"{tag}{hr_method} Pearson: undefined, {which} HR is constant "
                        f"across all {n} windows")
            else:
                value = float(np.corrcoef(pred_hr, gt_hr)[0][1])
                printer(f"{tag}{hr_method} Pearson: {value} "
                        f"+/- {float(np.sqrt(max(1 - value ** 2, 0.0) / (n - 2)))}")
        elif metric == "SNR":
            value = float(np.mean(snr))
            printer(f"{tag}{hr_method} SNR: {value} +/- {_standard_error(snr, n)} (dB)")
        elif metric == "MACC":
            value = float(np.mean(macc))
            printer(f"{tag}MACC: {value} +/- {_standard_error(macc, n)}")
        elif "BA" in metric:
            _bland_altman_plots(gt_hr, pred_hr, config, filename_id, hr_method, scope)
            value = None
        else:
            raise ValueError(
                f"Unsupported metric {metric!r}; known: {', '.join(SUPPORTED_METRICS)}"
            )
        results[metric] = value
    return results


def _bland_altman_plots(gt_hr, pred_hr, config, filename_id, hr_method, scope):
    """Write the scatter and difference plots for one group."""
    suffix = f"_{scope}" if scope else ""
    compare = BlandAltman(gt_hr, pred_hr, config, averaged=True)
    stem = f"{filename_id}{suffix}_{hr_method}_BlandAltman"
    compare.scatter_plot(
        x_label='GT HR [bpm]',
        y_label='rPPG HR [bpm]',
        show_legend=True, figure_size=(5, 5),
        the_title=f'{stem}_ScatterPlot',
        file_name=f'{stem}_ScatterPlot.pdf')
    compare.difference_plot(
        x_label='Difference between rPPG HR and GT HR [bpm]',
        y_label='Average of rPPG HR and GT HR [bpm]',
        show_legend=True, figure_size=(5, 5),
        the_title=f'{stem}_DifferencePlot',
        file_name=f'{stem}_DifferencePlot.pdf')


# ---------------------------------------------------------------------------
# Per-signal waveform and agreement plots (migration contract §6, plots 3-4)
#
# Implemented here once, never per model: cross-model comparability is the whole
# point of a standard plot set, and a plot written inside a model's own trainer
# stops being comparable the moment a second model draws its own version.
# ---------------------------------------------------------------------------
#: Windows overlaid per signal in the waveform figure. Enough to see whether a
#: prediction tracks at all across different recordings, few enough to read.
OVERLAYS_PER_SIGNAL = 4

#: What an absolute-class agreement scatter has a panel for, and how each is
#: derived from one window of waveform. There is no separate stats head, so
#: these are read off the predicted waveform exactly as they are off the label.
_AGREEMENT_PANELS = (
    ("window mean", np.mean),
    ("systolic (max)", np.max),
    ("diastolic (min)", np.min),
)


def to_physical(record, mode):
    """One saved window's prediction and label back in physical units.

    ``mode`` is that signal's own label normalisation, so this is exact rather
    than approximate — the stats that normalised the window ride in the record.
    """
    import torch

    inverse = INVERSES[mode]
    stats = {k: torch.as_tensor(v, dtype=torch.float32)
             for k, v in record["label_stats"].items()}
    prediction = torch.as_tensor(np.asarray(record["prediction"]), dtype=torch.float32)
    label = torch.as_tensor(np.asarray(record["label"]), dtype=torch.float32)
    return inverse(prediction, stats).numpy(), inverse(label, stats).numpy()


def _evenly_spaced(records, count):
    """``count`` records spread across the list, not the first ``count``.

    A test split is ordered by recording, so the first N windows all come from
    one recording and one posture — which is the least informative sample of
    them available.
    """
    if len(records) <= count:
        return records
    picks = np.linspace(0, len(records) - 1, count).round().astype(int)
    return [records[i] for i in sorted(set(picks.tolist()))]


def _save(figure, output_dir, file_name):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, file_name)
    figure.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(figure)
    print(f"Saved {file_name} to {output_dir}.")


def plot_waveform_overlays(records, norms, *, output_dir, filename_id, fs=None,
                           per_signal=OVERLAYS_PER_SIGNAL):
    """Prediction vs label for a few test windows, one figure per signal.

    Absolute-class signals are drawn in their physical units (mmHg), so the plot
    shows level agreement and not just shape; shape-class signals are drawn in
    the normalised space they are predicted in. Each panel is titled with the
    window's own metadata, because "which recording was that?" is the first
    question a suspicious-looking trace raises.
    """
    by_signal = {}
    for record in records:
        by_signal.setdefault(record["signal"], []).append(record)

    for signal, group in by_signal.items():
        chosen = _evenly_spaced(group, per_signal)
        if not chosen:
            continue
        physical = is_absolute(signal)
        unit = signal_unit(signal) if physical else "normalised"
        figure, axes = plt.subplots(len(chosen), 1, figsize=(9, 2.2 * len(chosen)),
                                    squeeze=False)
        for axis, record in zip(axes[:, 0], chosen):
            if physical:
                prediction, label = to_physical(record, norms[signal])
            else:
                prediction = np.asarray(record["prediction"])
                label = np.asarray(record["label"])
            time = (np.arange(len(label)) / fs) if fs else np.arange(len(label))
            axis.plot(time, label, label="label", linewidth=1.2)
            axis.plot(time, prediction, label="prediction", linewidth=1.0, alpha=0.85)
            axis.set_title(
                f"{record['recording_id']} cam{record['camera_id']} "
                f"@ frame {record['start_frame']}", fontsize=8)
            axis.set_ylabel(unit, fontsize=8)
            axis.tick_params(labelsize=7)
        axes[-1, 0].set_xlabel("time (s)" if fs else "sample")
        axes[0, 0].legend(fontsize=7, loc="upper right")
        figure.suptitle(f"{filename_id} — {signal} waveforms", fontsize=10)
        figure.tight_layout()
        _save(figure, output_dir, f"{filename_id}_{signal}_waveforms.pdf")


def plot_absolute_agreement(records, norms, *, output_dir, filename_id):
    """Predicted vs true window mean, systolic and diastolic, with the identity line.

    Only absolute-class signals get one: for a per-window normalised signal the
    label statistics are 0 and 1 by construction, so the scatter would measure
    nothing. This is the migration-time precursor of the Phase 7 clinical
    agreement analysis (IEEE 1708 / ISO 81060 bands), which consumes the same
    saved windows.
    """
    by_signal = {}
    for record in records:
        if is_absolute(record["signal"]):
            by_signal.setdefault(record["signal"], []).append(record)

    for signal, group in by_signal.items():
        pairs = [to_physical(record, norms[signal]) for record in group]
        unit = signal_unit(signal)
        figure, axes = plt.subplots(1, len(_AGREEMENT_PANELS),
                                    figsize=(4 * len(_AGREEMENT_PANELS), 4))
        for axis, (title, reduce_fn) in zip(np.atleast_1d(axes), _AGREEMENT_PANELS):
            predicted = np.array([reduce_fn(p) for p, _ in pairs])
            truth = np.array([reduce_fn(l) for _, l in pairs])
            axis.scatter(truth, predicted, s=12, alpha=0.5, edgecolors="none")
            limits = [min(truth.min(), predicted.min()),
                      max(truth.max(), predicted.max())]
            axis.plot(limits, limits, linestyle="--", linewidth=1, color="0.4")
            bias = float(np.mean(predicted - truth))
            error = float(np.mean(np.abs(predicted - truth)))
            axis.set_title(f"{title}\nbias {bias:+.1f} · MAE {error:.1f} {unit}",
                           fontsize=9)
            axis.set_xlabel(f"reference ({unit})", fontsize=8)
            axis.set_ylabel(f"predicted ({unit})", fontsize=8)
            axis.tick_params(labelsize=7)
        figure.suptitle(f"{filename_id} — {signal} agreement "
                        f"({len(group)} windows)", fontsize=10)
        figure.tight_layout()
        _save(figure, output_dir, f"{filename_id}_{signal}_agreement.pdf")
