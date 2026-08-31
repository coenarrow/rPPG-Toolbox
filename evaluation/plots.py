"""Every figure the evaluation produces, drawn once and never per model.

Cross-model comparability is the point of a standard plot set: a plot written
inside a model's own trainer stops being comparable the moment a second model
draws its own version.
"""

import os

import matplotlib
matplotlib.use("Agg")            # write files; never open a window on a cluster
import matplotlib.pyplot as plt
import numpy as np

from neural_methods.signals import beat_labels, is_absolute, signal_unit

STANDARD_PLOTS = ("waveforms", "agreement", "clinical")

#: Windows overlaid per signal. Enough to see whether a prediction tracks at
#: all across different recordings, few enough to read.
OVERLAYS_PER_SIGNAL = 4


def _save(figure, output_dir, file_name):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, file_name)
    figure.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(figure)
    print(f"Saved {file_name} to {output_dir}.")


def _evenly_spaced(items, count):
    """``count`` items spread across the list, not the first ``count``.

    A test split is ordered by recording, so the first N windows all come from
    one recording and one posture — the least informative sample available.
    """
    if len(items) <= count:
        return items
    picks = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in sorted(set(picks.tolist()))]


def draw(frame, run, *, output_dir, filename_id, plots=STANDARD_PLOTS) -> None:
    """Draw the requested figures from the records and the tidy frame."""
    for signal in run.signals():
        windows = [w for w in run.windows if w.signal == signal]
        if "waveforms" in plots:
            _waveforms(windows, signal, run.fs, output_dir, filename_id)
        if "agreement" in plots and is_absolute(signal):
            _agreement(windows, signal, output_dir, filename_id)
        if "clinical" in plots and is_absolute(signal):
            _bland_altman(frame, signal, output_dir, filename_id)


def _waveforms(windows, signal, fs, output_dir, filename_id):
    """Prediction vs label for a few windows, in the signal's own units."""
    chosen = _evenly_spaced(windows, OVERLAYS_PER_SIGNAL)
    if not chosen:
        return
    unit = signal_unit(signal)
    figure, axes = plt.subplots(len(chosen), 1, figsize=(9, 2.2 * len(chosen)),
                                squeeze=False)
    for axis, window in zip(axes[:, 0], chosen):
        time = np.arange(len(window.label)) / fs
        axis.plot(time, window.label, label="label", linewidth=1.2)
        axis.plot(time, window.prediction, label="prediction", linewidth=1.0,
                  alpha=0.85)
        axis.set_title(f"{window.recording_id} cam{window.camera_id} "
                       f"@ frame {window.start_frame}", fontsize=8)
        axis.set_ylabel(unit, fontsize=8)
        axis.tick_params(labelsize=7)
    axes[-1, 0].set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    figure.suptitle(f"{filename_id} — {signal} waveforms", fontsize=10)
    figure.tight_layout()
    _save(figure, output_dir, f"{filename_id}_{signal}_waveforms.pdf")


def _agreement(windows, signal, output_dir, filename_id):
    """Predicted vs true window statistics, named for the signal at hand."""
    labels = beat_labels(signal)
    unit = signal_unit(signal)
    panels = (("mean", np.mean), ("max", np.max), ("min", np.min))
    figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for axis, (statistic, reduce_fn) in zip(np.atleast_1d(axes), panels):
        predicted = np.array([reduce_fn(w.prediction) for w in windows])
        truth = np.array([reduce_fn(w.label) for w in windows])
        axis.scatter(truth, predicted, s=12, alpha=0.5, edgecolors="none")
        limits = [min(truth.min(), predicted.min()),
                  max(truth.max(), predicted.max())]
        axis.plot(limits, limits, linestyle="--", linewidth=1, color="0.4")
        bias = float(np.mean(predicted - truth))
        error = float(np.mean(np.abs(predicted - truth)))
        axis.set_title(f"{labels[statistic]}\nbias {bias:+.1f} · "
                       f"MAE {error:.1f} {unit}", fontsize=9)
        axis.set_xlabel(f"reference ({unit})", fontsize=8)
        axis.set_ylabel(f"predicted ({unit})", fontsize=8)
        axis.tick_params(labelsize=7)
    figure.suptitle(f"{filename_id} — {signal} agreement "
                    f"({len(windows)} windows)", fontsize=10)
    figure.tight_layout()
    _save(figure, output_dir, f"{filename_id}_{signal}_agreement.pdf")


def _bland_altman(frame, signal, output_dir, filename_id):
    """Per-subject bias with the 95% limits of agreement, one panel per statistic."""
    rows = frame[(frame["signal"] == signal) & (frame["level"] == "participant")
                 & (frame["metric"] == "bias")]
    if rows.empty:
        return
    labels = beat_labels(signal)
    unit = signal_unit(signal)
    statistics = [s for s in ("max", "mean", "min")
                  if not rows[rows["statistic"] == s].empty]
    figure, axes = plt.subplots(1, len(statistics),
                                figsize=(4 * len(statistics), 4), squeeze=False)
    for axis, statistic in zip(axes[0], statistics):
        bias = rows[rows["statistic"] == statistic]["value"].to_numpy()
        mean_bias = float(bias.mean())
        sd = float(bias.std(ddof=1)) if bias.size > 1 else float("nan")
        axis.scatter(np.arange(bias.size), bias, s=18, alpha=0.7)
        axis.axhline(mean_bias, linestyle="--", color="0.3", label="mean bias")
        if np.isfinite(sd):
            for limit in (mean_bias + 1.96 * sd, mean_bias - 1.96 * sd):
                axis.axhline(limit, linestyle=":", color="0.5")
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_title(f"{labels[statistic]} — bias {mean_bias:+.1f} {unit}",
                       fontsize=9)
        axis.set_xlabel("subject", fontsize=8)
        axis.set_ylabel(f"predicted − reference ({unit})", fontsize=8)
        axis.tick_params(labelsize=7)
    figure.suptitle(f"{filename_id} — {signal} per-subject agreement", fontsize=10)
    figure.tight_layout()
    _save(figure, output_dir, f"{filename_id}_{signal}_bland_altman.pdf")
