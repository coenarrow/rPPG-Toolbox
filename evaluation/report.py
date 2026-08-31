"""Assemble every family at every level into one tidy frame, then say it.

The frame — ``level x unit x signal x metric x value x se`` — is what the
digest, the CSV, the JSON and every plot are views over. It is also why the
trainer and the sweep reporter are the same code: they call this with
different records and get whichever levels those records can support.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.beats import detection_quality
from evaluation.levels import sections, unit_id
from evaluation.scoring import families_for, standards
from evaluation.scoring.clinical import beat_errors, grade_ieee1708, iso81060_3_verdict
from evaluation.scoring.rate import aggregate_rate, rate_metrics
from evaluation.scoring.waveform import waveform_metrics
from neural_methods.signals import beat_labels, is_absolute, signal_unit

FRAME_COLUMNS = ("level", "unit_id", "signal", "metric", "statistic",
                 "value", "se", "n")


def _row(level, unit, signal, metric, value, se=float("nan"), statistic="",
         n=1, **attrs):
    return {"level": level, "unit_id": unit, "signal": signal, "metric": metric,
            "statistic": statistic, "value": float(value), "se": float(se),
            "n": int(n), **attrs}


def _participant(window) -> str:
    """The store's own normalised participant id, with a legible fallback."""
    return window.attrs.get("participant") or window.recording_id.split("_")[0]


def build_frame(run, *, bootstrap=0, hr_method="FFT") -> pd.DataFrame:
    """Every metric the records can support, as one long frame."""
    rows = []
    for signal in run.signals():
        windows = [w for w in run.windows if w.signal == signal]
        families = families_for(signal)
        rows.extend(_window_rows(windows, signal, run.fs, families, bootstrap,
                                 hr_method))
        if "clinical" in families and run.label_norms.get(signal) == "raw":
            rows.extend(_clinical_rows(windows, signal, run.fs))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=list(FRAME_COLUMNS))
    ordered = list(FRAME_COLUMNS) + [c for c in frame.columns
                                     if c not in FRAME_COLUMNS]
    return frame[ordered]


def _window_rows(windows, signal, fs, families, bootstrap, hr_method) -> list:
    rows, rate_rows = [], []
    for window in windows:
        unit = unit_id("window", recording_id=window.recording_id,
                       camera_id=window.camera_id, start_frame=window.start_frame)
        # Store attrs first: a store that already carries `participant` must
        # not collide with the key we derive. dict(**attrs, participant=...)
        # would raise TypeError on exactly the stores we care about.
        common = {**window.attrs,
                  "recording_id": window.recording_id,
                  "camera_id": window.camera_id,
                  "participant": _participant(window)}
        if "waveform" in families:
            for metric, (value, se) in waveform_metrics(
                    window.prediction, window.label, fs=fs,
                    bootstrap=bootstrap).items():
                rows.append(_row("window", unit, signal, metric, value, se,
                                 **common))
        if "rate" in families:
            measured = rate_metrics(window.prediction, window.label, fs=fs,
                                    hr_method=hr_method)
            if measured:
                rate_rows.append(measured)
                for metric, (value, se) in measured.items():
                    rows.append(_row("window", unit, signal, f"rate_{metric}",
                                     value, se, **common))
    for metric, (value, se) in aggregate_rate(rate_rows).items():
        rows.append(_row("cohort", "all", signal, f"rate_{metric}", value, se,
                         n=len(rate_rows)))
    return rows


def _clinical_rows(windows, signal, fs) -> list:
    """Beat, section, recording, participant and cohort rows for one signal."""
    rows, per_subject = [], {}
    for section in sections(windows):
        section_unit = unit_id("section", recording_id=section.recording_id,
                               camera_id=section.camera_id,
                               section_index=section.index)
        participant = section.attrs.get("participant") or \
            section.recording_id.split("_")[0]
        common = {**section.attrs,
                  "recording_id": section.recording_id,
                  "camera_id": section.camera_id,
                  "participant": participant}

        errors = beat_errors(section.prediction, section.label, fs)
        for statistic, values in errors.items():
            for index, value in enumerate(values):
                beat_unit = unit_id(
                    "beat", recording_id=section.recording_id,
                    camera_id=section.camera_id, section_index=section.index,
                    beat_index=index)
                rows.append(_row("beat", beat_unit, signal, "beat_error", value,
                                 statistic=statistic, **common))
            if values.size:
                rows.append(_row("section", section_unit, signal, "bias",
                                 float(values.mean()), statistic=statistic,
                                 n=values.size, **common))
                per_subject.setdefault(participant, {}).setdefault(
                    statistic, []).append(float(values.mean()))

        quality = detection_quality(section.prediction, section.label, fs)
        for metric, value in quality.items():
            rows.append(_row("section", section_unit, signal,
                             f"detection_{metric}", value, **common))

    for statistic in ("max", "mean", "min"):
        subject_means = {name: float(np.mean(stats[statistic]))
                         for name, stats in per_subject.items()
                         if stats.get(statistic)}
        for participant, value in subject_means.items():
            rows.append(_row("participant", unit_id("participant",
                                                    participant=participant),
                             signal, "bias", value, statistic=statistic,
                             participant=participant))
        if not subject_means:
            continue
        errors = np.array(list(subject_means.values()))
        verdict = iso81060_3_verdict(errors)
        passes = verdict["passes"]
        rows.append(_row("cohort", "all", signal, "iso81060_3_passes",
                         float("nan") if passes is None else float(passes),
                         statistic=statistic, n=verdict["n_subjects"]))
        rows.append(_row("cohort", "all", signal, "mean_error",
                         verdict["mean_error"], verdict["se"],
                         statistic=statistic, n=verdict["n_subjects"]))
        rows.append(_row("cohort", "all", signal, "sd", verdict["sd"],
                         statistic=statistic, n=verdict["n_subjects"]))
        grade = grade_ieee1708(float(np.abs(errors).mean()))
        rows.append(_row("cohort", "all", signal, f"ieee1708_grade_{grade}",
                         float(np.abs(errors).mean()), statistic=statistic,
                         n=verdict["n_subjects"]))
    return rows


def digest(frame, run) -> str:
    """The fixed, readable summary. The CSV is where everything else lives."""
    lines = ["=== Evaluation report ==="]
    lines += [f"  {line}" for line in standards.provenance_lines()]
    for signal in run.signals():
        unit = signal_unit(signal)
        rows = frame[frame["signal"] == signal]
        if rows.empty:
            lines.append(f"[{signal}] no windows carried this label — skipped")
            continue
        windows = rows[rows["level"] == "window"]
        lines.append(f"--- {signal} ({unit}) ---")
        for metric in ("mae", "rmse", "pearson", "ccc"):
            values = windows[windows["metric"] == metric]["value"]
            if not values.empty:
                lines.append(f"  window {metric}: {values.mean():.4f} "
                             f"over {len(values)} windows")
        if is_absolute(signal):
            labels = beat_labels(signal)
            for statistic in ("max", "mean", "min"):
                bias = rows[(rows["level"] == "cohort") &
                            (rows["metric"] == "mean_error") &
                            (rows["statistic"] == statistic)]
                if bias.empty:
                    continue
                row = bias.iloc[0]
                lines.append(f"  {labels[statistic]}: bias {row['value']:+.2f} "
                             f"{unit} over {int(row['n'])} subjects")
    return "\n".join(lines)


def write(frame, digest_text, output_dir, filename_id):
    """Write the frame and the digest; return both paths."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = Path(output_dir) / f"{filename_id}_metrics.csv"
    json_path = Path(output_dir) / f"{filename_id}_report.json"
    frame.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"digest": digest_text,
                   "provenance": standards.provenance_lines(),
                   "study_design_unmet": standards.STUDY_DESIGN_REQUIREMENTS},
                  handle, indent=2)
    print(f"Saved metrics to {csv_path} and {json_path}")
    return csv_path, json_path
