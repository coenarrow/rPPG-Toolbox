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


def _normalise_participant(name) -> str:
    """The cache's own convention: ``P015`` -> ``015``, leading zeros kept."""
    text = str(name)
    return text[1:] if text[:1] in ("P", "p") else text


def _participant(record) -> str:
    """The store's own normalised participant id, with a legible fallback.

    Both a ``WindowRecord`` and a ``Section`` answer here. The fallback is
    normalised the way the store writes the attr, so pooling a fold whose
    pickle predates ``attrs`` with one that carries it cannot split one person
    into two subjects — which would also inflate ``n_subjects`` past the
    verdict threshold and grade a cohort of one.
    """
    stored = record.attrs.get("participant")
    if stored:
        return _normalise_participant(stored)
    return _normalise_participant(record.recording_id.split("_")[0])


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
    """Beat, section, recording, participant and cohort rows for one signal.

    Two accumulators, deliberately side by side: ISO 81060-3 wants a *signed*
    per-subject bias, IEEE 1708 wants a per-subject *mean absolute* error. The
    absolute value is taken at the beat, never after averaging — a subject
    whose errors alternate in sign would otherwise grade as if they were zero.
    """
    rows = []
    signed_sections = {}      # participant -> statistic -> [section mean]
    absolute_beats = {}       # participant -> statistic -> [|beat errors|]
    recording_beats = {}      # (recording, camera) -> statistic -> [errors]
    recording_common = {}

    for section in sections(windows):
        section_unit = unit_id("section", recording_id=section.recording_id,
                               camera_id=section.camera_id,
                               section_index=section.index)
        participant = _participant(section)
        common = {**section.attrs,
                  "recording_id": section.recording_id,
                  "camera_id": section.camera_id,
                  "participant": participant}
        recording_key = (section.recording_id, section.camera_id)
        recording_common.setdefault(recording_key, common)

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
                signed_sections.setdefault(participant, {}).setdefault(
                    statistic, []).append(float(values.mean()))
                absolute_beats.setdefault(participant, {}).setdefault(
                    statistic, []).append(np.abs(values))
                recording_beats.setdefault(recording_key, {}).setdefault(
                    statistic, []).append(values)

        quality = detection_quality(section.prediction, section.label, fs)
        for metric, value in quality.items():
            rows.append(_row("section", section_unit, signal,
                             f"detection_{metric}", value, **common))

    for statistic in ("max", "mean", "min"):
        rows.extend(_recording_rows(recording_beats, recording_common, signal,
                                    statistic))
        rows.extend(_ieee1708_rows(absolute_beats, signal, statistic))
        rows.extend(_iso81060_3_rows(signed_sections, signal, statistic))
    return rows


def _recording_rows(recording_beats, recording_common, signal, statistic) -> list:
    """One signed bias per camera view — each view is its own measurement."""
    rows = []
    for key, stats in recording_beats.items():
        if not stats.get(statistic):
            continue
        pooled = np.concatenate(stats[statistic])
        recording_id, camera_id = key
        rows.append(_row("recording",
                         unit_id("recording", recording_id=recording_id,
                                 camera_id=camera_id),
                         signal, "bias", float(pooled.mean()),
                         statistic=statistic, n=pooled.size,
                         **recording_common[key]))
    return rows


def _ieee1708_rows(absolute_beats, signal, statistic) -> list:
    """Per-subject MAE over that subject's beats, graded, then pooled.

    The grade rides in its own ``grade`` column rather than in the metric name:
    a consumer filtering ``metric == "ieee1708_mae"`` must find every subject.
    """
    rows = []
    per_subject = {name: np.concatenate(stats[statistic])
                   for name, stats in absolute_beats.items()
                   if stats.get(statistic)}
    for participant, beats in per_subject.items():
        mae = float(beats.mean())
        rows.append(_row("participant",
                         unit_id("participant", participant=participant),
                         signal, "ieee1708_mae", mae, statistic=statistic,
                         n=beats.size, participant=participant,
                         grade=grade_ieee1708(mae)))
    if per_subject:
        pooled = float(np.mean([beats.mean() for beats in per_subject.values()]))
        rows.append(_row("cohort", "all", signal, "ieee1708_mae", pooled,
                         statistic=statistic, n=len(per_subject),
                         grade=grade_ieee1708(pooled)))
    return rows


def _iso81060_3_rows(signed_sections, signal, statistic) -> list:
    """Signed per-subject bias, and the pooled verdict over those biases."""
    rows = []
    subject_bias = {name: float(np.mean(stats[statistic]))
                    for name, stats in signed_sections.items()
                    if stats.get(statistic)}
    for participant, value in subject_bias.items():
        rows.append(_row("participant",
                         unit_id("participant", participant=participant),
                         signal, "bias", value, statistic=statistic,
                         participant=participant))
    if not subject_bias:
        return rows
    verdict = iso81060_3_verdict(np.array(list(subject_bias.values())))
    passes = verdict["passes"]
    rows.append(_row("cohort", "all", signal, "iso81060_3_passes",
                     float("nan") if passes is None else float(passes),
                     statistic=statistic, n=verdict["n_subjects"],
                     note=verdict["note"]))
    rows.append(_row("cohort", "all", signal, "mean_error",
                     verdict["mean_error"], verdict["se"],
                     statistic=statistic, n=verdict["n_subjects"]))
    rows.append(_row("cohort", "all", signal, "sd", verdict["sd"],
                     statistic=statistic, n=verdict["n_subjects"]))
    return rows


def digest(frame, run) -> str:
    """The fixed, readable summary. The CSV is where everything else lives."""
    lines = ["=== Evaluation report ==="]
    lines += [f"  {line}" for line in standards.provenance_lines()]
    for signal in (run.traces or run.signals()):
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
            lines += _clinical_digest(rows, signal, unit)
    return "\n".join(lines)


def _clinical_digest(rows, signal, unit) -> list:
    """Bias, IEEE 1708 grade and ISO verdict — each with what qualifies it."""
    labels = beat_labels(signal)
    cohort = rows[rows["level"] == "cohort"]
    lines = []
    for statistic in ("max", "mean", "min"):
        label = labels[statistic]
        bias = cohort[(cohort["metric"] == "mean_error") &
                      (cohort["statistic"] == statistic)]
        if not bias.empty:
            row = bias.iloc[0]
            lines.append(f"  {label}: bias {row['value']:+.2f} "
                         f"{unit} over {int(row['n'])} subjects")
        mae = cohort[(cohort["metric"] == "ieee1708_mae") &
                     (cohort["statistic"] == statistic)]
        if not mae.empty:
            row = mae.iloc[0]
            lines.append(f"  {label}: IEEE 1708 grade "
                         f"{row.get('grade', 'ungraded')} "
                         f"(MAE {row['value']:.2f} {unit} over "
                         f"{int(row['n'])} subjects)")
        verdict = cohort[(cohort["metric"] == "iso81060_3_passes") &
                         (cohort["statistic"] == statistic)]
        if not verdict.empty:
            row = verdict.iloc[0]
            state = ("not computable" if not np.isfinite(row["value"])
                     else ("PASS" if row["value"] else "FAIL"))
            note = row.get("note", "")
            caveat = f" — {note}" if isinstance(note, str) and note else ""
            lines.append(f"  {label}: ISO 81060-3 {state}{caveat}")
    return lines


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
