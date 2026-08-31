"""Per-beat pressure agreement, and the acceptance criteria over it.

Beats are reference-anchored (``evaluation.beats``), so each reference beat
contributes exactly one comparison of each statistic and the error
distribution is not filtered by how detectable the prediction's own beats are.
"""

import numpy as np

from evaluation.beats import BEAT_STATS, beat_intervals, beat_stats
from evaluation.scoring import standards
from evaluation.uncertainty import mean_se


def beat_errors(prediction, label, fs) -> dict:
    """``prediction - label`` per beat, for each of ``BEAT_STATS``.

    The intervals come from the label, and both traces are read inside them.
    """
    intervals = beat_intervals(label, fs)
    if not intervals:
        return {name: np.array([], dtype=np.float64) for name in BEAT_STATS}
    predicted = beat_stats(prediction, intervals)
    reference = beat_stats(label, intervals)
    return {name: predicted[name] - reference[name] for name in BEAT_STATS}


def grade_ieee1708(mae) -> str:
    """The A-D band a mean absolute error falls in."""
    if mae is None or not np.isfinite(mae):
        return "ungraded"
    for limit, grade in standards.IEEE_1708_GRADE_BANDS:
        if mae <= limit:
            return grade
    return standards.IEEE_1708_FALLBACK_GRADE


def iso81060_3_verdict(per_subject_errors) -> dict:
    """Pooled mean error and SD against the acceptance limits.

    ``per_subject_errors`` is one mean error per subject, so the samples are
    independent and the standard error is naive — the standard prescribes this
    aggregation, and layering a HAC estimate on top would depart from it.

    ``passes`` is ``None``, never ``False``, when there are too few subjects
    for the verdict to mean anything — which is every LOSO fold.
    """
    errors = np.asarray(per_subject_errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    n_subjects = errors.size
    mean_error, sd, se = mean_se(errors, method="naive")
    result = {
        "n_subjects": int(n_subjects),
        "mean_error": float(mean_error),
        "sd": float(sd),
        "se": float(se),
        "mean_error_limit": standards.ISO_81060_3_MEAN_ERROR_LIMIT,
        "sd_limit": standards.ISO_81060_3_SD_LIMIT,
        "source": standards.ISO_81060_3_SOURCE,
        "note": "",
    }
    if n_subjects < standards.MIN_SUBJECTS_FOR_VERDICT:
        result["passes"] = None
        result["note"] = (f"not computable at n = {n_subjects}: the pooled SD "
                          f"needs at least "
                          f"{standards.MIN_SUBJECTS_FOR_VERDICT} subjects")
        return result
    result["passes"] = bool(
        abs(mean_error) <= standards.ISO_81060_3_MEAN_ERROR_LIMIT
        and sd <= standards.ISO_81060_3_SD_LIMIT)
    return result
