"""Clinical acceptance criteria, expressed as data.

Every threshold is a named constant with a source, and every report prints
where its numbers came from. Adding ESH 2023 or ISO 81060-2 later is rows in
``CRITERIA``, not new code.
"""

from dataclasses import dataclass

#: Set to True only once each constant below has been checked line by line
#: against the purchased standard. Until then every report says so out loud.
VERIFIED_AGAINST_STANDARD_TEXT = False

# --- IEEE 1708 ------------------------------------------------------------
#: Mean absolute error in mmHg, and the grade it earns. Anything above the
#: last band is grade D.
#: Source: IEEE 1708-2014, amended by 1708a-2019 — UNVERIFIED.
IEEE_1708_GRADE_BANDS = ((5.0, "A"), (6.0, "B"), (7.0, "C"))
IEEE_1708_FALLBACK_GRADE = "D"
IEEE_1708_SOURCE = "IEEE 1708-2014 / 1708a-2019 (UNVERIFIED)"

# --- ISO 81060-3 ----------------------------------------------------------
#: Continuous non-invasive BP against an invasive reference. Acceptance is on
#: the mean error and its standard deviation, pooled across subjects.
#: Source: ISO 81060-3:2022 — UNVERIFIED.
ISO_81060_3_MEAN_ERROR_LIMIT = 5.0     # mmHg
ISO_81060_3_SD_LIMIT = 8.0             # mmHg
ISO_81060_3_SOURCE = "ISO 81060-3:2022 (UNVERIFIED)"

#: Minimum subjects before a pooled verdict means anything. A LOSO fold has
#: one, so the report declines rather than grading.
MIN_SUBJECTS_FOR_VERDICT = 2

PROVENANCE = [
    f"IEEE 1708 grade bands from {IEEE_1708_SOURCE}",
    f"ISO 81060-3 limits from {ISO_81060_3_SOURCE}",
]

#: Requirements a metrics report can note but never satisfy: they constrain the
#: study, not the arithmetic. Printed with the numbers we do have, so nobody
#: reads a printed grade as a validation result.
STUDY_DESIGN_REQUIREMENTS = [
    "subject count and recruitment (a research cohort is not a validation study)",
    "prescribed distribution of reference pressures across the cohort",
    "reference-device protocol and its calibration record",
    "cuff procedure and observer training",
    "arm-circumference distribution",
]


@dataclass(frozen=True)
class Criterion:
    """One computable acceptance test."""

    name: str
    level: str          # the hierarchy level it consumes
    statistic: str      # which of BEAT_STATS it applies to
    source: str


CRITERIA = tuple(
    Criterion(name=name, level="participant", statistic=statistic, source=source)
    for name, source in (("ieee1708", IEEE_1708_SOURCE),
                         ("iso81060_3", ISO_81060_3_SOURCE))
    for statistic in ("max", "mean", "min")
)


def provenance_lines() -> list:
    """What a report prints above its clinical numbers."""
    lines = list(PROVENANCE)
    if not VERIFIED_AGAINST_STANDARD_TEXT:
        lines.append(
            "WARNING: thresholds have NOT been verified against the standards' "
            "text; treat grades as indicative only")
        lines.append(
            "WARNING: the aggregation formulas are equally UNVERIFIED against "
            "the standards' text — in particular the ISO 81060-3 SD here is a "
            "BETWEEN-SUBJECT SD of per-subject biases, which is far smaller "
            "than the SD of paired beat differences an invasive-reference "
            "standard tests, so the pass verdict is biased toward passing")
    lines.append("Not satisfiable by this dataset: "
                 + "; ".join(STUDY_DESIGN_REQUIREMENTS))
    return lines
