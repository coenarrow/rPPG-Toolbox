"""Clinical criteria: a known error distribution must earn a known grade."""
import numpy as np
import pytest

from evaluation.scoring.clinical import (
    beat_errors, grade_ieee1708, iso81060_3_verdict)


def test_grades_follow_the_bands():
    assert grade_ieee1708(4.9) == "A"
    assert grade_ieee1708(5.5) == "B"
    assert grade_ieee1708(6.5) == "C"
    assert grade_ieee1708(9.0) == "D"
    assert grade_ieee1708(float("nan")) == "ungraded"


def test_iso_verdict_passes_a_tight_distribution_and_fails_a_wide_one():
    tight = iso81060_3_verdict(np.full(20, 1.0))
    assert tight["passes"] and tight["mean_error"] == pytest.approx(1.0)

    wide = iso81060_3_verdict(np.linspace(-40.0, 40.0, 20))
    assert not wide["passes"]


def test_a_single_subject_is_declined_rather_than_graded():
    verdict = iso81060_3_verdict(np.array([1.0]))
    assert verdict["passes"] is None
    assert "n = 1" in verdict["note"]


def test_beat_errors_are_zero_against_the_reference_itself():
    t = np.arange(300) / 30.0
    trace = 100.0 + 20.0 * -np.cos(2 * np.pi * 1.2 * t)
    errors = beat_errors(trace, trace, 30.0)
    assert errors["max"].size >= 8
    assert np.allclose(errors["mean"], 0.0)
