from __future__ import annotations

import math

import pytest

from bactrainus.evaluation import (
    calibrate_score,
    estimate_calibration_factor,
    evaluate_answers,
    evaluate_evidence,
    evaluate_joint,
    finite_population_correction,
    normalize_answer,
    score_answer,
    score_evidence,
    score_joint,
    worst_case_margin_of_error,
)
from bactrainus.evaluation._alignment import PredictionAlignmentError
from bactrainus.evaluation.calibration import CalibrationFactor
from bactrainus.schemas import SupportingFact


def test_official_answer_normalization_and_boolean_rule() -> None:
    assert normalize_answer("  The, City! ") == "city"
    assert score_answer("the Tehran", "Tehran").exact_match == 1.0
    assert score_answer("yes indeed", "yes").f1 == 0.0
    assert score_answer("yes", "yes").f1 == 1.0


def test_answer_aggregation_is_id_strict() -> None:
    metrics = evaluate_answers(
        {"a": "Tehran", "b": "yes"},
        {"a": "the Tehran", "b": "no"},
    )
    assert metrics.count == 2
    assert metrics.exact_match == 0.5
    assert metrics.f1 == 0.5

    with pytest.raises(PredictionAlignmentError, match="missing IDs"):
        evaluate_answers({"a": "x"}, {})
    assert evaluate_answers({"a": "x"}, {}, strict=False).exact_match == 0.0


def test_supporting_fact_and_joint_metrics() -> None:
    gold = [SupportingFact("A", 0), SupportingFact("B", 1)]
    predicted = [SupportingFact("A", 0), SupportingFact("C", 2)]
    evidence = score_evidence(predicted, gold)
    assert evidence.precision == 0.5
    assert evidence.recall == 0.5
    assert evidence.f1 == 0.5
    assert evidence.exact_match == 0.0

    answer = score_answer("alpha beta", "alpha gamma")
    joint = score_joint("alpha beta", "alpha gamma", predicted, gold)
    assert joint.precision == answer.precision * evidence.precision
    assert joint.recall == answer.recall * evidence.recall

    evidence_metrics = evaluate_evidence({"x": gold}, {"x": gold})
    assert evidence_metrics.exact_match == 1.0
    joint_metrics = evaluate_joint({"x": "yes"}, {"x": gold}, {"x": "yes"}, {"x": gold})
    assert joint_metrics.exact_match == 1.0
    assert joint_metrics.f1 == 1.0


def test_calibration_uses_full_mean_over_subset_mean() -> None:
    factor = estimate_calibration_factor(
        subset_scores={"m1": 50.0, "m2": 70.0},
        full_scores={"m1": 55.0, "m2": 65.0},
    )
    assert factor.value == 1.0
    assert calibrate_score(61.5, factor) == 61.5

    factor = estimate_calibration_factor(
        subset_scores={"m1": 40.0, "m2": 60.0},
        full_scores={"m1": 50.0, "m2": 70.0},
    )
    assert factor.value == pytest.approx(1.2)
    assert calibrate_score(50.0, factor) == pytest.approx(60.0)

    with pytest.raises(ValueError, match="IDs differ"):
        estimate_calibration_factor({"m1": 1.0}, {"m2": 1.0})


def test_finite_population_precision_reference() -> None:
    correction = finite_population_correction(7_405, 700)
    assert correction == pytest.approx(math.sqrt(6_705 / 7_404))
    margin = worst_case_margin_of_error(700, population_size=7_405)
    assert margin == pytest.approx(3.524806, abs=0.000001)


@pytest.mark.parametrize(
    ("subset", "full", "error"),
    [
        ({}, {}, "must not be empty"),
        ({" model": 1.0}, {" model": 1.0}, "model IDs"),
        ({"model": True}, {"model": 1.0}, "must be numeric"),
        ({"model": -1.0}, {"model": 1.0}, "non-negative"),
    ],
)
def test_calibration_rejects_invalid_reference_scores(
    subset: dict[str, float], full: dict[str, float], error: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        estimate_calibration_factor(subset, full)


def test_calibration_validation_and_unclipped_application() -> None:
    with pytest.raises(ZeroDivisionError):
        estimate_calibration_factor({"model": 0.0}, {"model": 1.0})
    with pytest.raises(TypeError, match="reference_count"):
        CalibrationFactor(1.0, True, 1.0, 1.0)
    with pytest.raises(ValueError, match="positive"):
        CalibrationFactor(1.0, 0, 1.0, 1.0)
    with pytest.raises(TypeError, match="numeric"):
        CalibrationFactor("x", 1, 1.0, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        CalibrationFactor(math.inf, 1, 1.0, 1.0)

    assert calibrate_score(80.0, 1.5) == 120.0
    with pytest.raises(TypeError, match="score"):
        calibrate_score(True, 1.0)
    with pytest.raises(TypeError, match="factor"):
        calibrate_score(1.0, True)
    with pytest.raises(ValueError, match="score"):
        calibrate_score(math.nan, 1.0)
    with pytest.raises(ValueError, match="factor"):
        calibrate_score(1.0, -1.0)


@pytest.mark.parametrize(
    ("population", "sample", "error"),
    [
        (True, 1, TypeError),
        (10.0, 1, TypeError),
        (1, 1, ValueError),
        (10, 0, ValueError),
        (10, 11, ValueError),
    ],
)
def test_finite_population_validation(
    population: object, sample: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        finite_population_correction(population, sample)  # type: ignore[arg-type]


def test_margin_of_error_validation_and_fraction_output() -> None:
    assert worst_case_margin_of_error(100, percentage_points=False) == pytest.approx(
        1.959963984540054 * 0.05
    )
    with pytest.raises(TypeError, match="sample_size"):
        worst_case_margin_of_error(True)
    with pytest.raises(ValueError, match="positive"):
        worst_case_margin_of_error(0)
    with pytest.raises(TypeError, match="z_score"):
        worst_case_margin_of_error(10, z_score=True)
    with pytest.raises(ValueError, match="finite"):
        worst_case_margin_of_error(10, z_score=math.inf)
