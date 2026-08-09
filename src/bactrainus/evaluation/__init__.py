"""Official HotpotQA answer, evidence, joint, and calibration utilities."""

from .answer import AnswerMetrics, AnswerScore, evaluate_answers, normalize_answer, score_answer
from .calibration import (
    CalibrationFactor,
    calibrate_score,
    estimate_calibration_factor,
    finite_population_correction,
    worst_case_margin_of_error,
)
from .evidence import EvidenceMetrics, EvidenceScore, evaluate_evidence, score_evidence
from .joint import JointMetrics, JointScore, evaluate_joint, score_joint

__all__ = [
    "AnswerMetrics",
    "AnswerScore",
    "CalibrationFactor",
    "EvidenceMetrics",
    "EvidenceScore",
    "JointMetrics",
    "JointScore",
    "calibrate_score",
    "estimate_calibration_factor",
    "evaluate_answers",
    "evaluate_evidence",
    "evaluate_joint",
    "finite_population_correction",
    "normalize_answer",
    "score_answer",
    "score_evidence",
    "score_joint",
    "worst_case_margin_of_error",
]
