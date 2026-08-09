"""Official HotpotQA joint answer/supporting-fact metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ._alignment import validate_ids
from .answer import score_answer
from .evidence import FactInput, score_evidence


@dataclass(frozen=True, slots=True)
class JointScore:
    exact_match: float
    f1: float
    precision: float
    recall: float


@dataclass(frozen=True, slots=True)
class JointMetrics:
    count: int
    exact_match: float
    f1: float
    precision: float
    recall: float

    def as_percentages(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "exact_match": 100.0 * self.exact_match,
            "f1": 100.0 * self.f1,
            "precision": 100.0 * self.precision,
            "recall": 100.0 * self.recall,
        }


def score_joint(
    predicted_answer: object,
    gold_answer: object,
    predicted_facts: Sequence[FactInput],
    gold_facts: Sequence[FactInput],
) -> JointScore:
    """Score one answer/evidence pair using official joint composition."""

    answer = score_answer(predicted_answer, gold_answer)
    evidence = score_evidence(predicted_facts, gold_facts)
    precision = answer.precision * evidence.precision
    recall = answer.recall * evidence.recall
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact_match = answer.exact_match * evidence.exact_match
    return JointScore(exact_match, f1, precision, recall)


def evaluate_joint(
    gold_answers: Mapping[str, object],
    gold_facts: Mapping[str, Sequence[FactInput]],
    predicted_answers: Mapping[str, object],
    predicted_facts: Mapping[str, Sequence[FactInput]],
    *,
    strict: bool = True,
) -> JointMetrics:
    """Aggregate joint metrics with strict cross-artifact ID alignment."""

    answer_ids = tuple(gold_answers)
    fact_ids = tuple(gold_facts)
    if answer_ids != fact_ids:
        raise ValueError("gold answer and supporting-fact mappings must have identical order")
    identifiers = validate_ids(gold_answers, (predicted_answers, predicted_facts), strict=strict)
    totals = [0.0, 0.0, 0.0, 0.0]
    for identifier in identifiers:
        score = score_joint(
            predicted_answers.get(identifier, ""),
            gold_answers[identifier],
            predicted_facts.get(identifier, ()),
            gold_facts[identifier],
        )
        totals[0] += score.exact_match
        totals[1] += score.f1
        totals[2] += score.precision
        totals[3] += score.recall
    count = len(identifiers)
    return JointMetrics(count, *(total / count for total in totals))
