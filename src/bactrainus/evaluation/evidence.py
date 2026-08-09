"""Official HotpotQA supporting-fact metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..schemas import SupportingFact
from ._alignment import validate_ids

FactInput = SupportingFact | tuple[str, int] | list[object]


def _fact(value: FactInput) -> SupportingFact:
    if isinstance(value, SupportingFact):
        return value
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError("supporting facts must be SupportingFact or [title, index] pairs")
    title, index = value
    if not isinstance(title, str) or isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("supporting facts require a string title and integer index")
    return SupportingFact(title, index)


def coerce_facts(values: Sequence[FactInput]) -> tuple[SupportingFact, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("supporting facts must be a sequence of pairs")
    return tuple(_fact(value) for value in values)


@dataclass(frozen=True, slots=True)
class EvidenceScore:
    exact_match: float
    f1: float
    precision: float
    recall: float


@dataclass(frozen=True, slots=True)
class EvidenceMetrics:
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


def score_evidence(prediction: Sequence[FactInput], gold: Sequence[FactInput]) -> EvidenceScore:
    """Score one set of predicted supporting facts."""

    predicted_set = set(coerce_facts(prediction))
    gold_set = set(coerce_facts(gold))
    true_positives = len(predicted_set & gold_set)
    false_positives = len(predicted_set - gold_set)
    false_negatives = len(gold_set - predicted_set)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact_match = float(false_positives + false_negatives == 0)
    return EvidenceScore(exact_match, f1, precision, recall)


def evaluate_evidence(
    gold: Mapping[str, Sequence[FactInput]],
    predictions: Mapping[str, Sequence[FactInput]],
    *,
    strict: bool = True,
) -> EvidenceMetrics:
    """Aggregate supporting-fact metrics in gold-ID order."""

    identifiers = validate_ids(gold, (predictions,), strict=strict)
    totals = [0.0, 0.0, 0.0, 0.0]
    for identifier in identifiers:
        score = score_evidence(predictions.get(identifier, ()), gold[identifier])
        totals[0] += score.exact_match
        totals[1] += score.f1
        totals[2] += score.precision
        totals[3] += score.recall
    count = len(identifiers)
    return EvidenceMetrics(count, *(total / count for total in totals))
