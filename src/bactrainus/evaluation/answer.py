"""Official HotpotQA normalized answer metrics."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from ._alignment import validate_ids

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)
_SPECIAL_ANSWERS = frozenset({"yes", "no", "noanswer"})


def normalize_answer(value: object) -> str:
    """Lowercase and remove punctuation, articles, and extra whitespace."""

    if value is None:
        return ""
    text = str(value).lower().translate(_PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", text).split())


@dataclass(frozen=True, slots=True)
class AnswerScore:
    exact_match: float
    f1: float
    precision: float
    recall: float


@dataclass(frozen=True, slots=True)
class AnswerMetrics:
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


def score_answer(prediction: object, gold: object) -> AnswerScore:
    """Score one answer using the official HotpotQA normalization rules."""

    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    exact_match = float(normalized_prediction == normalized_gold)

    if (
        normalized_prediction in _SPECIAL_ANSWERS or normalized_gold in _SPECIAL_ANSWERS
    ) and normalized_prediction != normalized_gold:
        return AnswerScore(exact_match, 0.0, 0.0, 0.0)

    prediction_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    overlap = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return AnswerScore(exact_match, 0.0, 0.0, 0.0)
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2.0 * precision * recall / (precision + recall)
    return AnswerScore(exact_match, f1, precision, recall)


def evaluate_answers(
    gold: Mapping[str, object],
    predictions: Mapping[str, object],
    *,
    strict: bool = True,
) -> AnswerMetrics:
    """Aggregate answer metrics in gold-ID order.

    With ``strict=False``, missing predictions are scored as empty strings and
    extra prediction IDs are ignored, matching the official evaluator's
    treatment of missing answers.
    """

    identifiers = validate_ids(gold, (predictions,), strict=strict)
    totals = [0.0, 0.0, 0.0, 0.0]
    for identifier in identifiers:
        score = score_answer(predictions.get(identifier, ""), gold[identifier])
        totals[0] += score.exact_match
        totals[1] += score.f1
        totals[2] += score.precision
        totals[3] += score.recall
    count = len(identifiers)
    return AnswerMetrics(count, *(total / count for total in totals))
