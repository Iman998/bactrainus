"""Strict prediction/gold ID alignment shared by evaluators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


class PredictionAlignmentError(ValueError):
    """Raised when prediction IDs do not match gold IDs."""


def validate_ids(
    gold: Mapping[str, object],
    predictions: Sequence[Mapping[str, object]],
    *,
    strict: bool,
) -> tuple[str, ...]:
    if not gold:
        raise PredictionAlignmentError("gold mapping must not be empty")
    gold_ids = tuple(gold)
    if any(not isinstance(identifier, str) or not identifier for identifier in gold_ids):
        raise PredictionAlignmentError("gold IDs must be non-empty strings")
    if strict:
        expected = set(gold_ids)
        for position, prediction in enumerate(predictions):
            missing = [identifier for identifier in gold_ids if identifier not in prediction]
            unexpected = [identifier for identifier in prediction if identifier not in expected]
            if missing or unexpected:
                raise PredictionAlignmentError(
                    f"prediction mapping {position} has missing IDs {missing} "
                    f"and unexpected IDs {unexpected}"
                )
    return gold_ids
