"""Reference-model calibration for fixed-subset API screening."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CalibrationFactor:
    """Multiplicative full-set/subset correction estimated on references."""

    value: float
    reference_count: int
    subset_mean: float
    full_mean: float

    def __post_init__(self) -> None:
        if isinstance(self.reference_count, bool) or not isinstance(self.reference_count, int):
            raise TypeError("reference_count must be an integer")
        if self.reference_count <= 0:
            raise ValueError("reference_count must be positive")
        for field_name in ("value", "subset_mean", "full_mean"):
            raw_value = getattr(self, field_name)
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise TypeError(f"{field_name} must be numeric")
            if not math.isfinite(raw_value):
                raise ValueError(f"{field_name} must be finite")


def _validated_scores(scores: Mapping[str, float], label: str) -> dict[str, float]:
    if not scores:
        raise ValueError(f"{label} scores must not be empty")
    validated: dict[str, float] = {}
    for model_id, raw_score in scores.items():
        if not isinstance(model_id, str) or not model_id or model_id != model_id.strip():
            raise TypeError(f"{label} model IDs must be non-empty strings")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise TypeError(f"{label} score for {model_id!r} must be numeric")
        score = float(raw_score)
        if not math.isfinite(score) or score < 0:
            raise ValueError(f"{label} score for {model_id!r} must be finite and non-negative")
        validated[model_id] = score
    return validated


def estimate_calibration_factor(
    subset_scores: Mapping[str, float], full_scores: Mapping[str, float]
) -> CalibrationFactor:
    """Estimate ``mean(full reference) / mean(subset reference)``.

    Model IDs must match exactly.  Means are computed over models, so every
    reference checkpoint receives equal weight.
    """

    subset = _validated_scores(subset_scores, "subset")
    full = _validated_scores(full_scores, "full")
    if set(subset) != set(full):
        missing = sorted(set(full) - set(subset))
        unexpected = sorted(set(subset) - set(full))
        raise ValueError(
            f"reference model IDs differ; missing subset IDs={missing}, "
            f"unexpected subset IDs={unexpected}"
        )
    count = len(subset)
    subset_mean = math.fsum(subset[model_id] for model_id in full) / count
    full_mean = math.fsum(full.values()) / count
    if subset_mean == 0:
        raise ZeroDivisionError("mean reference subset score is zero")
    return CalibrationFactor(
        value=full_mean / subset_mean,
        reference_count=count,
        subset_mean=subset_mean,
        full_mean=full_mean,
    )


def calibrate_score(score: float, factor: CalibrationFactor | float) -> float:
    """Apply a multiplicative calibration factor without hidden clipping."""

    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError("score must be numeric")
    if not isinstance(factor, CalibrationFactor) and (
        isinstance(factor, bool) or not isinstance(factor, (int, float))
    ):
        raise TypeError("factor must be numeric or a CalibrationFactor")
    value = float(score)
    multiplier = factor.value if isinstance(factor, CalibrationFactor) else float(factor)
    if not math.isfinite(value) or value < 0:
        raise ValueError("score must be finite and non-negative")
    if not math.isfinite(multiplier) or multiplier < 0:
        raise ValueError("factor must be finite and non-negative")
    return value * multiplier


def finite_population_correction(population_size: int, sample_size: int) -> float:
    """Return ``sqrt((N-n)/(N-1))`` for sampling without replacement."""

    if isinstance(population_size, bool) or isinstance(sample_size, bool):
        raise TypeError("population_size and sample_size must be integers")
    if not isinstance(population_size, int) or not isinstance(sample_size, int):
        raise TypeError("population_size and sample_size must be integers")
    if population_size <= 1:
        raise ValueError("population_size must exceed one")
    if sample_size <= 0 or sample_size > population_size:
        raise ValueError("sample_size must be between one and population_size")
    return math.sqrt((population_size - sample_size) / (population_size - 1))


def worst_case_margin_of_error(
    sample_size: int,
    *,
    population_size: int | None = None,
    z_score: float = 1.959963984540054,
    percentage_points: bool = True,
) -> float:
    """Return the worst-case binomial margin of error.

    This is a design-based reference for random sampling.  A finite-population
    correction is applied when ``population_size`` is provided.
    """

    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError("sample_size must be an integer")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if isinstance(z_score, bool) or not isinstance(z_score, (int, float)):
        raise TypeError("z_score must be numeric")
    if not math.isfinite(z_score) or z_score <= 0:
        raise ValueError("z_score must be finite and positive")
    margin = z_score * math.sqrt(0.25 / sample_size)
    if population_size is not None:
        margin *= finite_population_correction(population_size, sample_size)
    return 100.0 * margin if percentage_points else margin
