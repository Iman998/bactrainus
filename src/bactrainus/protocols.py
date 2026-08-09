"""Narrow interfaces used for dependency injection in Bactrainus."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .schemas import (
    GenerationRequest,
    GenerationResult,
    HotpotExample,
    ReaderEvidence,
    SupportingFact,
)


class TextGenerator(Protocol):
    """Generate text for an ordered batch of backend-neutral requests."""

    def generate(self, requests: Sequence[GenerationRequest]) -> Sequence[GenerationResult]: ...


class ParagraphSelector(Protocol):
    """Select relevant paragraph titles from an example's candidate set."""

    def select(self, example: HotpotExample) -> Sequence[str]: ...


class QuestionDecomposer(Protocol):
    """Create subquestions after paragraph selection."""

    def decompose(
        self, example: HotpotExample, paragraph_titles: Sequence[str]
    ) -> Sequence[str]: ...


class SentenceSelector(Protocol):
    """Select sentence-level facts from already selected paragraphs."""

    def select(
        self,
        example: HotpotExample,
        paragraph_titles: Sequence[str],
        subquestions: Sequence[str] = (),
    ) -> Sequence[SupportingFact]: ...


class AnswerReader(Protocol):
    """Answer a question from an explicit evidence bundle."""

    def answer(self, example: HotpotExample, evidence: ReaderEvidence) -> str: ...
