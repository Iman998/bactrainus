"""Strict orchestration of the modular Bactrainus inference path."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .protocols import AnswerReader, ParagraphSelector, QuestionDecomposer, SentenceSelector
from .schemas import (
    EvidenceMode,
    HotpotExample,
    PipelinePrediction,
    ReaderEvidence,
    SupportingFact,
)


class PipelineValidationError(ValueError):
    """Raised when a component returns invalid or misaligned output."""


def _titles(example: HotpotExample, values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise PipelineValidationError("paragraph selector must return a sequence")
    titles = tuple(values)
    if not titles:
        raise PipelineValidationError("paragraph selector returned no paragraphs")
    if any(not isinstance(title, str) or not title.strip() for title in titles):
        raise PipelineValidationError("paragraph titles must be non-empty strings")
    if len(titles) != len(set(titles)):
        raise PipelineValidationError("paragraph selector returned duplicate titles")
    unknown = [title for title in titles if title not in example.paragraph_by_title]
    if unknown:
        raise PipelineValidationError(f"paragraph selector returned unknown titles: {unknown}")
    return titles


def _subquestions(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise PipelineValidationError("decomposer must return a sequence")
    if any(not isinstance(question, str) for question in values):
        raise PipelineValidationError("decomposer must return strings")
    questions = tuple(question.strip() for question in values)
    if not questions or any(not question for question in questions):
        raise PipelineValidationError("decomposer returned empty subquestions")
    if len(questions) != len(set(questions)):
        raise PipelineValidationError("decomposer returned duplicate subquestions")
    return questions


def _facts(
    example: HotpotExample,
    paragraph_titles: Sequence[str],
    values: Sequence[SupportingFact],
) -> tuple[SupportingFact, ...]:
    if isinstance(values, (str, bytes)):
        raise PipelineValidationError("sentence selector must return a sequence")
    facts = tuple(values)
    if not facts:
        raise PipelineValidationError("sentence selector returned no facts")
    if any(not isinstance(fact, SupportingFact) for fact in facts):
        raise PipelineValidationError("sentence selector must return SupportingFact objects")
    if len(facts) != len(set(facts)):
        raise PipelineValidationError("sentence selector returned duplicate facts")
    selected = set(paragraph_titles)
    for fact in facts:
        if fact.title not in selected:
            raise PipelineValidationError(
                f"sentence selector returned fact outside selected paragraphs: {fact}"
            )
        paragraph = example.paragraph(fact.title)
        if fact.sentence_index >= len(paragraph.sentences):
            raise PipelineValidationError(
                f"sentence selector returned an out-of-range index: {fact}"
            )
    return facts


@dataclass(slots=True)
class BactrainusPipeline:
    """Compose independently testable paragraph, sentence, and reader stages."""

    paragraph_selector: ParagraphSelector
    sentence_selector: SentenceSelector
    reader: AnswerReader
    decomposer: QuestionDecomposer | None = None
    reader_mode: EvidenceMode = EvidenceMode.FACTS

    def __post_init__(self) -> None:
        if not isinstance(self.reader_mode, EvidenceMode):
            self.reader_mode = EvidenceMode(self.reader_mode)

    def run(self, example: HotpotExample) -> PipelinePrediction:
        paragraph_titles = _titles(example, self.paragraph_selector.select(example))
        subquestions: tuple[str, ...] = ()
        if self.decomposer is not None:
            subquestions = _subquestions(self.decomposer.decompose(example, paragraph_titles))
        supporting_facts = _facts(
            example,
            paragraph_titles,
            self.sentence_selector.select(example, paragraph_titles, subquestions=subquestions),
        )
        evidence = ReaderEvidence(
            mode=self.reader_mode,
            paragraph_titles=paragraph_titles,
            supporting_facts=supporting_facts,
        )
        answer = self.reader.answer(example, evidence)
        if not isinstance(answer, str) or not answer.strip():
            raise PipelineValidationError("reader returned an empty answer")
        return PipelinePrediction(
            example_id=example.example_id,
            answer=answer.strip(),
            paragraph_titles=paragraph_titles,
            supporting_facts=supporting_facts,
            subquestions=subquestions,
        )

    def run_many(self, examples: Iterable[HotpotExample]) -> tuple[PipelinePrediction, ...]:
        """Run examples in input order and reject duplicate IDs."""

        predictions: list[PipelinePrediction] = []
        seen: set[str] = set()
        for example in examples:
            if example.example_id in seen:
                raise PipelineValidationError(f"duplicate input example ID {example.example_id!r}")
            seen.add(example.example_id)
            predictions.append(self.run(example))
        return tuple(predictions)
