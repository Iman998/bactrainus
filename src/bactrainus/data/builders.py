"""Deterministic builders for Bactrainus dataset views."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, TypeVar

from ..prompts.catalog import PromptCatalog
from ..prompts.renderer import (
    decomposer_message,
    question_context_message,
    selector_message,
    sentence_selector_message,
)
from ..schemas import (
    ChatMessage,
    HotpotExample,
    MessageRole,
    StructuredRecord,
    SupportingFact,
    TaskKind,
    TrainingRecord,
)

RecordT = TypeVar("RecordT", StructuredRecord, TrainingRecord, covariant=True)
_PROMPTS = PromptCatalog()
_RELEASE_QUESTION_TYPES = frozenset({"bridge", "comparison"})
_RELEASE_DIFFICULTIES = frozenset({"easy", "medium", "hard"})
_RELEASE_MIN_PARAGRAPHS = 2
_RELEASE_MAX_PARAGRAPHS = 10


class ExampleViewBuilder(Protocol[RecordT]):
    """Build one deterministic view from a canonical example."""

    def build(self, example: HotpotExample) -> RecordT: ...


def validate_release_example(example: HotpotExample) -> None:
    """Validate invariants shared by every public train configuration."""

    if not _RELEASE_MIN_PARAGRAPHS <= len(example.context) <= _RELEASE_MAX_PARAGRAPHS:
        raise ValueError(
            "release examples require between "
            f"{_RELEASE_MIN_PARAGRAPHS} and {_RELEASE_MAX_PARAGRAPHS} candidate "
            f"paragraphs; received {len(example.context)}"
        )
    if example.question_type not in _RELEASE_QUESTION_TYPES:
        raise ValueError(
            "release question_type must be 'bridge' or 'comparison'; "
            f"received {example.question_type!r}"
        )
    if example.level not in _RELEASE_DIFFICULTIES:
        raise ValueError(
            f"release difficulty must be 'easy', 'medium', or 'hard'; received {example.level!r}"
        )
    if example.split not in {None, "train"}:
        raise ValueError(f"public task views are train-only; received split {example.split!r}")


def _paragraphs_for_titles(example: HotpotExample, titles: Sequence[str] | None) -> tuple[str, ...]:
    if titles is None:
        return tuple(paragraph.title for paragraph in example.context)
    if isinstance(titles, (str, bytes)):
        raise TypeError("paragraph titles must be a sequence of strings")
    resolved = tuple(titles)
    if any(not isinstance(title, str) or not title.strip() for title in resolved):
        raise TypeError("paragraph titles must be non-empty strings")
    if len(resolved) != len(set(resolved)):
        raise ValueError("paragraph title selection contains duplicates")
    unknown = [title for title in resolved if title not in example.paragraph_by_title]
    if unknown:
        raise ValueError(f"unknown paragraph title(s): {unknown}")
    return resolved


def serialize_context(example: HotpotExample, paragraph_titles: Sequence[str] | None = None) -> str:
    """Serialize complete paragraphs with stable paragraph/sentence indices."""

    titles = _paragraphs_for_titles(example, paragraph_titles)
    blocks: list[str] = []
    for paragraph_position, title in enumerate(titles):
        paragraph = example.paragraph(title)
        lines = [f"Paragraph {paragraph_position} ({paragraph.title}):"]
        lines.extend(
            f"Sentence {sentence_position}: {sentence}"
            for sentence_position, sentence in enumerate(paragraph.sentences)
        )
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def serialize_facts(example: HotpotExample, facts: Sequence[SupportingFact] | None = None) -> str:
    """Serialize sentence-level evidence in fact order."""

    resolved = tuple(example.supporting_facts if facts is None else facts)
    blocks: list[str] = []
    for fact in resolved:
        if not isinstance(fact, SupportingFact):
            raise TypeError("facts must contain SupportingFact objects")
        paragraph = example.paragraph(fact.title)
        if fact.sentence_index >= len(paragraph.sentences):
            raise ValueError(f"sentence {fact.sentence_index} is out of range for {fact.title!r}")
        blocks.append(
            f"Paragraph ({fact.title}):\n"
            f"Sentence {fact.sentence_index}: {paragraph.sentences[fact.sentence_index]}"
        )
    return "\n".join(blocks)


def format_paragraph_target(titles: Sequence[str]) -> str:
    if isinstance(titles, (str, bytes)):
        raise TypeError("paragraph titles must be a sequence of strings")
    resolved = tuple(titles)
    if not resolved:
        raise ValueError("paragraph target must not be empty")
    if any(not isinstance(title, str) or not title.strip() for title in resolved):
        raise TypeError("paragraph titles must be non-empty strings")
    if len(resolved) != len(set(resolved)):
        raise ValueError("paragraph target contains duplicate titles")
    return "selected paragraphs:\n" + "\n".join(f"paragraph ***{title}***" for title in resolved)


def format_supporting_fact_target(facts: Sequence[SupportingFact]) -> str:
    if isinstance(facts, (str, bytes)):
        raise TypeError("supporting facts must be a sequence")
    resolved = tuple(facts)
    if not resolved:
        raise ValueError("supporting-fact target must not be empty")
    if any(not isinstance(fact, SupportingFact) for fact in resolved):
        raise TypeError("supporting facts must contain SupportingFact objects")
    if len(resolved) != len(set(resolved)):
        raise ValueError("supporting-fact target contains duplicates")
    return "supporting facts:\n" + "\n".join(
        f"paragraph ***{fact.title}***\nsentence ***{fact.sentence_index}***" for fact in resolved
    )


def format_answer_target(answer: str) -> str:
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer target must not be empty")
    return f"answer: ***{answer.strip()}***"


def build_gold_subquestions(example: HotpotExample) -> tuple[str, ...]:
    """Create an ordered, evidence-grounded decomposition for selected paragraphs.

    The paragraph-selection stage supplies the titles used here, so the target
    does not expose information unavailable to the decomposer at inference time.
    The transformation is deterministic and never depends on row position or a
    teacher API response.
    """

    titles = example.gold_paragraph_titles
    if not titles:
        raise ValueError("question decomposition requires at least one gold paragraph")
    questions: list[str] = []
    for index, title in enumerate(titles):
        if index == 0:
            questions.append(
                f'What information in "{title}" is needed to answer the original question?'
            )
            continue
        prior_titles = ", ".join(f'"{value}"' for value in titles[:index])
        questions.append(
            f'How does the relevant information in "{title}" combine with the evidence '
            f"from {prior_titles} to determine the answer?"
        )
    return tuple(questions)


def format_decomposition_target(subquestions: Sequence[str]) -> str:
    """Serialize an ordered sub-question target."""

    if isinstance(subquestions, (str, bytes)):
        raise TypeError("subquestions must be a sequence of strings")
    resolved = tuple(subquestions)
    if not resolved:
        raise ValueError("decomposition target must not be empty")
    if any(not isinstance(item, str) or not item.strip() for item in resolved):
        raise TypeError("subquestions must be non-empty strings")
    if len(resolved) != len(set(resolved)):
        raise ValueError("decomposition target contains duplicates")
    return "sub-questions:\n" + "\n".join(
        f"{index}. {question.strip()}" for index, question in enumerate(resolved, start=1)
    )


def format_rationale_target(example: HotpotExample) -> str:
    """Serialize a faithful evidence trace followed by the reference answer."""

    steps = []
    for index, fact in enumerate(example.supporting_facts, start=1):
        sentence = example.paragraph(fact.title).sentences[fact.sentence_index].strip()
        steps.append(f"{index}. [{fact.title}, sentence {fact.sentence_index}] {sentence}")
    return "rationale:\n" + "\n".join(steps) + "\n" + format_answer_target(example.answer)


def format_joint_target(example: HotpotExample) -> str:
    return (
        f"{format_supporting_fact_target(example.supporting_facts)}\n"
        f"{format_answer_target(example.answer)}"
    )


def _chat_record(
    example: HotpotExample,
    task: TaskKind,
    system_prompt: str,
    user_prompt: str,
    target: str,
) -> TrainingRecord:
    return TrainingRecord(
        example_id=example.example_id,
        task=task,
        messages=(
            ChatMessage(MessageRole.SYSTEM, system_prompt),
            ChatMessage(MessageRole.USER, user_prompt),
            ChatMessage(MessageRole.ASSISTANT, target),
        ),
    )


class StructuredViewBuilder:
    """Preserve the canonical structured example without prompt flattening."""

    def build(self, example: HotpotExample) -> StructuredRecord:
        validate_release_example(example)
        return StructuredRecord(example)


class ReaderViewBuilder:
    """Build answer-generation supervision from gold supporting facts."""

    def build(self, example: HotpotExample) -> TrainingRecord:
        validate_release_example(example)
        return _chat_record(
            example,
            TaskKind.READER,
            _PROMPTS.direct_reader,
            question_context_message(example.question, serialize_facts(example)),
            format_answer_target(example.answer),
        )


class CotReaderViewBuilder:
    """Build complete gold-evidence rationale and answer supervision."""

    def build(self, example: HotpotExample) -> TrainingRecord:
        validate_release_example(example)
        return _chat_record(
            example,
            TaskKind.COT_READER,
            _PROMPTS.rationale_teacher,
            question_context_message(example.question, serialize_facts(example)),
            format_rationale_target(example),
        )


class ParagraphSelectorViewBuilder:
    """Build paragraph-selection supervision from all candidate paragraphs."""

    def build(self, example: HotpotExample) -> TrainingRecord:
        validate_release_example(example)
        return _chat_record(
            example,
            TaskKind.PARAGRAPH_SELECTOR,
            _PROMPTS.paragraph_selector,
            selector_message(example.question, serialize_context(example)),
            format_paragraph_target(example.gold_paragraph_titles),
        )


class QuestionDecomposerViewBuilder:
    """Build complete decomposer supervision after paragraph selection."""

    def build(self, example: HotpotExample) -> TrainingRecord:
        validate_release_example(example)
        subquestions = build_gold_subquestions(example)
        return _chat_record(
            example,
            TaskKind.QUESTION_DECOMPOSER,
            _PROMPTS.question_decomposer,
            decomposer_message(
                example.question,
                serialize_context(example, example.gold_paragraph_titles),
            ),
            format_decomposition_target(subquestions),
        )


class SentenceSelectorViewBuilder:
    """Build supporting-sentence supervision within gold paragraphs."""

    def build(self, example: HotpotExample) -> TrainingRecord:
        validate_release_example(example)
        return _chat_record(
            example,
            TaskKind.SENTENCE_SELECTOR,
            _PROMPTS.sentence_selector,
            sentence_selector_message(
                example.question,
                serialize_context(example, example.gold_paragraph_titles),
            ),
            format_supporting_fact_target(example.supporting_facts),
        )


class DecomposedSentenceSelectorViewBuilder:
    """Build supporting-sentence supervision with an explicit decomposition."""

    def build(self, example: HotpotExample) -> TrainingRecord:
        validate_release_example(example)
        return _chat_record(
            example,
            TaskKind.DECOMPOSED_SENTENCE_SELECTOR,
            _PROMPTS.sentence_selector,
            sentence_selector_message(
                example.question,
                serialize_context(example, example.gold_paragraph_titles),
                subquestions=build_gold_subquestions(example),
            ),
            format_supporting_fact_target(example.supporting_facts),
        )


class JointViewBuilder:
    """Build joint supporting-fact and answer supervision."""

    def build(self, example: HotpotExample) -> TrainingRecord:
        validate_release_example(example)
        return _chat_record(
            example,
            TaskKind.JOINT,
            _PROMPTS.all_in_one,
            selector_message(example.question, serialize_context(example)),
            format_joint_target(example),
        )


def build_many(
    builder: ExampleViewBuilder[RecordT], examples: Iterable[HotpotExample]
) -> tuple[RecordT, ...]:
    """Build an ordered view and reject duplicate or mismatched IDs."""

    records: list[RecordT] = []
    seen: set[str] = set()
    for example in examples:
        if example.example_id in seen:
            raise ValueError(f"duplicate example ID {example.example_id!r}")
        seen.add(example.example_id)
        record = builder.build(example)
        if record.example_id != example.example_id:
            raise ValueError(
                f"builder changed example ID {example.example_id!r} to {record.example_id!r}"
            )
        records.append(record)
    return tuple(records)
