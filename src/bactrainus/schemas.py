"""Typed domain objects shared by Bactrainus modules.

The dataclasses in this module deliberately contain no framework-specific
types.  They form a small, stable boundary between data preparation,
generation backends, pipeline components, and evaluation code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


def _clean_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _source_text(value: str, field_name: str) -> str:
    """Validate source text without rewriting it."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _source_identifier(value: str, field_name: str) -> str:
    """Validate a canonical identifier without silently trimming it."""

    resolved = _source_text(value, field_name)
    if resolved != resolved.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    return resolved


class TaskKind(str, Enum):
    """Supported supervised views of HotpotQA."""

    STRUCTURED = "structured"
    READER = "reader"
    PARAGRAPH_SELECTOR = "paragraph_selector"
    SENTENCE_SELECTOR = "sentence_selector"
    JOINT = "joint_selector_reader"


class MessageRole(str, Enum):
    """Roles permitted by the public chat-training schema."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class EvidenceMode(str, Enum):
    """Evidence representation supplied to the final reader."""

    FACTS = "facts"
    PARAGRAPHS = "paragraphs"


@dataclass(frozen=True, slots=True, order=True)
class SupportingFact:
    """A sentence-level evidence reference."""

    title: str
    sentence_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", _source_identifier(self.title, "title"))
        if isinstance(self.sentence_index, bool) or not isinstance(self.sentence_index, int):
            raise TypeError("sentence_index must be an integer")
        if self.sentence_index < 0:
            raise ValueError("sentence_index must be non-negative")

    def to_list(self) -> list[object]:
        return [self.title, self.sentence_index]

    def to_dict(self) -> dict[str, object]:
        """Serialize to the public Hugging Face evidence struct."""

        return {"title": self.title, "sentence_index": self.sentence_index}


@dataclass(frozen=True, slots=True)
class Paragraph:
    """A titled paragraph represented as its ordered sentences."""

    title: str
    sentences: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", _source_identifier(self.title, "title"))
        if isinstance(self.sentences, str):
            raise TypeError("sentences must be a sequence of strings")
        sentences = tuple(self.sentences)
        if not sentences:
            raise ValueError("a paragraph must contain at least one sentence")
        for sentence in sentences:
            if not isinstance(sentence, str):
                raise TypeError("every sentence must be a string")
        object.__setattr__(self, "sentences", sentences)

    def to_list(self) -> list[object]:
        return [self.title, list(self.sentences)]

    def to_dict(self) -> dict[str, object]:
        """Serialize to the public Hugging Face paragraph struct."""

        return {"title": self.title, "sentences": list(self.sentences)}


@dataclass(frozen=True, slots=True)
class HotpotExample:
    """Canonical, validated HotpotQA example.

    Supporting facts are validated against the supplied context at creation
    time.  This prevents silently producing corrupted supervision when a
    title or sentence index is invalid.
    """

    example_id: str
    question: str
    answer: str
    context: tuple[Paragraph, ...]
    supporting_facts: tuple[SupportingFact, ...]
    split: str | None = None
    question_type: str | None = None
    level: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "example_id", _source_identifier(self.example_id, "example_id"))
        object.__setattr__(self, "question", _source_text(self.question, "question"))
        object.__setattr__(self, "answer", _source_text(self.answer, "answer"))

        context = tuple(self.context)
        facts = tuple(self.supporting_facts)
        if not context:
            raise ValueError("context must contain at least one paragraph")
        if not facts:
            raise ValueError("supporting_facts must not be empty")

        titles = [paragraph.title for paragraph in context]
        if len(titles) != len(set(titles)):
            raise ValueError("paragraph titles must be unique within an example")
        paragraph_by_title = {paragraph.title: paragraph for paragraph in context}

        if len(facts) != len(set(facts)):
            raise ValueError("supporting_facts must not contain duplicates")
        for fact in facts:
            paragraph = paragraph_by_title.get(fact.title)
            if paragraph is None:
                raise ValueError(f"supporting fact references unknown paragraph {fact.title!r}")
            if fact.sentence_index >= len(paragraph.sentences):
                raise ValueError(
                    "supporting fact index is out of range for paragraph "
                    f"{fact.title!r}: {fact.sentence_index}"
                )

        object.__setattr__(self, "context", context)
        object.__setattr__(self, "supporting_facts", facts)
        for field_name in ("split", "question_type", "level"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _source_identifier(value, field_name))

    @property
    def paragraph_by_title(self) -> dict[str, Paragraph]:
        return {paragraph.title: paragraph for paragraph in self.context}

    @property
    def gold_paragraph_titles(self) -> tuple[str, ...]:
        """Return gold titles in first-supporting-fact order."""

        seen: set[str] = set()
        titles: list[str] = []
        for fact in self.supporting_facts:
            if fact.title not in seen:
                seen.add(fact.title)
                titles.append(fact.title)
        return tuple(titles)

    def paragraph(self, title: str) -> Paragraph:
        try:
            return self.paragraph_by_title[title]
        except KeyError as error:
            raise KeyError(f"example {self.example_id!r} has no paragraph {title!r}") from error

    def to_dict(self) -> dict[str, Any]:
        """Serialize the exact canonical Hugging Face dataset schema.

        Release metadata is required here even though it remains optional on
        the domain object so inference-only examples can omit it.
        """

        if self.question_type is None:
            raise ValueError("question_type is required for structured serialization")
        if self.level is None:
            raise ValueError("difficulty is required for structured serialization")
        return {
            "source_id": self.example_id,
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type,
            "difficulty": self.level,
            "candidate_paragraphs": [paragraph.to_dict() for paragraph in self.context],
            "supporting_facts": [fact.to_dict() for fact in self.supporting_facts],
            "gold_paragraph_titles": list(self.gold_paragraph_titles),
        }


@dataclass(frozen=True, slots=True)
class StructuredRecord:
    """Serializable canonical dataset view."""

    example: HotpotExample
    task: TaskKind = TaskKind.STRUCTURED

    def __post_init__(self) -> None:
        if not isinstance(self.example, HotpotExample):
            raise TypeError("example must be a HotpotExample")
        if not isinstance(self.task, TaskKind):
            try:
                object.__setattr__(self, "task", TaskKind(self.task))
            except (TypeError, ValueError) as error:
                raise ValueError(f"unsupported task kind: {self.task!r}") from error

    @property
    def example_id(self) -> str:
        return self.example.example_id

    def to_dict(self) -> dict[str, Any]:
        return self.example.to_dict()


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One framework-neutral chat turn in an SFT record."""

    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            try:
                object.__setattr__(self, "role", MessageRole(self.role))
            except (TypeError, ValueError) as error:
                raise ValueError(f"unsupported message role: {self.role!r}") from error
        object.__setattr__(self, "content", _clean_text(self.content, "message content"))

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    """Public chat-formatted supervised-training view."""

    example_id: str
    task: TaskKind
    messages: tuple[ChatMessage, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.task, TaskKind):
            try:
                object.__setattr__(self, "task", TaskKind(self.task))
            except (TypeError, ValueError) as error:
                raise ValueError(f"unsupported task kind: {self.task!r}") from error
        object.__setattr__(self, "example_id", _clean_text(self.example_id, "example_id"))
        messages = tuple(self.messages)
        if len(messages) < 2:
            raise ValueError("messages must contain at least user and assistant turns")
        if any(not isinstance(message, ChatMessage) for message in messages):
            raise TypeError("messages must contain ChatMessage objects")
        if not any(message.role is MessageRole.USER for message in messages):
            raise ValueError("messages must contain a user turn")
        if messages[-1].role is not MessageRole.ASSISTANT:
            raise ValueError("the final message must be the assistant target")
        object.__setattr__(self, "messages", messages)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.example_id,
            "task": self.task.value,
            "messages": [message.to_dict() for message in self.messages],
        }


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Backend-neutral generation request."""

    request_id: str
    example_id: str
    system_prompt: str
    user_prompt: str

    def __post_init__(self) -> None:
        for field_name in ("request_id", "example_id", "system_prompt", "user_prompt"):
            object.__setattr__(self, field_name, _clean_text(getattr(self, field_name), field_name))


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Backend-neutral generated text tied to its request."""

    request_id: str
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _clean_text(self.request_id, "request_id"))
        object.__setattr__(self, "text", _clean_text(self.text, "text"))


@dataclass(frozen=True, slots=True)
class ReaderEvidence:
    """Validated evidence selection passed to an answer reader."""

    mode: EvidenceMode
    paragraph_titles: tuple[str, ...]
    supporting_facts: tuple[SupportingFact, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.mode, EvidenceMode):
            try:
                object.__setattr__(self, "mode", EvidenceMode(self.mode))
            except (TypeError, ValueError) as error:
                raise ValueError(f"unsupported evidence mode: {self.mode!r}") from error
        titles = tuple(_clean_text(title, "paragraph title") for title in self.paragraph_titles)
        facts = tuple(self.supporting_facts)
        if not titles:
            raise ValueError("reader evidence must contain at least one paragraph")
        if len(titles) != len(set(titles)):
            raise ValueError("paragraph_titles must not contain duplicates")
        if any(not isinstance(fact, SupportingFact) for fact in facts):
            raise TypeError("supporting_facts must contain SupportingFact objects")
        if len(facts) != len(set(facts)):
            raise ValueError("supporting_facts must not contain duplicates")
        if self.mode is EvidenceMode.FACTS and not facts:
            raise ValueError("fact evidence must contain at least one supporting fact")
        unknown_fact_titles = sorted({fact.title for fact in facts} - set(titles))
        if unknown_fact_titles:
            raise ValueError(
                "supporting facts reference paragraphs outside paragraph_titles: "
                f"{unknown_fact_titles}"
            )
        object.__setattr__(self, "paragraph_titles", titles)
        object.__setattr__(self, "supporting_facts", facts)


@dataclass(frozen=True, slots=True)
class JointPrediction:
    """Parsed answer and supporting-fact prediction."""

    answer: str
    supporting_facts: tuple[SupportingFact, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "answer", _clean_text(self.answer, "answer"))
        facts = tuple(self.supporting_facts)
        if not facts:
            raise ValueError("a joint prediction must contain supporting facts")
        if len(facts) != len(set(facts)):
            raise ValueError("supporting_facts must not contain duplicates")
        object.__setattr__(self, "supporting_facts", facts)


@dataclass(frozen=True, slots=True)
class PipelinePrediction:
    """Complete output of a modular Bactrainus pipeline."""

    example_id: str
    answer: str
    paragraph_titles: tuple[str, ...]
    supporting_facts: tuple[SupportingFact, ...]
    subquestions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "example_id", _clean_text(self.example_id, "example_id"))
        object.__setattr__(self, "answer", _clean_text(self.answer, "answer"))
        titles = tuple(_clean_text(title, "paragraph title") for title in self.paragraph_titles)
        facts = tuple(self.supporting_facts)
        subquestions = tuple(_clean_text(question, "subquestion") for question in self.subquestions)
        if not titles or not facts:
            raise ValueError("pipeline predictions require paragraphs and supporting facts")
        if len(titles) != len(set(titles)):
            raise ValueError("paragraph_titles must not contain duplicates")
        if len(facts) != len(set(facts)):
            raise ValueError("supporting_facts must not contain duplicates")
        if any(not isinstance(fact, SupportingFact) for fact in facts):
            raise TypeError("supporting_facts must contain SupportingFact objects")
        unknown_fact_titles = sorted({fact.title for fact in facts} - set(titles))
        if unknown_fact_titles:
            raise ValueError(
                "supporting facts reference paragraphs outside paragraph_titles: "
                f"{unknown_fact_titles}"
            )
        if len(subquestions) != len(set(subquestions)):
            raise ValueError("subquestions must not contain duplicates")
        object.__setattr__(self, "paragraph_titles", titles)
        object.__setattr__(self, "supporting_facts", facts)
        object.__setattr__(self, "subquestions", subquestions)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.example_id,
            "answer": self.answer,
            "paragraph_titles": list(self.paragraph_titles),
            "supporting_facts": [fact.to_list() for fact in self.supporting_facts],
            "subquestions": list(self.subquestions),
        }


def metadata_value(mapping: Mapping[str, object], key: str) -> str | None:
    """Return an optional metadata value with strict string validation."""

    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string when provided")
    return _source_identifier(value, key)
