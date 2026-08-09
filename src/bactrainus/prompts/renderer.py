"""Deterministic serialization and message construction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol


class ParagraphLike(Protocol):
    """Structural type required by paragraph serializers."""

    title: str
    sentences: Sequence[str]


def serialize_paragraphs(
    paragraphs: Iterable[ParagraphLike], *, include_sentence_indices: bool = True
) -> str:
    """Serialize paragraphs deterministically in their supplied order."""

    blocks: list[str] = []
    for paragraph in paragraphs:
        lines = [f"paragraph ({paragraph.title}):"]
        for index, sentence in enumerate(paragraph.sentences):
            prefix = f"sentence {index}: " if include_sentence_indices else ""
            lines.append(f"{prefix}{sentence}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def question_context_message(question: str, context: str) -> str:
    """Construct the reader user message."""

    return f"Question: {question}\n\nContext:\n{context}"


def selector_message(question: str, sources: str) -> str:
    """Construct a selector user message."""

    return f"Question: {question}\n\nInformation Sources:\n{sources}"


def decomposer_message(question: str, selected_paragraphs: str) -> str:
    """Construct the decomposer user message."""

    return f"Question: {question}\n\nSelected Paragraphs:\n{selected_paragraphs}"


def sentence_selector_message(
    question: str,
    selected_paragraphs: str,
    *,
    subquestions: Sequence[str] = (),
) -> str:
    """Construct a sentence-selector user message with optional decomposition."""

    decomposition = ""
    if subquestions:
        items = "\n".join(f"{index}. {text}" for index, text in enumerate(subquestions, 1))
        decomposition = f"\n\nSub-questions:\n{items}"
    return f"Question: {question}{decomposition}\n\nSelected Paragraphs:\n{selected_paragraphs}"
