from dataclasses import dataclass

from bactrainus.prompts.renderer import (
    question_context_message,
    sentence_selector_message,
    serialize_paragraphs,
)


@dataclass(frozen=True)
class Paragraph:
    title: str
    sentences: tuple[str, ...]


def test_serialize_paragraphs_preserves_order_and_indices() -> None:
    text = serialize_paragraphs(
        [
            Paragraph("Second", ("S0", "S1")),
            Paragraph("First", ("T0",)),
        ]
    )
    assert text.index("Second") < text.index("First")
    assert "sentence 1: S1" in text


def test_reader_message_has_explicit_sections() -> None:
    message = question_context_message("Who?", "paragraph (A):\nsentence 0: A")
    assert message.startswith("Question: Who?")
    assert "\n\nContext:\n" in message


def test_sentence_selector_message_adds_numbered_subquestions() -> None:
    message = sentence_selector_message("Why?", "context", subquestions=("First?", "Second?"))
    assert "1. First?" in message
    assert "2. Second?" in message
