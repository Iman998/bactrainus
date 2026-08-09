"""Strict parsers for Bactrainus' text-generation interfaces."""

from __future__ import annotations

import re

from .schemas import JointPrediction, SupportingFact


class ParseError(ValueError):
    """Raised when generated text violates an expected output grammar."""


_LABELED_LINE = re.compile(r"^(?P<label>[A-Za-z][A-Za-z _-]*)(?:\s*:\s*|\s+)(?P<payload>.+?)\s*$")
_LIST_PREFIX = re.compile(r"^\s*(?:[-*](?:\s+|$)|\d+[.)](?:\s+|$))?")


def _nonempty_lines(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("generated text must be a string")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _marked_value(line: str, expected_label: str) -> str:
    match = _LABELED_LINE.fullmatch(line)
    if match is None or match.group("label").strip().casefold() != expected_label.casefold():
        raise ParseError(f"expected a {expected_label!r} line, received {line!r}")
    payload = match.group("payload").strip()
    if (
        not payload.startswith("***")
        or not payload.endswith("***")
        or payload.startswith("****")
        or payload.endswith("****")
    ):
        raise ParseError(f"{expected_label} values must be enclosed by exactly three asterisks")
    value = payload[3:-3].strip()
    if not value or "***" in value or value.startswith("*") or value.endswith("*"):
        raise ParseError(f"invalid marked {expected_label} value")
    return value


def parse_paragraph_titles(text: str) -> tuple[str, ...]:
    """Parse ordered ``Paragraph: ***title***`` lines."""

    lines = _nonempty_lines(text)
    if lines and lines[0].rstrip(":").casefold() == "selected paragraphs":
        lines = lines[1:]
    if not lines:
        raise ParseError("paragraph selection is empty")
    titles = tuple(_marked_value(line, "paragraph") for line in lines)
    if len(titles) != len(set(titles)):
        raise ParseError("paragraph selection contains duplicate titles")
    return titles


def parse_supporting_facts(text: str) -> tuple[SupportingFact, ...]:
    """Parse paragraph/sentence pairs while preserving generated order.

    One paragraph line may be followed by one or more sentence lines.  A
    sentence line before any paragraph line is invalid.
    """

    lines = _nonempty_lines(text)
    if lines and lines[0].rstrip(":").casefold() in {
        "supporting facts",
        "evidence",
    }:
        lines = lines[1:]
    if not lines:
        raise ParseError("supporting-fact selection is empty")

    current_title: str | None = None
    current_fact_count = 0
    facts: list[SupportingFact] = []
    for line in lines:
        match = _LABELED_LINE.fullmatch(line)
        if match is None:
            raise ParseError(f"invalid supporting-fact line {line!r}")
        label = match.group("label").strip().casefold()
        if label == "paragraph":
            if current_title is not None and current_fact_count == 0:
                raise ParseError(f"paragraph {current_title!r} has no following sentence line")
            current_title = _marked_value(line, "paragraph")
            current_fact_count = 0
            continue
        if label != "sentence":
            raise ParseError(f"unexpected supporting-fact label {label!r}")
        if current_title is None:
            raise ParseError("a sentence line must follow a paragraph line")
        raw_index = _marked_value(line, "sentence")
        if not raw_index.isdecimal():
            raise ParseError(f"sentence index must be a non-negative integer: {raw_index!r}")
        facts.append(SupportingFact(current_title, int(raw_index)))
        current_fact_count += 1

    if current_title is not None and current_fact_count == 0:
        raise ParseError(f"paragraph {current_title!r} has no following sentence line")
    if not facts:
        raise ParseError("supporting-fact selection contains no sentence indices")
    if len(facts) != len(set(facts)):
        raise ParseError("supporting-fact selection contains duplicates")
    return tuple(facts)


def parse_answer(text: str, *, allow_plain: bool = True) -> str:
    """Parse a concise answer, optionally accepting an unlabelled response."""

    lines = _nonempty_lines(text)
    if not lines:
        raise ParseError("answer is empty")
    if len(lines) == 1 and _LABELED_LINE.fullmatch(lines[0]):
        match = _LABELED_LINE.fullmatch(lines[0])
        assert match is not None
        if match.group("label").strip().casefold() == "answer":
            return _marked_value(lines[0], "answer")
    if allow_plain:
        answer = " ".join(lines).strip()
        if answer:
            return answer
    raise ParseError("expected exactly one Answer: ***...*** line")


def parse_joint_prediction(text: str) -> JointPrediction:
    """Parse supporting facts followed by exactly one marked answer line."""

    lines = _nonempty_lines(text)
    answer_positions: list[int] = []
    for index, line in enumerate(lines):
        match = _LABELED_LINE.fullmatch(line)
        if match and match.group("label").strip().casefold() == "answer":
            answer_positions.append(index)
    if len(answer_positions) != 1:
        raise ParseError("joint output must contain exactly one answer line")
    answer_index = answer_positions[0]
    if answer_index != len(lines) - 1:
        raise ParseError("the answer line must be the final non-empty line")
    answer = _marked_value(lines[answer_index], "answer")
    facts = parse_supporting_facts("\n".join(lines[:answer_index]))
    return JointPrediction(answer=answer, supporting_facts=facts)


def parse_subquestions(text: str) -> tuple[str, ...]:
    """Parse ordered bullet, numbered, or plain-line subquestions."""

    lines = _nonempty_lines(text)
    if lines and lines[0].rstrip(":").casefold() in {
        "sub-questions",
        "subquestions",
    }:
        lines = lines[1:]
    questions: list[str] = []
    for line in lines:
        value = _LIST_PREFIX.sub("", line).strip()
        if value.casefold().startswith("subquestion:"):
            value = value.split(":", 1)[1].strip()
        if not value:
            raise ParseError("subquestion must not be empty")
        questions.append(value)
    if not questions:
        raise ParseError("decomposition is empty")
    if len(questions) != len(set(questions)):
        raise ParseError("decomposition contains duplicate subquestions")
    return tuple(questions)
