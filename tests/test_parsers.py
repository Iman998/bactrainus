from __future__ import annotations

import pytest

from bactrainus.parsers import (
    ParseError,
    parse_answer,
    parse_joint_prediction,
    parse_paragraph_titles,
    parse_subquestions,
    parse_supporting_facts,
)
from bactrainus.schemas import SupportingFact


def test_parse_paragraph_titles_preserves_order() -> None:
    output = "selected paragraphs:\nParagraph: ***Alpha***\nParagraph ***Beta***"
    assert parse_paragraph_titles(output) == ("Alpha", "Beta")


def test_parse_supporting_facts_accepts_multiple_sentences_per_paragraph() -> None:
    output = """Supporting facts:
Paragraph: ***Alpha***
Sentence: ***0***
Sentence: ***2***
Paragraph: ***Beta***
Sentence: ***1***
"""
    assert parse_supporting_facts(output) == (
        SupportingFact("Alpha", 0),
        SupportingFact("Alpha", 2),
        SupportingFact("Beta", 1),
    )


def test_parsers_reject_ambiguous_or_malformed_output() -> None:
    with pytest.raises(ParseError, match="exactly three"):
        parse_paragraph_titles("Paragraph: ****Alpha***")
    with pytest.raises(ParseError, match="must follow"):
        parse_supporting_facts("Sentence: ***0***")
    with pytest.raises(ParseError, match="non-negative integer"):
        parse_supporting_facts("Paragraph: ***Alpha***\nSentence: ***one***")
    with pytest.raises(ParseError, match="duplicates"):
        parse_supporting_facts("Paragraph: ***Alpha***\nSentence: ***0***\nSentence: ***0***")
    with pytest.raises(ParseError, match="no following sentence"):
        parse_supporting_facts("Paragraph: ***Alpha***\nParagraph: ***Beta***\nSentence: ***0***")
    with pytest.raises(ParseError, match="no following sentence"):
        parse_supporting_facts("Paragraph: ***Alpha***")


def test_answer_and_joint_parsing() -> None:
    assert parse_answer("Answer: ***Tehran***") == "Tehran"
    assert parse_answer("Tehran") == "Tehran"

    joint = parse_joint_prediction(
        "Supporting facts:\nParagraph: ***Alpha***\nSentence: ***1***\nAnswer: ***Tehran***"
    )
    assert joint.answer == "Tehran"
    assert joint.supporting_facts == (SupportingFact("Alpha", 1),)

    with pytest.raises(ParseError, match="final"):
        parse_joint_prediction("Answer: ***Tehran***\nParagraph: ***Alpha***\nSentence: ***1***")


def test_subquestion_parser_handles_numbered_and_bulleted_lines() -> None:
    assert parse_subquestions("sub-questions:\n1. Who founded Alpha?\n- Where is Beta?") == (
        "Who founded Alpha?",
        "Where is Beta?",
    )
    with pytest.raises(ParseError, match="duplicate"):
        parse_subquestions("Who?\nWho?")
    with pytest.raises(ParseError, match="must not be empty"):
        parse_subquestions("-")
