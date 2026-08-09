from __future__ import annotations

import json

import pytest

from bactrainus.data.builders import (
    JointViewBuilder,
    ParagraphSelectorViewBuilder,
    ReaderViewBuilder,
    SentenceSelectorViewBuilder,
    StructuredViewBuilder,
    build_many,
)
from bactrainus.data.hotpot import (
    DatasetFormatError,
    load_hotpot_examples,
    parse_hotpot_example,
    parse_hotpot_examples,
)
from bactrainus.schemas import TaskKind


def raw_example(identifier: str = "example-1") -> dict[str, object]:
    return {
        "_id": identifier,
        "question": "Which city links Alpha and Beta?",
        "answer": "Tehran",
        "context": [
            ["Alpha", ["Alpha was founded early.", "Alpha is in Tehran."]],
            ["Beta", ["Beta also operates in Tehran.", "Beta has a museum."]],
            ["Distractor", ["This sentence is irrelevant."]],
            ["Distractor 2", ["This sentence is also irrelevant."]],
            ["Distractor 3", ["This sentence is also irrelevant."]],
            ["Distractor 4", ["This sentence is also irrelevant."]],
            ["Distractor 5", ["This sentence is also irrelevant."]],
            ["Distractor 6", ["This sentence is also irrelevant."]],
            ["Distractor 7", ["This sentence is also irrelevant."]],
            ["Distractor 8", ["This sentence is also irrelevant."]],
        ],
        "supporting_facts": [["Alpha", 1], ["Beta", 0]],
        "type": "bridge",
        "level": "hard",
    }


def test_parse_and_structured_view_preserve_order() -> None:
    example = parse_hotpot_example(raw_example(), split="train")

    assert example.example_id == "example-1"
    assert tuple(paragraph.title for paragraph in example.context)[:3] == (
        "Alpha",
        "Beta",
        "Distractor",
    )
    assert example.gold_paragraph_titles == ("Alpha", "Beta")
    assert example.split == "train"

    record = StructuredViewBuilder().build(example)
    payload = record.to_dict()
    assert set(payload) == {
        "source_id",
        "question",
        "answer",
        "question_type",
        "difficulty",
        "candidate_paragraphs",
        "supporting_facts",
        "gold_paragraph_titles",
    }
    assert payload["source_id"] == "example-1"
    assert payload["candidate_paragraphs"][0] == {
        "title": "Alpha",
        "sentences": ["Alpha was founded early.", "Alpha is in Tehran."],
    }
    assert payload["supporting_facts"] == [
        {"title": "Alpha", "sentence_index": 1},
        {"title": "Beta", "sentence_index": 0},
    ]
    assert payload["gold_paragraph_titles"] == ["Alpha", "Beta"]


def test_training_views_are_explicit_and_machine_parseable() -> None:
    example = parse_hotpot_example(raw_example())

    reader = ReaderViewBuilder().build(example)
    paragraph = ParagraphSelectorViewBuilder().build(example)
    sentence = SentenceSelectorViewBuilder().build(example)
    joint = JointViewBuilder().build(example)

    assert reader.task is TaskKind.READER
    assert [message.role.value for message in reader.messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert "Sentence 1: Alpha is in Tehran." in reader.messages[1].content
    assert reader.messages[-1].content == "answer: ***Tehran***"

    assert paragraph.task is TaskKind.PARAGRAPH_SELECTOR
    assert "Distractor" in paragraph.messages[1].content
    assert paragraph.messages[-1].content.splitlines() == [
        "selected paragraphs:",
        "paragraph ***Alpha***",
        "paragraph ***Beta***",
    ]

    assert sentence.task is TaskKind.SENTENCE_SELECTOR
    assert "Distractor" not in sentence.messages[1].content
    assert "sentence ***1***" in sentence.messages[-1].content

    assert joint.task is TaskKind.JOINT
    assert joint.messages[-1].content.startswith("supporting facts:\n")
    assert joint.messages[-1].content.endswith("answer: ***Tehran***")

    for record in (reader, paragraph, sentence, joint):
        payload = record.to_dict()
        assert set(payload) == {"source_id", "task", "messages"}
        assert payload["source_id"] == example.example_id
        assert payload["messages"][-1]["role"] == "assistant"


def test_build_many_rejects_duplicate_ids() -> None:
    examples = parse_hotpot_examples([raw_example(), raw_example("example-2")])
    records = build_many(ReaderViewBuilder(), examples)
    assert tuple(record.example_id for record in records) == ("example-1", "example-2")

    with pytest.raises(ValueError, match="duplicate example ID"):
        build_many(ReaderViewBuilder(), (examples[0], examples[0]))


def test_invalid_supporting_fact_is_not_silently_repaired() -> None:
    raw = raw_example()
    raw["supporting_facts"] = [["Alpha", 99]]
    with pytest.raises(ValueError, match="out of range"):
        parse_hotpot_example(raw)

    raw = raw_example()
    raw["supporting_facts"] = [["Missing", 0]]
    with pytest.raises(ValueError, match="unknown paragraph"):
        parse_hotpot_example(raw)

    raw = raw_example()
    raw["_id"] = " example-1 "
    with pytest.raises(ValueError, match="surrounding whitespace"):
        parse_hotpot_example(raw)


def test_loader_supports_json_array_and_jsonl(tmp_path) -> None:
    array_path = tmp_path / "examples.json"
    array_path.write_text(json.dumps([raw_example()]), encoding="utf-8")
    assert load_hotpot_examples(array_path)[0].example_id == "example-1"

    jsonl_path = tmp_path / "examples.jsonl"
    jsonl_path.write_text(json.dumps(raw_example()) + "\n", encoding="utf-8")
    assert load_hotpot_examples(jsonl_path)[0].question_type == "bridge"


def test_loader_rejects_duplicate_and_non_object_records(tmp_path) -> None:
    with pytest.raises(DatasetFormatError, match="duplicate"):
        parse_hotpot_examples([raw_example(), raw_example()])

    path = tmp_path / "bad.json"
    path.write_text("[1]", encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="JSON object"):
        load_hotpot_examples(path)
