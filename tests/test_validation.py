from __future__ import annotations

import pytest

from bactrainus.data.builders import (
    ReaderViewBuilder,
    format_answer_target,
    format_paragraph_target,
    format_supporting_fact_target,
    serialize_context,
    serialize_facts,
    validate_release_example,
)
from bactrainus.data.hotpot import parse_hotpot_example
from bactrainus.pipeline import (
    PipelineValidationError,
    _facts,
    _subquestions,
    _titles,
)
from bactrainus.schemas import (
    ChatMessage,
    EvidenceMode,
    HotpotExample,
    JointPrediction,
    MessageRole,
    Paragraph,
    PipelinePrediction,
    ReaderEvidence,
    StructuredRecord,
    SupportingFact,
    TaskKind,
    TrainingRecord,
    metadata_value,
)


def raw_example() -> dict[str, object]:
    return {
        "_id": "validation-1",
        "question": "Where do Alpha and Beta overlap?",
        "answer": "Tehran",
        "type": "bridge",
        "level": "easy",
        "context": [
            ["Alpha", ["Alpha is in Tehran."]],
            ["Beta", ["Beta is in Tehran."]],
            *[[f"Noise {index}", ["Irrelevant."]] for index in range(1, 9)],
        ],
        "supporting_facts": [["Alpha", 0], ["Beta", 0]],
    }


def example() -> HotpotExample:
    return parse_hotpot_example(raw_example(), split="train")


def test_atomic_schema_objects_reject_invalid_values() -> None:
    with pytest.raises(TypeError, match="integer"):
        SupportingFact("Alpha", True)
    with pytest.raises(ValueError, match="non-negative"):
        SupportingFact("Alpha", -1)
    with pytest.raises(ValueError, match="surrounding whitespace"):
        SupportingFact(" Alpha", 0)
    with pytest.raises(TypeError, match="sequence"):
        Paragraph("Alpha", "sentence")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        Paragraph("Alpha", ())
    with pytest.raises(TypeError, match="every sentence"):
        Paragraph("Alpha", (1,))  # type: ignore[arg-type]


def test_example_and_structured_record_validation() -> None:
    paragraph = Paragraph("Alpha", ("Text.",))
    fact = SupportingFact("Alpha", 0)
    with pytest.raises(ValueError, match="context"):
        HotpotExample("id", "q", "a", (), (fact,))
    with pytest.raises(ValueError, match="must not be empty"):
        HotpotExample("id", "q", "a", (paragraph,), ())
    with pytest.raises(ValueError, match="unique"):
        HotpotExample("id", "q", "a", (paragraph, paragraph), (fact,))
    with pytest.raises(ValueError, match="duplicates"):
        HotpotExample("id", "q", "a", (paragraph,), (fact, fact))

    inference_example = HotpotExample("id", "q", "a", (paragraph,), (fact,))
    with pytest.raises(ValueError, match="question_type"):
        inference_example.to_dict()
    with pytest.raises(KeyError, match="no paragraph"):
        inference_example.paragraph("Missing")
    with pytest.raises(TypeError, match="HotpotExample"):
        StructuredRecord("wrong")  # type: ignore[arg-type]
    assert StructuredRecord(example(), task="structured").task is TaskKind.STRUCTURED  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported task"):
        StructuredRecord(example(), task="unknown")  # type: ignore[arg-type]


def test_chat_and_training_record_contracts() -> None:
    user = ChatMessage("user", "Question")  # type: ignore[arg-type]
    assistant = ChatMessage(MessageRole.ASSISTANT, "Answer")
    assert user.role is MessageRole.USER
    with pytest.raises(ValueError, match="unsupported message"):
        ChatMessage("tool", "x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        ChatMessage(MessageRole.USER, " ")
    with pytest.raises(ValueError, match="at least"):
        TrainingRecord("id", TaskKind.READER, (assistant,))
    with pytest.raises(TypeError, match="ChatMessage"):
        TrainingRecord("id", TaskKind.READER, (user, "bad"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="user turn"):
        TrainingRecord("id", TaskKind.READER, (assistant, assistant))
    with pytest.raises(ValueError, match="assistant target"):
        TrainingRecord("id", TaskKind.READER, (user, user))
    record = TrainingRecord("id", "reader", (user, assistant))  # type: ignore[arg-type]
    assert record.task is TaskKind.READER
    with pytest.raises(ValueError, match="unsupported task"):
        TrainingRecord("id", "bad", (user, assistant))  # type: ignore[arg-type]


def test_reader_and_prediction_evidence_contracts() -> None:
    fact = SupportingFact("Alpha", 0)
    assert ReaderEvidence("facts", ("Alpha",), (fact,)).mode is EvidenceMode.FACTS  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported evidence"):
        ReaderEvidence("bad", ("Alpha",), (fact,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one paragraph"):
        ReaderEvidence(EvidenceMode.PARAGRAPHS, ())
    with pytest.raises(ValueError, match="duplicates"):
        ReaderEvidence(EvidenceMode.PARAGRAPHS, ("Alpha", "Alpha"))
    with pytest.raises(TypeError, match="SupportingFact"):
        ReaderEvidence(EvidenceMode.PARAGRAPHS, ("Alpha",), ("bad",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one supporting"):
        ReaderEvidence(EvidenceMode.FACTS, ("Alpha",))
    with pytest.raises(ValueError, match="outside"):
        ReaderEvidence(EvidenceMode.FACTS, ("Alpha",), (SupportingFact("Beta", 0),))

    with pytest.raises(ValueError, match="must contain"):
        JointPrediction("answer", ())
    with pytest.raises(ValueError, match="duplicates"):
        JointPrediction("answer", (fact, fact))
    with pytest.raises(ValueError, match="require"):
        PipelinePrediction("id", "answer", (), ())
    with pytest.raises(ValueError, match="outside"):
        PipelinePrediction("id", "answer", ("Alpha",), (SupportingFact("Beta", 0),))
    with pytest.raises(ValueError, match="subquestions"):
        PipelinePrediction("id", "answer", ("Alpha",), (fact,), ("q", "q"))


def test_release_builder_and_serialization_guards() -> None:
    item = example()
    assert "Paragraph 0 (Alpha)" in serialize_context(item)
    with pytest.raises(TypeError, match="sequence"):
        serialize_context(item, "Alpha")
    with pytest.raises(ValueError, match="duplicates"):
        serialize_context(item, ("Alpha", "Alpha"))
    with pytest.raises(ValueError, match="unknown"):
        serialize_context(item, ("Missing",))
    with pytest.raises(TypeError, match="SupportingFact"):
        serialize_facts(item, ("bad",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="out of range"):
        serialize_facts(item, (SupportingFact("Alpha", 2),))

    with pytest.raises(TypeError, match="sequence"):
        format_paragraph_target("Alpha")
    with pytest.raises(ValueError, match="must not be empty"):
        format_paragraph_target(())
    with pytest.raises(ValueError, match="duplicate"):
        format_paragraph_target(("Alpha", "Alpha"))
    with pytest.raises(TypeError, match="SupportingFact"):
        format_supporting_fact_target(("bad",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        format_answer_target(" ")

    raw = raw_example()
    raw["type"] = "other"
    with pytest.raises(ValueError, match="question_type"):
        ReaderViewBuilder().build(parse_hotpot_example(raw, split="train"))
    raw = raw_example()
    raw["level"] = "unknown"
    with pytest.raises(ValueError, match="difficulty"):
        validate_release_example(parse_hotpot_example(raw, split="train"))
    with pytest.raises(ValueError, match="train-only"):
        validate_release_example(parse_hotpot_example(raw_example(), split="validation"))


def test_pipeline_validation_helpers_cover_every_boundary() -> None:
    item = example()
    with pytest.raises(PipelineValidationError, match="sequence"):
        _titles(item, "Alpha")
    with pytest.raises(PipelineValidationError, match="no paragraphs"):
        _titles(item, ())
    with pytest.raises(PipelineValidationError, match="non-empty"):
        _titles(item, ("",))
    with pytest.raises(PipelineValidationError, match="duplicate"):
        _titles(item, ("Alpha", "Alpha"))
    with pytest.raises(PipelineValidationError, match="unknown"):
        _titles(item, ("Missing",))
    with pytest.raises(PipelineValidationError, match="sequence"):
        _subquestions("Question")
    with pytest.raises(PipelineValidationError, match="empty"):
        _subquestions(())
    with pytest.raises(PipelineValidationError, match="duplicate"):
        _subquestions(("q", "q"))
    with pytest.raises(PipelineValidationError, match="sequence"):
        _facts(item, ("Alpha",), "bad")  # type: ignore[arg-type]
    with pytest.raises(PipelineValidationError, match="no facts"):
        _facts(item, ("Alpha",), ())
    with pytest.raises(PipelineValidationError, match="SupportingFact"):
        _facts(item, ("Alpha",), ("bad",))  # type: ignore[arg-type]
    with pytest.raises(PipelineValidationError, match="duplicate"):
        fact = SupportingFact("Alpha", 0)
        _facts(item, ("Alpha",), (fact, fact))
    with pytest.raises(PipelineValidationError, match="out-of-range"):
        _facts(item, ("Alpha",), (SupportingFact("Alpha", 2),))


def test_metadata_value_preserves_source_identity() -> None:
    assert metadata_value({}, "type") is None
    assert metadata_value({"type": "bridge"}, "type") == "bridge"
    with pytest.raises(TypeError, match="string"):
        metadata_value({"type": 1}, "type")
