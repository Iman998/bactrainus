from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import pytest

from bactrainus.components import (
    GenerativeAnswerReader,
    GenerativeParagraphSelector,
    GenerativeQuestionDecomposer,
    GenerativeSentenceSelector,
)
from bactrainus.components.generative import GenerationContractError
from bactrainus.data.hotpot import parse_hotpot_example
from bactrainus.pipeline import BactrainusPipeline, PipelineValidationError
from bactrainus.prompts import PROMPT_VERSION, PromptCatalog
from bactrainus.schemas import (
    EvidenceMode,
    GenerationRequest,
    GenerationResult,
    HotpotExample,
    ReaderEvidence,
    SupportingFact,
)


def example(identifier: str = "p-1") -> HotpotExample:
    return parse_hotpot_example(
        {
            "_id": identifier,
            "question": "Where do Alpha and Beta overlap?",
            "answer": "Tehran",
            "context": [
                ["Alpha", ["Alpha is in Tehran."]],
                ["Beta", ["Beta is also in Tehran."]],
                ["Noise", ["Noise is elsewhere."]],
            ],
            "supporting_facts": [["Alpha", 0], ["Beta", 0]],
        }
    )


class FakeParagraphSelector:
    def select(self, item: HotpotExample) -> Sequence[str]:
        return ("Alpha", "Beta")


class FakeDecomposer:
    def decompose(self, item: HotpotExample, paragraph_titles: Sequence[str]) -> Sequence[str]:
        assert tuple(paragraph_titles) == ("Alpha", "Beta")
        return ("Where is Alpha?", "Where is Beta?")


class FakeSentenceSelector:
    def select(
        self,
        item: HotpotExample,
        paragraph_titles: Sequence[str],
        subquestions: Sequence[str] = (),
    ) -> Sequence[SupportingFact]:
        assert tuple(paragraph_titles) == ("Alpha", "Beta")
        return (SupportingFact("Alpha", 0), SupportingFact("Beta", 0))


class FakeReader:
    def __init__(self) -> None:
        self.last_evidence: ReaderEvidence | None = None

    def answer(self, item: HotpotExample, evidence: ReaderEvidence) -> str:
        self.last_evidence = evidence
        return "Tehran"


def test_pipeline_injects_components_and_preserves_outputs() -> None:
    reader = FakeReader()
    pipeline = BactrainusPipeline(
        paragraph_selector=FakeParagraphSelector(),
        decomposer=FakeDecomposer(),
        sentence_selector=FakeSentenceSelector(),
        reader=reader,
        reader_mode=EvidenceMode.PARAGRAPHS,
    )
    prediction = pipeline.run(example())

    assert prediction.answer == "Tehran"
    assert prediction.paragraph_titles == ("Alpha", "Beta")
    assert prediction.subquestions == ("Where is Alpha?", "Where is Beta?")
    assert reader.last_evidence is not None
    assert reader.last_evidence.mode is EvidenceMode.PARAGRAPHS


def test_pipeline_run_many_is_ordered_and_id_strict() -> None:
    pipeline = BactrainusPipeline(
        paragraph_selector=FakeParagraphSelector(),
        sentence_selector=FakeSentenceSelector(),
        reader=FakeReader(),
    )
    predictions = pipeline.run_many((example("first"), example("second")))
    assert tuple(prediction.example_id for prediction in predictions) == (
        "first",
        "second",
    )
    duplicate = example("same")
    with pytest.raises(PipelineValidationError, match="duplicate input"):
        pipeline.run_many((duplicate, duplicate))


class BadSentenceSelector(FakeSentenceSelector):
    def select(
        self,
        item: HotpotExample,
        paragraph_titles: Sequence[str],
        subquestions: Sequence[str] = (),
    ) -> Sequence[SupportingFact]:
        return (SupportingFact("Noise", 0),)


def test_pipeline_rejects_cross_stage_evidence_leakage() -> None:
    pipeline = BactrainusPipeline(
        paragraph_selector=FakeParagraphSelector(),
        sentence_selector=BadSentenceSelector(),
        reader=FakeReader(),
    )
    with pytest.raises(PipelineValidationError, match="outside selected"):
        pipeline.run(example())


class ScriptedGenerator:
    responses: ClassVar[dict[str, str]] = {
        "paragraph-selector": ("selected paragraphs:\nparagraph ***Alpha***\nparagraph ***Beta***"),
        "decomposer": ("sub-questions:\n1. Where is Alpha?\n2. Where is Beta?"),
        "sentence-selector": (
            "supporting facts:\nparagraph ***Alpha***\nsentence ***0***\n"
            "paragraph ***Beta***\nsentence ***0***"
        ),
        "reader": "answer: ***Tehran***",
    }

    def generate(self, requests: Sequence[GenerationRequest]) -> Sequence[GenerationResult]:
        results = []
        for request in requests:
            stage = request.request_id.rsplit(":", 1)[1]
            results.append(GenerationResult(request.request_id, self.responses[stage]))
        return results


def test_full_generative_pipeline_uses_backend_contract() -> None:
    generator = ScriptedGenerator()
    paragraph_selector = GenerativeParagraphSelector(generator)
    decomposer = GenerativeQuestionDecomposer(generator)
    sentence_selector = GenerativeSentenceSelector(generator)
    reader = GenerativeAnswerReader(generator)
    pipeline = BactrainusPipeline(
        paragraph_selector=paragraph_selector,
        decomposer=decomposer,
        sentence_selector=sentence_selector,
        reader=reader,
    )
    prediction = pipeline.run(example())
    assert prediction.answer == "Tehran"
    assert prediction.supporting_facts == (
        SupportingFact("Alpha", 0),
        SupportingFact("Beta", 0),
    )
    assert {
        paragraph_selector.prompt_version,
        decomposer.prompt_version,
        sentence_selector.prompt_version,
        reader.prompt_version,
    } == {PROMPT_VERSION}
    prompts = PromptCatalog()
    assert paragraph_selector.system_prompt == prompts.paragraph_selector
    assert reader.system_prompt == prompts.direct_reader


class MisalignedGenerator(ScriptedGenerator):
    def generate(self, requests: Sequence[GenerationRequest]) -> Sequence[GenerationResult]:
        return (GenerationResult("wrong-id", "Paragraph: ***Alpha***"),)


def test_generation_backend_must_preserve_request_ids() -> None:
    selector = GenerativeParagraphSelector(MisalignedGenerator())
    with pytest.raises(GenerationContractError, match="expected"):
        selector.select(example())
