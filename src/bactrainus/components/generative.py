"""Backend-neutral generative implementations of pipeline components."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..data.builders import serialize_context, serialize_facts
from ..parsers import (
    parse_answer,
    parse_paragraph_titles,
    parse_subquestions,
    parse_supporting_facts,
)
from ..prompts import PROMPT_VERSION, PromptCatalog
from ..prompts.renderer import (
    decomposer_message,
    question_context_message,
    selector_message,
    sentence_selector_message,
)
from ..protocols import TextGenerator
from ..schemas import (
    EvidenceMode,
    GenerationRequest,
    GenerationResult,
    HotpotExample,
    ReaderEvidence,
    SupportingFact,
)

_PROMPTS = PromptCatalog()


class GenerationContractError(RuntimeError):
    """Raised when a generation backend breaks request/result alignment."""


def _generate_one(generator: TextGenerator, request: GenerationRequest) -> str:
    results = tuple(generator.generate((request,)))
    if len(results) != 1:
        raise GenerationContractError(f"backend returned {len(results)} results for one request")
    result = results[0]
    if not isinstance(result, GenerationResult):
        raise GenerationContractError("backend must return GenerationResult objects")
    if result.request_id != request.request_id:
        raise GenerationContractError(
            f"backend returned request ID {result.request_id!r}; expected {request.request_id!r}"
        )
    return result.text


def _validate_titles(example: HotpotExample, titles: Sequence[str]) -> tuple[str, ...]:
    if isinstance(titles, (str, bytes)):
        raise TypeError("paragraph titles must be a sequence of strings")
    resolved = tuple(titles)
    if not resolved:
        raise ValueError("paragraph selection must not be empty")
    if any(not isinstance(title, str) or not title.strip() for title in resolved):
        raise TypeError("paragraph titles must be non-empty strings")
    if len(resolved) != len(set(resolved)):
        raise ValueError("paragraph selection contains duplicate titles")
    unknown = [title for title in resolved if title not in example.paragraph_by_title]
    if unknown:
        raise ValueError(f"selector returned unknown paragraph titles: {unknown}")
    return resolved


@dataclass(slots=True)
class AllParagraphSelector:
    """Deterministic selector used for full-context ablations."""

    def select(self, example: HotpotExample) -> tuple[str, ...]:
        return tuple(paragraph.title for paragraph in example.context)


@dataclass(slots=True)
class GenerativeParagraphSelector:
    generator: TextGenerator
    system_prompt: str = _PROMPTS.paragraph_selector
    prompt_version: str = field(default=PROMPT_VERSION, init=False)

    def select(self, example: HotpotExample) -> tuple[str, ...]:
        request = GenerationRequest(
            request_id=f"{example.example_id}:paragraph-selector",
            example_id=example.example_id,
            system_prompt=self.system_prompt,
            user_prompt=selector_message(example.question, serialize_context(example)),
        )
        return _validate_titles(
            example, parse_paragraph_titles(_generate_one(self.generator, request))
        )


@dataclass(slots=True)
class GenerativeQuestionDecomposer:
    generator: TextGenerator
    system_prompt: str = _PROMPTS.question_decomposer
    prompt_version: str = field(default=PROMPT_VERSION, init=False)

    def decompose(self, example: HotpotExample, paragraph_titles: Sequence[str]) -> tuple[str, ...]:
        titles = _validate_titles(example, paragraph_titles)
        request = GenerationRequest(
            request_id=f"{example.example_id}:decomposer",
            example_id=example.example_id,
            system_prompt=self.system_prompt,
            user_prompt=decomposer_message(example.question, serialize_context(example, titles)),
        )
        return parse_subquestions(_generate_one(self.generator, request))


@dataclass(slots=True)
class GenerativeSentenceSelector:
    generator: TextGenerator
    system_prompt: str = _PROMPTS.sentence_selector
    prompt_version: str = field(default=PROMPT_VERSION, init=False)

    def select(
        self,
        example: HotpotExample,
        paragraph_titles: Sequence[str],
        subquestions: Sequence[str] = (),
    ) -> tuple[SupportingFact, ...]:
        titles = _validate_titles(example, paragraph_titles)
        request = GenerationRequest(
            request_id=f"{example.example_id}:sentence-selector",
            example_id=example.example_id,
            system_prompt=self.system_prompt,
            user_prompt=sentence_selector_message(
                example.question,
                serialize_context(example, titles),
                subquestions=subquestions,
            ),
        )
        facts = parse_supporting_facts(_generate_one(self.generator, request))
        title_set = set(titles)
        for fact in facts:
            if fact.title not in title_set:
                raise ValueError(
                    f"sentence selector returned fact outside selected paragraphs: {fact}"
                )
            paragraph = example.paragraph(fact.title)
            if fact.sentence_index >= len(paragraph.sentences):
                raise ValueError(f"sentence selector returned out-of-range fact: {fact}")
        return facts


@dataclass(slots=True)
class GenerativeAnswerReader:
    generator: TextGenerator
    system_prompt: str = _PROMPTS.direct_reader
    prompt_version: str = field(default=PROMPT_VERSION, init=False)

    def answer(self, example: HotpotExample, evidence: ReaderEvidence) -> str:
        title_set = set(evidence.paragraph_titles)
        unknown = title_set - set(example.paragraph_by_title)
        if unknown:
            raise ValueError(f"reader evidence contains unknown paragraphs: {sorted(unknown)}")
        if evidence.mode is EvidenceMode.FACTS:
            for fact in evidence.supporting_facts:
                if fact.title not in title_set:
                    raise ValueError(f"fact {fact} is not contained in reader paragraph selection")
            serialized = serialize_facts(example, evidence.supporting_facts)
        else:
            serialized = serialize_context(example, evidence.paragraph_titles)
        request = GenerationRequest(
            request_id=f"{example.example_id}:reader",
            example_id=example.example_id,
            system_prompt=self.system_prompt,
            user_prompt=question_context_message(example.question, serialized),
        )
        return parse_answer(_generate_one(self.generator, request), allow_plain=True)
