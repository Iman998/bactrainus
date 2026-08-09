"""Strict loading of canonical HotpotQA records."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..schemas import HotpotExample, Paragraph, SupportingFact, metadata_value


class DatasetFormatError(ValueError):
    """Raised when a source record does not satisfy the HotpotQA schema."""


def _sequence(value: object, field_name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DatasetFormatError(f"{field_name} must be a sequence")
    return value


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetFormatError(f"{key} must be a non-empty string")
    return value


def parse_hotpot_example(raw: Mapping[str, object], *, split: str | None = None) -> HotpotExample:
    """Validate and convert one raw HotpotQA mapping.

    Both the official ``_id`` field and release-friendly ``id`` field are
    accepted.  Context and supporting-fact order are preserved exactly.
    """

    raw_id = raw.get("_id", raw.get("id"))
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise DatasetFormatError("record must contain a non-empty '_id' or 'id'")

    paragraphs: list[Paragraph] = []
    for position, item in enumerate(_sequence(raw.get("context"), "context")):
        pair = _sequence(item, f"context[{position}]")
        if len(pair) != 2:
            raise DatasetFormatError(f"context[{position}] must contain a title and sentence list")
        title = pair[0]
        if not isinstance(title, str):
            raise DatasetFormatError(f"context[{position}] title must be a string")
        raw_sentences = _sequence(pair[1], f"context[{position}].sentences")
        sentences: list[str] = []
        for sentence_position, sentence in enumerate(raw_sentences):
            if not isinstance(sentence, str):
                raise DatasetFormatError(
                    f"context[{position}].sentences[{sentence_position}] must be a string"
                )
            sentences.append(sentence)
        paragraphs.append(Paragraph(title, tuple(sentences)))

    facts: list[SupportingFact] = []
    for position, item in enumerate(_sequence(raw.get("supporting_facts"), "supporting_facts")):
        pair = _sequence(item, f"supporting_facts[{position}]")
        if len(pair) != 2:
            raise DatasetFormatError(f"supporting_facts[{position}] must contain a title and index")
        title, index = pair
        if not isinstance(title, str):
            raise DatasetFormatError(f"supporting_facts[{position}] title must be a string")
        if isinstance(index, bool) or not isinstance(index, int):
            raise DatasetFormatError(f"supporting_facts[{position}] index must be an integer")
        facts.append(SupportingFact(title, index))

    resolved_split = split if split is not None else metadata_value(raw, "split")
    return HotpotExample(
        example_id=raw_id,
        question=_required_string(raw, "question"),
        answer=_required_string(raw, "answer"),
        context=tuple(paragraphs),
        supporting_facts=tuple(facts),
        split=resolved_split,
        question_type=metadata_value(raw, "type"),
        level=metadata_value(raw, "level"),
    )


def parse_hotpot_examples(
    records: Iterable[Mapping[str, object]], *, split: str | None = None
) -> tuple[HotpotExample, ...]:
    """Parse examples in source order and reject duplicate IDs."""

    examples: list[HotpotExample] = []
    seen: set[str] = set()
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise DatasetFormatError(f"record {position} must be a mapping")
        example = parse_hotpot_example(record, split=split)
        if example.example_id in seen:
            raise DatasetFormatError(f"duplicate example ID {example.example_id!r}")
        seen.add(example.example_id)
        examples.append(example)
    return tuple(examples)


def _load_json_records(path: Path) -> list[Mapping[str, object]]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise DatasetFormatError(f"dataset file is empty: {path}")
    records: list[Mapping[str, object]] = []
    if text.lstrip().startswith("["):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise DatasetFormatError(f"invalid JSON in {path}") from error
        if not isinstance(payload, list):
            raise DatasetFormatError("JSON dataset root must be a list")
        candidates: Iterable[Any] = payload
    else:
        parsed_lines: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                parsed_lines.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise DatasetFormatError(f"invalid JSON on line {line_number} of {path}") from error
        candidates = parsed_lines
    for position, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise DatasetFormatError(f"record {position} must be a JSON object")
        records.append(candidate)
    return records


def load_hotpot_examples(
    path: str | Path, *, split: str | None = None
) -> tuple[HotpotExample, ...]:
    """Load a JSON array or JSONL file while preserving source order."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return parse_hotpot_examples(_load_json_records(source), split=split)
