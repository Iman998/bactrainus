"""Deterministic, ID-safe merging for generated data shards."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


class ShardMergeError(ValueError):
    """Raised when shards contain missing, duplicate, or unexpected records."""


def _record_id(record: Mapping[str, object], id_field: str) -> str:
    value = record.get(id_field)
    if not isinstance(value, str) or not value.strip():
        raise ShardMergeError(f"every shard record requires a non-empty {id_field!r}")
    if value != value.strip():
        raise ShardMergeError(f"record {id_field!r} must not contain surrounding whitespace")
    return value


def merge_shards(
    shards: Iterable[Iterable[Mapping[str, object]]],
    *,
    expected_ids: Sequence[str] | None = None,
    id_field: str = "id",
    require_complete: bool = True,
) -> tuple[dict[str, object], ...]:
    """Merge records by ID without relying on DataFrame row alignment.

    When ``expected_ids`` is provided, output order follows it exactly.
    Otherwise shard and record encounter order is preserved.
    """

    if not isinstance(id_field, str) or not id_field.strip():
        raise ShardMergeError("id_field must be a non-empty string")
    records_by_id: dict[str, dict[str, object]] = {}
    encounter_order: list[str] = []
    for shard_position, shard in enumerate(shards):
        for record_position, record in enumerate(shard):
            if not isinstance(record, Mapping):
                raise ShardMergeError(
                    f"shard {shard_position} record {record_position} is not a mapping"
                )
            identifier = _record_id(record, id_field)
            if identifier in records_by_id:
                raise ShardMergeError(f"duplicate record ID {identifier!r}")
            records_by_id[identifier] = dict(record)
            encounter_order.append(identifier)

    if expected_ids is None:
        order = encounter_order
    else:
        order = list(expected_ids)
        if any(
            not isinstance(identifier, str)
            or not identifier.strip()
            or identifier != identifier.strip()
            for identifier in order
        ):
            raise ShardMergeError("expected_ids must contain non-empty strings")
        if len(order) != len(set(order)):
            raise ShardMergeError("expected_ids contains duplicates")
        expected = set(order)
        unexpected = [identifier for identifier in encounter_order if identifier not in expected]
        if unexpected:
            raise ShardMergeError(f"unexpected record IDs: {unexpected}")
        missing = [identifier for identifier in order if identifier not in records_by_id]
        if missing and require_complete:
            raise ShardMergeError(f"missing record IDs: {missing}")
        order = [identifier for identifier in order if identifier in records_by_id]

    return tuple(records_by_id[identifier] for identifier in order)


def read_jsonl(path: str | Path) -> tuple[dict[str, object], ...]:
    """Read strict JSONL records."""

    source = Path(path)
    records: list[dict[str, object]] = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ShardMergeError(f"invalid JSON on line {line_number} of {source}") from error
            if not isinstance(value, dict):
                raise ShardMergeError(f"line {line_number} of {source} is not a JSON object")
            records.append(value)
    return tuple(records)


def write_jsonl_atomic(path: str | Path, records: Iterable[Mapping[str, object]]) -> None:
    """Atomically write records so an interrupted merge cannot corrupt output."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            for record in records:
                temporary.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
        Path(temporary_name).replace(destination)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def merge_jsonl_shards(
    shard_paths: Sequence[str | Path],
    *,
    expected_ids: Sequence[str] | None = None,
    id_field: str = "id",
    require_complete: bool = True,
    output_path: str | Path | None = None,
) -> tuple[dict[str, object], ...]:
    """Read, validate, merge, and optionally atomically write JSONL shards."""

    merged = merge_shards(
        (read_jsonl(path) for path in shard_paths),
        expected_ids=expected_ids,
        id_field=id_field,
        require_complete=require_complete,
    )
    if output_path is not None:
        write_jsonl_atomic(output_path, merged)
    return merged
