from __future__ import annotations

import json

import pytest

from bactrainus.data.shards import (
    ShardMergeError,
    merge_jsonl_shards,
    merge_shards,
    read_jsonl,
)


def test_merge_shards_uses_expected_id_order() -> None:
    shards = (
        ({"id": "b", "value": 2},),
        ({"id": "a", "value": 1}, {"id": "c", "value": 3}),
    )
    merged = merge_shards(shards, expected_ids=("a", "b", "c"))
    assert tuple(record["id"] for record in merged) == ("a", "b", "c")


def test_merge_shards_rejects_duplicate_missing_and_unexpected_ids() -> None:
    with pytest.raises(ShardMergeError, match="duplicate"):
        merge_shards((({"id": "a"},), ({"id": "a"},)))
    with pytest.raises(ShardMergeError, match="missing"):
        merge_shards((({"id": "a"},),), expected_ids=("a", "b"))
    with pytest.raises(ShardMergeError, match="unexpected"):
        merge_shards((({"id": "x"},),), expected_ids=("a",))


def test_jsonl_merge_is_atomic_and_round_trips(tmp_path) -> None:
    first = tmp_path / "part-1.jsonl"
    second = tmp_path / "part-2.jsonl"
    output = tmp_path / "merged.jsonl"
    first.write_text(json.dumps({"id": "b", "value": 2}) + "\n", encoding="utf-8")
    second.write_text(json.dumps({"id": "a", "value": 1}) + "\n", encoding="utf-8")

    merged = merge_jsonl_shards((first, second), expected_ids=("a", "b"), output_path=output)
    assert merged == read_jsonl(output)
    assert tuple(record["id"] for record in merged) == ("a", "b")


def test_jsonl_reader_reports_bad_line(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "a"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ShardMergeError, match="line 2"):
        read_jsonl(path)
