"""HotpotQA loading, view construction, and deterministic shard utilities."""

from .builders import (
    JointViewBuilder,
    ParagraphSelectorViewBuilder,
    ReaderViewBuilder,
    SentenceSelectorViewBuilder,
    StructuredViewBuilder,
    build_many,
    validate_release_example,
)
from .hotpot import load_hotpot_examples, parse_hotpot_example, parse_hotpot_examples
from .shards import merge_jsonl_shards, merge_shards

__all__ = [
    "JointViewBuilder",
    "ParagraphSelectorViewBuilder",
    "ReaderViewBuilder",
    "SentenceSelectorViewBuilder",
    "StructuredViewBuilder",
    "build_many",
    "load_hotpot_examples",
    "merge_jsonl_shards",
    "merge_shards",
    "parse_hotpot_example",
    "parse_hotpot_examples",
    "validate_release_example",
]
