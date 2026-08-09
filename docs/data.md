# Data Release

The canonical public training-data repository is [`bactrianus/bactrainus-hotpotqa`](https://huggingface.co/datasets/bactrianus/bactrainus-hotpotqa).

The release contains deterministic training views of the official HotpotQA distractor training split at upstream revision `1908d6afbbead072334abe2965f91bd2709910ab`.

## Release boundary

Every completed configuration must contain 90,447 records with 90,447 unique HotpotQA source IDs. All configurations are train-only and preserve the same source-record identity.

| Configuration | Purpose | Record shape |
|---|---|---|
| `structured` | Canonical lossless task representation | Structured fields |
| `reader-sft` | Answer generation from selected evidence | Chat messages |
| `cot-reader-sft` | Indexed evidence trace plus answer | Chat messages |
| `paragraph-selector-sft` | Relevant-paragraph selection | Chat messages |
| `question-decomposer-sft` | Grounded ordered sub-question generation | Chat messages |
| `sentence-selector-sft` | Supporting-sentence selection | Chat messages |
| `decomposed-sentence-selector-sft` | Supporting-sentence selection with sub-questions | Chat messages |
| `joint-selector-reader-sft` | All-in-one evidence + answer target | Chat messages |

## Canonical structured schema

Each `structured` record contains:

```text
source_id: string
question: string
answer: string
question_type: string
difficulty: string
candidate_paragraphs:
  - title: string
    sentences: list[string]
supporting_facts:
  - title: string
    sentence_index: integer
gold_paragraph_titles: list[string]
```

The candidate paragraph order and the sentence order within every paragraph are preserved. Supporting-fact indices remain zero-based. `gold_paragraph_titles` is derived deterministically from the supporting-fact titles without losing the canonical fact representation.

## SFT-view schema

Each SFT record contains:

```text
source_id: string
task: string
messages:
  - role: string
    content: string
```

The chat views are deterministic projections of the same 90,447 canonical records. `source_id` is the join key across configurations and must never be regenerated from row position.

## Loading from Hugging Face

```python
from datasets import load_dataset

structured = load_dataset(
    "bactrianus/bactrainus-hotpotqa",
    "structured",
    split="train",
)

reader = load_dataset(
    "bactrianus/bactrainus-hotpotqa",
    "reader-sft",
    split="train",
)
```

Pin a dataset revision for reproducible work:

```python
structured = load_dataset(
    "bactrianus/bactrainus-hotpotqa",
    "structured",
    split="train",
    revision="<commit-or-release-tag>",
)
```

## Local construction

The authoritative release scripts and source/patch manifests are published with the dataset. The package exposes every view builder through the CLI:

```bash
bactrainus data build --help
```

Local builders satisfy these invariants:

1. Exactly one output record is produced per valid source example.
2. `source_id` remains identical across every generated view.
3. Source order is stable unless an explicitly documented sort is requested.
4. Candidate and sentence order remain unchanged.
5. Source annotation repairs are explicit and keyed by source ID.
6. Shard merging rejects duplicate IDs and incompatible schemas.

## Validation checklist

Before publishing a release, verify:

- 90,447 rows in each configuration;
- 90,447 unique `source_id` values in each configuration;
- identical ID sets across all eight configurations;
- the exact upstream candidate set (two to ten paragraphs; 89,609 records contain ten);
- every supporting-fact title exists among its candidates;
- every supporting-fact index is valid for its paragraph;
- no dev/test IDs, predictions, scores, secrets, or absolute local paths;
- deterministic checksums for every published shard.

## License and attribution

HotpotQA is distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The Bactrainus training release is an adaptation and uses the same license. Users must retain attribution, indicate modifications, and distribute adaptations under compatible terms.

The code that constructs these views is separately licensed under Apache-2.0. A code license does not override the dataset license.
