"""Command-line interface for deterministic release workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from .backends import OpenAICompatibleGenerator
from .components import (
    GenerativeAnswerReader,
    GenerativeParagraphSelector,
    GenerativeQuestionDecomposer,
    GenerativeSentenceSelector,
)
from .config import load_config
from .data import (
    CotReaderViewBuilder,
    DecomposedSentenceSelectorViewBuilder,
    JointViewBuilder,
    ParagraphSelectorViewBuilder,
    QuestionDecomposerViewBuilder,
    ReaderViewBuilder,
    SentenceSelectorViewBuilder,
    StructuredViewBuilder,
    build_many,
    load_hotpot_examples,
)
from .data.shards import read_jsonl, write_jsonl_atomic
from .evaluation import evaluate_answers, evaluate_evidence, evaluate_joint
from .pipeline import BactrainusPipeline
from .schemas import EvidenceMode, GenerationRequest
from .training import SftRecipe
from .training.sft import run_sft

app = typer.Typer(
    name="bactrainus",
    help="Clean data, inference, evaluation, and reproduction utilities for Bactrainus.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Build and validate ID-preserving training views.")
app.add_typer(data_app, name="data")
console = Console()


class DataView(str, Enum):
    """Public deterministic training views."""

    STRUCTURED = "structured"
    READER = "reader-sft"
    COT_READER = "cot-reader-sft"
    PARAGRAPH = "paragraph-selector-sft"
    DECOMPOSER = "question-decomposer-sft"
    SENTENCE = "sentence-selector-sft"
    DECOMPOSED_SENTENCE = "decomposed-sentence-selector-sft"
    JOINT = "joint-selector-reader-sft"


def _view_builder(view: DataView) -> Any:
    builders = {
        DataView.STRUCTURED: StructuredViewBuilder,
        DataView.READER: ReaderViewBuilder,
        DataView.COT_READER: CotReaderViewBuilder,
        DataView.PARAGRAPH: ParagraphSelectorViewBuilder,
        DataView.DECOMPOSER: QuestionDecomposerViewBuilder,
        DataView.SENTENCE: SentenceSelectorViewBuilder,
        DataView.DECOMPOSED_SENTENCE: DecomposedSentenceSelectorViewBuilder,
        DataView.JOINT: JointViewBuilder,
    }
    return builders[view]()


def _require_count(actual: int, expected: int | None, label: str) -> None:
    if expected is not None and actual != expected:
        raise typer.BadParameter(f"{label} has {actual:,} records; expected {expected:,}")


@data_app.command("validate")
def validate_data(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    expected_count: Annotated[
        int | None,
        typer.Option("--expected-count", min=1, help="Fail unless this many records load."),
    ] = None,
    split: Annotated[str, typer.Option(help="Split label assigned during validation.")] = "train",
) -> None:
    """Validate canonical HotpotQA JSON or JSONL without writing output."""

    examples = load_hotpot_examples(source, split=split)
    _require_count(len(examples), expected_count, str(source))
    unique_titles = all(
        len(example.context) == len(example.paragraph_by_title) for example in examples
    )
    console.print(
        {
            "records": len(examples),
            "unique_ids": len({example.example_id for example in examples}),
            "all_paragraph_titles_unique": unique_titles,
            "split": split,
        }
    )


@data_app.command("build")
def build_data(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Argument(dir_okay=False)],
    view: Annotated[DataView, typer.Option(case_sensitive=False)] = DataView.STRUCTURED,
    expected_count: Annotated[
        int | None,
        typer.Option("--expected-count", min=1, help="Fail before writing on count mismatch."),
    ] = None,
    split: Annotated[str, typer.Option(help="Split label assigned to source records.")] = "train",
) -> None:
    """Build one deterministic, source-ID-preserving JSONL training view."""

    examples = load_hotpot_examples(source, split=split)
    _require_count(len(examples), expected_count, str(source))
    records = build_many(_view_builder(view), examples)
    write_jsonl_atomic(output, (record.to_dict() for record in records))
    console.print(f"Wrote {len(records):,} {view.value} records to {output}")


def _record_id(record: Mapping[str, object], position: int) -> str:
    identifier = record.get("source_id", record.get("id", record.get("_id")))
    if not isinstance(identifier, str) or not identifier.strip():
        raise typer.BadParameter(f"record {position} has no non-empty id")
    return identifier


def _answer_map(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    values: dict[str, object] = {}
    for position, record in enumerate(records):
        identifier = _record_id(record, position)
        if identifier in values:
            raise typer.BadParameter(f"duplicate id {identifier!r}")
        values[identifier] = record.get("answer", "")
    return values


def _fact_map(records: Sequence[Mapping[str, object]]) -> dict[str, Sequence[Any]]:
    values: dict[str, Sequence[Any]] = {}
    for position, record in enumerate(records):
        identifier = _record_id(record, position)
        facts = record.get("supporting_facts", ())
        if isinstance(facts, (str, bytes)) or not isinstance(facts, Sequence):
            raise typer.BadParameter(f"record {identifier!r} supporting_facts must be a sequence")
        values[identifier] = facts
    return values


def _metrics_table(
    answer: Mapping[str, float | int],
    evidence: Mapping[str, float | int],
    joint: Mapping[str, float | int],
) -> Table:
    table = Table(title="HotpotQA metrics (percentage points)")
    table.add_column("Metric")
    table.add_column("Answer", justify="right")
    table.add_column("Supporting facts", justify="right")
    table.add_column("Joint", justify="right")
    labels = (
        ("Exact match", "exact_match"),
        ("F1", "f1"),
        ("Precision", "precision"),
        ("Recall", "recall"),
    )
    for label, key in labels:
        table.add_row(
            label,
            f"{float(answer[key]):.2f}",
            f"{float(evidence[key]):.2f}",
            f"{float(joint[key]):.2f}",
        )
    return table


@app.command("evaluate")
def evaluate(
    gold: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    predictions: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    strict: Annotated[
        bool,
        typer.Option("--strict/--allow-missing", help="Require identical prediction IDs."),
    ] = True,
    json_output: Annotated[Path | None, typer.Option("--json-output", dir_okay=False)] = None,
) -> None:
    """Evaluate answer, supporting-fact, and joint HotpotQA metrics."""

    gold_records = read_jsonl(gold)
    prediction_records = read_jsonl(predictions)
    gold_answers = _answer_map(gold_records)
    predicted_answers = _answer_map(prediction_records)
    gold_facts = _fact_map(gold_records)
    predicted_facts = _fact_map(prediction_records)
    answer = evaluate_answers(gold_answers, predicted_answers, strict=strict).as_percentages()
    evidence = evaluate_evidence(gold_facts, predicted_facts, strict=strict).as_percentages()
    joint = evaluate_joint(
        gold_answers,
        gold_facts,
        predicted_answers,
        predicted_facts,
        strict=strict,
    ).as_percentages()
    console.print(_metrics_table(answer, evidence, joint))
    if json_output is not None:
        write_jsonl_atomic(
            json_output,
            ({"answer": answer, "supporting_facts": evidence, "joint": joint},),
        )


@app.command("infer")
def infer(
    requests_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Argument(dir_okay=False)],
    model: Annotated[str, typer.Option(help="OpenAI-compatible model identifier.")],
    api_key_env: Annotated[
        str,
        typer.Option(help="Name of the environment variable containing the key."),
    ] = "OPENAI_API_KEY",
    base_url: Annotated[str | None, typer.Option(help="Optional compatible API base URL.")] = None,
    temperature: Annotated[float, typer.Option(min=0.0)] = 0.0,
    top_p: Annotated[float, typer.Option(min=0.000001, max=1.0)] = 1.0,
    max_tokens: Annotated[int, typer.Option(min=1)] = 512,
) -> None:
    """Generate raw responses for strict JSONL request records."""

    records = read_jsonl(requests_path)
    requests: list[GenerationRequest] = []
    for position, record in enumerate(records):
        identifier = _record_id(record, position)
        system_prompt = record.get("system_prompt")
        user_prompt = record.get("user_prompt")
        if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
            raise typer.BadParameter(
                f"record {identifier!r} requires system_prompt and user_prompt strings"
            )
        requests.append(GenerationRequest(identifier, identifier, system_prompt, user_prompt))
    backend = OpenAICompatibleGenerator(
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    results = backend.generate(requests)
    write_jsonl_atomic(
        output,
        ({"id": result.request_id, "text": result.text} for result in results),
    )
    console.print(f"Wrote {len(results):,} aligned generations to {output}")


@app.command("pipeline")
def run_pipeline(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Argument(dir_okay=False)],
    model: Annotated[str, typer.Option(help="OpenAI-compatible model identifier.")],
    api_key_env: Annotated[str, typer.Option()] = "OPENAI_API_KEY",
    base_url: Annotated[str | None, typer.Option()] = None,
    limit: Annotated[int | None, typer.Option(min=1, help="Optional explicit cost guard.")] = None,
    decomposition: Annotated[bool, typer.Option("--decomposition/--no-decomposition")] = True,
    reader_mode: Annotated[EvidenceMode, typer.Option(case_sensitive=False)] = EvidenceMode.FACTS,
) -> None:
    """Run the modular selector--reader pipeline through one compatible API."""

    examples = load_hotpot_examples(source)
    if limit is not None:
        examples = examples[:limit]
    backend = OpenAICompatibleGenerator(
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
    )
    pipeline = BactrainusPipeline(
        paragraph_selector=GenerativeParagraphSelector(backend),
        decomposer=GenerativeQuestionDecomposer(backend) if decomposition else None,
        sentence_selector=GenerativeSentenceSelector(backend),
        reader=GenerativeAnswerReader(backend),
        reader_mode=reader_mode,
    )
    predictions = pipeline.run_many(examples)
    write_jsonl_atomic(output, (prediction.to_dict() for prediction in predictions))
    console.print(f"Wrote {len(predictions):,} pipeline predictions to {output}")


@app.command("train")
def train(
    recipe: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    execute: Annotated[
        bool,
        typer.Option(
            "--execute",
            help="Start the expensive training run; otherwise only validate the recipe.",
        ),
    ] = False,
) -> None:
    """Validate or execute a fully resolved SFT recipe."""

    resolved = load_config(recipe, SftRecipe)
    console.print(
        {
            "name": resolved.name,
            "model_id": resolved.model_id,
            "model_revision": resolved.model_revision,
            "dataset": f"{resolved.dataset_id}/{resolved.dataset_config}",
            "dataset_revision": resolved.dataset_revision,
            "effective_batch_size_per_process": resolved.effective_batch_size,
        }
    )
    if execute:
        run_sft(resolved)
    else:
        console.print("Recipe is valid. Re-run with --execute to start training.")


@app.command("version")
def version() -> None:
    """Print the installed package version."""

    from . import __version__

    console.print(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
