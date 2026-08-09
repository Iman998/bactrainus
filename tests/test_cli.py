from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import bactrainus.cli as cli_module
from bactrainus.cli import app
from bactrainus.schemas import GenerationResult

runner = CliRunner()


def source_record() -> dict[str, object]:
    return {
        "_id": "example-1",
        "question": "Where was the writer born?",
        "answer": "Paris",
        "type": "bridge",
        "level": "easy",
        "context": [
            ["Writer", ["The writer was born in Paris."]],
            ["Paris", ["Paris is in France."]],
            ["Distractor 1", ["Unrelated sentence one."]],
            ["Distractor 2", ["Unrelated sentence two."]],
            ["Distractor 3", ["Unrelated sentence three."]],
            ["Distractor 4", ["Unrelated sentence four."]],
            ["Distractor 5", ["Unrelated sentence five."]],
            ["Distractor 6", ["Unrelated sentence six."]],
            ["Distractor 7", ["Unrelated sentence seven."]],
            ["Distractor 8", ["Unrelated sentence eight."]],
        ],
        "supporting_facts": [["Writer", 0], ["Paris", 0]],
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_cli_version_and_help() -> None:
    version = runner.invoke(app, ["version"])
    assert version.exit_code == 0
    assert "0.1.0" in version.stdout
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "pipeline" in help_result.stdout


def test_cli_validates_and_builds_data(tmp_path: Path) -> None:
    source = tmp_path / "train.json"
    output = tmp_path / "reader.jsonl"
    write_json(source, [source_record()])

    validated = runner.invoke(app, ["data", "validate", str(source), "--expected-count", "1"])
    assert validated.exit_code == 0
    assert "records" in validated.stdout

    built = runner.invoke(
        app,
        [
            "data",
            "build",
            str(source),
            str(output),
            "--view",
            "reader-sft",
            "--expected-count",
            "1",
        ],
    )
    assert built.exit_code == 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["source_id"] == "example-1"
    assert record["task"] == "reader"
    assert [message["role"] for message in record["messages"]] == [
        "system",
        "user",
        "assistant",
    ]


def test_cli_count_guard_does_not_write(tmp_path: Path) -> None:
    source = tmp_path / "train.json"
    output = tmp_path / "output.jsonl"
    write_json(source, [source_record()])
    result = runner.invoke(
        app,
        [
            "data",
            "build",
            str(source),
            str(output),
            "--expected-count",
            "2",
        ],
    )
    assert result.exit_code != 0
    assert not output.exists()


def test_cli_evaluates_joint_predictions(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    metrics = tmp_path / "metrics.jsonl"
    record = {
        "id": "example-1",
        "answer": "Paris",
        "supporting_facts": [["Writer", 0], ["Paris", 0]],
    }
    write_jsonl(gold, [record])
    write_jsonl(predictions, [record])

    result = runner.invoke(
        app,
        ["evaluate", str(gold), str(predictions), "--json-output", str(metrics)],
    )

    assert result.exit_code == 0
    assert "100.00" in result.stdout
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert payload["joint"]["exact_match"] == 100.0


def test_cli_validates_resolved_training_recipe(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        """name: reader-test
model_id: local/model
model_revision: commit
dataset_id: local/data
dataset_config: reader-sft
dataset_revision: dataset-commit
output_dir: outputs/reader
epochs: 1
per_device_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 0.0001
max_sequence_length: 512
warmup_ratio: 0.03
lora:
  rank: 8
  alpha: 16
  dropout: 0.05
  target_modules: [q_proj, v_proj]
""",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["train", str(recipe)])

    assert result.exit_code == 0
    assert "Recipe is valid" in result.stdout
    assert "effective_batch_size_per_process" in result.stdout


class ScriptedBackend:
    def __init__(self, **_: object) -> None:
        pass

    def generate(self, requests: object) -> tuple[GenerationResult, ...]:
        results: list[GenerationResult] = []
        for request in requests:  # type: ignore[union-attr]
            identifier = request.request_id
            if identifier.endswith(":paragraph-selector"):
                text = "Paragraph: ***Writer***\nParagraph: ***Paris***"
            elif identifier.endswith(":decomposer"):
                text = "1. Where was the writer born?\n2. Which country contains it?"
            elif identifier.endswith(":sentence-selector"):
                text = (
                    "Paragraph: ***Writer***\nSentence: ***0***\n"
                    "Paragraph: ***Paris***\nSentence: ***0***"
                )
            elif identifier.endswith(":reader"):
                text = "Answer: ***Paris***"
            else:
                text = "raw response"
            results.append(GenerationResult(identifier, text))
        return tuple(results)


def test_cli_infer_and_pipeline_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_module, "OpenAICompatibleGenerator", ScriptedBackend)
    requests = tmp_path / "requests.jsonl"
    raw_output = tmp_path / "raw.jsonl"
    write_jsonl(
        requests,
        [{"id": "request-1", "system_prompt": "System", "user_prompt": "User"}],
    )
    inferred = runner.invoke(
        app, ["infer", str(requests), str(raw_output), "--model", "test-model"]
    )
    assert inferred.exit_code == 0
    assert json.loads(raw_output.read_text(encoding="utf-8"))["text"] == "raw response"

    source = tmp_path / "source.json"
    pipeline_output = tmp_path / "pipeline.jsonl"
    write_json(source, [source_record()])
    pipelined = runner.invoke(
        app,
        [
            "pipeline",
            str(source),
            str(pipeline_output),
            "--model",
            "test-model",
            "--limit",
            "1",
        ],
    )
    assert pipelined.exit_code == 0
    prediction = json.loads(pipeline_output.read_text(encoding="utf-8"))
    assert prediction["answer"] == "Paris"
    assert prediction["subquestions"]
