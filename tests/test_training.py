from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError

from bactrainus.training.lora import LoraRecipe
from bactrainus.training.sft import SftRecipe, _render_chat_record, run_sft


def test_lora_targets_must_be_unique() -> None:
    with pytest.raises(ValidationError):
        LoraRecipe(rank=8, alpha=16, dropout=0.05, target_modules=("q_proj", "q_proj"))


def test_lora_recipe_is_immutable() -> None:
    recipe = LoraRecipe(rank=8, alpha=16, dropout=0.05, target_modules=("q_proj",))
    with pytest.raises(ValidationError):
        recipe.rank = 16  # type: ignore[misc]


def recipe(tmp_path: Path) -> SftRecipe:
    return SftRecipe(
        name="reader-test",
        model_id="local/model",
        model_revision="model-commit",
        dataset_id="local/data",
        dataset_config="reader-sft",
        dataset_revision="dataset-commit",
        output_dir=tmp_path / "output",
        epochs=1,
        per_device_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        max_sequence_length=512,
        warmup_ratio=0.03,
        lora=LoraRecipe(
            rank=8,
            alpha=16,
            dropout=0.05,
            target_modules=("q_proj", "v_proj"),
        ),
    )


def test_sft_recipe_effective_batch_and_required_revision(tmp_path: Path) -> None:
    resolved = recipe(tmp_path)
    assert resolved.effective_batch_size == 8
    payload = resolved.model_dump()
    payload.pop("dataset_revision")
    with pytest.raises(ValidationError, match="dataset_revision"):
        SftRecipe.model_validate(payload)


class FakeTokenizer:
    def apply_chat_template(self, messages: object, **options: object) -> str:
        assert messages
        assert options == {"tokenize": False, "add_generation_prompt": False}
        return "rendered-chat"


def test_chat_materialization_is_strict() -> None:
    valid = {
        "messages": [
            {"role": "system", "content": "Instruction"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
    }
    assert _render_chat_record(valid, FakeTokenizer()) == {"text": "rendered-chat"}
    with pytest.raises(ValueError, match="messages sequence"):
        _render_chat_record({}, FakeTokenizer())
    with pytest.raises(ValueError, match="unsupported role"):
        _render_chat_record({"messages": [{"role": "tool", "content": "x"}]}, FakeTokenizer())
    with pytest.raises(ValueError, match="assistant target"):
        _render_chat_record({"messages": [{"role": "user", "content": "x"}]}, FakeTokenizer())


def test_run_sft_pins_data_and_materializes_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}

    class FakeDataset:
        def map(self, function: Any, **kwargs: object) -> FakeDataset:
            calls["map_options"] = kwargs
            calls["mapped"] = function(
                {
                    "messages": [
                        {"role": "user", "content": "Question"},
                        {"role": "assistant", "content": "Answer"},
                    ]
                }
            )
            return self

    datasets_module = ModuleType("datasets")

    def load_dataset(*args: object, **kwargs: object) -> FakeDataset:
        calls["dataset_args"] = args
        calls["dataset_options"] = kwargs
        return FakeDataset()

    datasets_module.load_dataset = load_dataset  # type: ignore[attr-defined]

    transformers_module = ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeTokenizer:
            calls["tokenizer"] = (args, kwargs)
            return FakeTokenizer()

    class AutoModel:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            calls["model"] = (args, kwargs)
            return object()

    transformers_module.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
    transformers_module.AutoModelForCausalLM = AutoModel  # type: ignore[attr-defined]

    trl_module = ModuleType("trl")

    class SFTConfig:
        def __init__(self, **kwargs: object) -> None:
            calls["sft_config"] = kwargs

    class SFTTrainer:
        def __init__(self, **kwargs: object) -> None:
            calls["trainer"] = kwargs

        def train(self) -> None:
            calls["trained"] = True

        def save_model(self) -> None:
            calls["saved"] = True

    trl_module.SFTConfig = SFTConfig  # type: ignore[attr-defined]
    trl_module.SFTTrainer = SFTTrainer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "trl", trl_module)
    monkeypatch.setattr(LoraRecipe, "to_peft_config", lambda self: "peft-config")
    monkeypatch.setattr(FakeTokenizer, "save_pretrained", lambda self, path: None, raising=False)

    trainer = run_sft(recipe(tmp_path))

    assert isinstance(trainer, SFTTrainer)
    assert calls["dataset_options"] == {
        "split": "train",
        "revision": "dataset-commit",
    }
    assert calls["mapped"] == {"text": "rendered-chat"}
    assert calls["sft_config"]["dataset_text_field"] == "text"
    assert calls["trained"] and calls["saved"]
