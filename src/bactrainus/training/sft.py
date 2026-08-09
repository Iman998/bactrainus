"""Supervised fine-tuning recipe validation and lazy trainer construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from bactrainus.training.lora import LoraRecipe


class SftRecipe(BaseModel):
    """Complete SFT recipe independent of a specific trainer implementation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    model_id: str
    model_revision: str
    dataset_id: str
    dataset_config: str
    dataset_revision: str
    split: str = "train"
    output_dir: Path
    epochs: float = Field(gt=0)
    per_device_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    max_sequence_length: int = Field(gt=0)
    warmup_ratio: float = Field(ge=0, lt=1)
    scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True
    lora: LoraRecipe

    @property
    def effective_batch_size(self) -> int:
        """Return the per-process effective batch size."""

        return self.per_device_batch_size * self.gradient_accumulation_steps


def run_sft(recipe: SftRecipe) -> Any:
    """Run SFT from a validated recipe using optional training dependencies.

    This implementation is a clean reproduction derived from the paper's recorded
    hyperparameters. It does not claim to be the unavailable historical training script.
    """

    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Install Bactrainus with the 'training' extra") from error

    dataset = load_dataset(
        recipe.dataset_id,
        recipe.dataset_config,
        split=recipe.split,
        revision=recipe.dataset_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        recipe.model_id,
        revision=recipe.model_revision,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        recipe.model_id,
        revision=recipe.model_revision,
        torch_dtype="auto",
        device_map="auto",
    )
    dataset = dataset.map(
        lambda record: _render_chat_record(record, tokenizer),
        desc="Rendering model-native chat templates",
    )
    arguments = SFTConfig(
        output_dir=str(recipe.output_dir),
        num_train_epochs=recipe.epochs,
        per_device_train_batch_size=recipe.per_device_batch_size,
        gradient_accumulation_steps=recipe.gradient_accumulation_steps,
        learning_rate=recipe.learning_rate,
        max_seq_length=recipe.max_sequence_length,
        warmup_ratio=recipe.warmup_ratio,
        lr_scheduler_type=recipe.scheduler,
        seed=recipe.seed,
        bf16=recipe.bf16,
        gradient_checkpointing=recipe.gradient_checkpointing,
        report_to="none",
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=recipe.lora.to_peft_config(),
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(recipe.output_dir)
    return trainer


def _render_chat_record(record: Mapping[str, object], tokenizer: Any) -> dict[str, str]:
    """Materialize one validated role/content record with the native chat template."""

    messages = record.get("messages")
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise ValueError("every SFT record requires a messages sequence")
    normalized: list[dict[str, str]] = []
    for position, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"messages[{position}] must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"messages[{position}] has an unsupported role")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{position}] has empty content")
        normalized.append({"role": str(role), "content": content})
    if not normalized or normalized[-1]["role"] != "assistant":
        raise ValueError("the final SFT message must be the assistant target")
    rendered = tokenizer.apply_chat_template(
        normalized,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str) or not rendered.strip():
        raise ValueError("the tokenizer returned an empty chat serialization")
    return {"text": rendered}
