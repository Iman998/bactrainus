"""Typed LoRA configuration and lazy PEFT integration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoraRecipe(BaseModel):
    """LoRA adapter configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    target_modules: tuple[str, ...]
    bias: str = "none"

    @field_validator("target_modules")
    @classmethod
    def require_targets(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("At least one LoRA target module is required")
        if len(set(value)) != len(value):
            raise ValueError("LoRA target modules must be unique")
        return value

    def to_peft_config(self) -> Any:
        """Build a ``peft.LoraConfig`` without importing PEFT at module import time."""

        try:
            from peft import LoraConfig
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("Install Bactrainus with the 'training' extra") from error
        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=list(self.target_modules),
            bias=self.bias,
            task_type="CAUSAL_LM",
        )
