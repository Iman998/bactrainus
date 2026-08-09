"""Typed configuration loading with environment-variable expansion."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml
from pydantic import BaseModel

ConfigT = TypeVar("ConfigT", bound=BaseModel)


class ConfigurationError(ValueError):
    """Raised when a configuration file is missing or invalid."""


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and expand ``${ENVIRONMENT_VARIABLE}`` references."""

    if not path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ConfigurationError(f"Expected a YAML mapping in {path}")
    return cast(dict[str, Any], _expand(payload))


def load_config(path: Path, model: type[ConfigT]) -> ConfigT:
    """Validate a YAML file against a Pydantic configuration model."""

    return model.model_validate(load_yaml(path))
