from pathlib import Path

import pytest
from pydantic import BaseModel

from bactrainus.config import ConfigurationError, load_config, load_yaml


class ExampleConfig(BaseModel):
    name: str


def test_load_config_expands_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BACTRAINUS_TEST_NAME", "reader")
    path = tmp_path / "config.yaml"
    path.write_text("name: ${BACTRAINUS_TEST_NAME}\n", encoding="utf-8")
    assert load_config(path, ExampleConfig).name == "reader"


def test_load_yaml_requires_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_yaml(path)
