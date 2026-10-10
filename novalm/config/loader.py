from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from novalm.model.config import ModelConfig

ConfigType = TypeVar("ConfigType")


def load_validated_config(path: str | Path, config_type: type[ConfigType],
                          section: str | None = None) -> ConfigType:
    """Load a YAML mapping into a dataclass, applying its post-init validation."""
    if not is_dataclass(config_type):
        raise TypeError("config_type must be a dataclass type")
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise TypeError("configuration YAML must contain a mapping")
    values = raw.get(section, raw) if section else raw
    if not isinstance(values, dict):
        raise TypeError(f"configuration section {section!r} must be a mapping")
    try:
        return config_type(**values)
    except TypeError as exc:
        raise ValueError(f"invalid fields for {config_type.__name__}: {exc}") from exc


def load_model_config(path: str | Path) -> ModelConfig:
    with Path(path).open(encoding="utf-8") as stream:
        raw: dict[str, Any] = yaml.safe_load(stream)
    values = raw.get("model", raw)
    return ModelConfig(**values)
