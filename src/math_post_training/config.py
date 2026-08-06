"""Small, shared helpers for loading project configuration."""

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a project configuration file is malformed."""


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file whose top-level value must be a mapping."""

    config_path = Path(path)
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"Cannot read config {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in {config_path}: {error}") from error

    if not isinstance(data, dict):
        raise ConfigError(f"Config {config_path} must contain a mapping at the top level")

    return data
