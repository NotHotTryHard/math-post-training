"""Load project configuration files."""

from pathlib import Path

import yaml


def load_yaml_config(path):
    """Load a YAML file whose top-level value must be a mapping."""

    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise TypeError(f"Config {config_path} must contain a mapping at the top level")

    return data
