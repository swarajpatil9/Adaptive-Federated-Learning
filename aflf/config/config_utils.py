"""Config loading and normalization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .defaults import get_config_defaults
from .validator import ConfigValidator


def load_and_validate_config(
    config_path: str | Path,
    validator: ConfigValidator | None = None,
) -> Dict[str, Any]:
    """Load YAML config, apply defaults, and validate fields."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML config '{path}': {exc}") from exc

    config = _deep_merge_dicts(get_config_defaults(), raw)
    (validator or ConfigValidator()).validate(config)
    return config


def _deep_merge_dicts(defaults: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    for key, value in user.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged
