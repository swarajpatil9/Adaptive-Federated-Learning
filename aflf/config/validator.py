"""Strict configuration validator for reproducible AFLF experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ValidationRule:
    """Rule for validating one config key path."""

    path: str
    expected_type: type
    minimum: float | None = None
    allowed_values: List[Any] | None = None


class ConfigValidator:
    """Validate config schema and value ranges before runtime."""

    REQUIRED_SECTIONS = ("data", "model", "federated", "training")

    RULES = (
        ValidationRule("training.learning_rate", float, minimum=0.0),
        ValidationRule("training.local_epochs", int, minimum=1),
        ValidationRule("training.batch_size", int, minimum=1),
        ValidationRule("federated.num_rounds", int, minimum=1),
        ValidationRule("federated.clients_per_round", int, minimum=1),
        ValidationRule("data.num_clients", int, minimum=1),
    )

    def validate(self, config: Dict[str, Any]) -> None:
        """Raise ValueError when config is invalid."""
        errors: List[str] = []

        for section in self.REQUIRED_SECTIONS:
            if section not in config:
                errors.append(f"Missing required section: '{section}'")

        for rule in self.RULES:
            value, exists = self._read_path(config, rule.path)
            if not exists:
                errors.append(f"Missing required field: '{rule.path}'")
                continue

            if rule.expected_type is float and isinstance(value, int):
                value = float(value)

            if not isinstance(value, rule.expected_type):
                errors.append(
                    f"Field '{rule.path}' must be of type "
                    f"{rule.expected_type.__name__}, got {type(value).__name__}"
                )
                continue

            if rule.minimum is not None:
                if rule.path == "training.learning_rate":
                    if float(value) <= rule.minimum:
                        errors.append("Field 'training.learning_rate' must be > 0")
                elif float(value) < rule.minimum:
                    errors.append(
                        f"Field '{rule.path}' must be >= {rule.minimum}, got {value}"
                    )

            if rule.allowed_values and value not in rule.allowed_values:
                errors.append(
                    f"Field '{rule.path}' must be one of {rule.allowed_values}, got {value}"
                )

        if not errors:
            return

        error_msg = "Configuration validation failed:\n- " + "\n- ".join(errors)
        raise ValueError(error_msg)

    @staticmethod
    def _read_path(config: Dict[str, Any], path: str) -> tuple[Any, bool]:
        current: Any = config
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return None, False
            current = current[part]
        return current, True
