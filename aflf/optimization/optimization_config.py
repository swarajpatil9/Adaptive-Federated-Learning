"""Configuration primitives for adaptive optimization controls."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class OptimizationConfig:
    """Configuration for adaptive learning-rate behavior."""

    enabled: bool = False
    initial_lr: float = 0.01
    min_lr: float = 1e-4
    max_lr: float = 1.0
    decay_factor: float = 0.95
    slow_decay_factor: float = 0.98
    plateau_decay_factor: float = 0.90
    smoothing_factor: float = 0.2
    min_decay_per_round: float = 0.0
    max_decay_per_round: float = 0.20
    loss_improvement_threshold: float = 0.01
    plateau_threshold: float = 0.01
    accuracy_improvement_threshold: float = 0.001
    instability_std_threshold: float = 0.02
    window_size: int = 3

    @classmethod
    def from_dict(
        cls,
        optimization_config: Dict[str, Any],
        fallback_learning_rate: float,
    ) -> "OptimizationConfig":
        """Build config from parsed YAML sections."""
        adaptive = (optimization_config or {}).get("adaptive_lr", {})
        initial_lr = float(adaptive.get("initial_lr", fallback_learning_rate))
        min_lr = float(adaptive.get("min_lr", 1e-4))
        max_lr = float(adaptive.get("max_lr", max(initial_lr, 1.0)))

        return cls(
            enabled=bool(adaptive.get("enabled", False)),
            initial_lr=initial_lr,
            min_lr=min_lr,
            max_lr=max_lr,
            decay_factor=float(adaptive.get("decay_factor", 0.95)),
            slow_decay_factor=float(adaptive.get("slow_decay_factor", 0.98)),
            plateau_decay_factor=float(adaptive.get("plateau_decay_factor", 0.90)),
            smoothing_factor=float(adaptive.get("smoothing_factor", 0.2)),
            min_decay_per_round=float(adaptive.get("min_decay_per_round", 0.0)),
            max_decay_per_round=float(adaptive.get("max_decay_per_round", 0.20)),
            loss_improvement_threshold=float(
                adaptive.get("loss_improvement_threshold", 0.01)
            ),
            plateau_threshold=float(adaptive.get("plateau_threshold", 0.01)),
            accuracy_improvement_threshold=float(
                adaptive.get("accuracy_improvement_threshold", 0.001)
            ),
            instability_std_threshold=float(
                adaptive.get("instability_std_threshold", 0.02)
            ),
            window_size=int(adaptive.get("window_size", 3)),
        )
