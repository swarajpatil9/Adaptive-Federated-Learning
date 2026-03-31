"""Learning-rate scheduler abstractions for adaptive federated training."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

from .optimization_config import OptimizationConfig
from .optimization_utils import bounded


class LRScheduler(ABC):
    """Abstract scheduler used by adaptive LR controller."""

    @abstractmethod
    def compute_next_lr(
        self,
        current_lr: float,
        monitor_state: Dict[str, float],
    ) -> Tuple[float, str]:
        """Return next learning rate and human-readable reason."""


class AdaptiveDecayLRScheduler(LRScheduler):
    """Policy scheduler with plateau and instability-aware decay."""

    def __init__(self, config: OptimizationConfig):
        self.config = config

    def compute_next_lr(
        self,
        current_lr: float,
        monitor_state: Dict[str, float],
    ) -> Tuple[float, str]:
        """Compute adaptive LR update from smoothed trend diagnostics."""
        loss_improvement = float(monitor_state.get("loss_improvement", 0.0))
        is_unstable = bool(monitor_state.get("is_unstable", 0.0))
        is_plateau = bool(monitor_state.get("is_plateau", 0.0))

        if is_plateau:
            decay = self.config.plateau_decay_factor
            reason = "plateau_detected"
        elif is_unstable:
            decay = self.config.decay_factor
            reason = "loss_unstable"
        elif loss_improvement > self.config.loss_improvement_threshold:
            decay = self.config.slow_decay_factor
            reason = "healthy_convergence"
        else:
            decay = self.config.decay_factor
            reason = "slow_convergence"

        proposed = current_lr * decay

        if proposed < current_lr:
            ratio = proposed / current_lr
            ratio = bounded(
                ratio,
                1.0 - self.config.max_decay_per_round,
                1.0 - self.config.min_decay_per_round,
            )
            proposed = current_lr * ratio

        smoothed = (self.config.smoothing_factor * current_lr) + (
            (1.0 - self.config.smoothing_factor) * proposed
        )
        bounded_lr = bounded(smoothed, self.config.min_lr, self.config.max_lr)
        return bounded_lr, reason
