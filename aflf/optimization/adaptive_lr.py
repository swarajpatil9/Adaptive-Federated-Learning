"""Adaptive learning-rate controller for federated round orchestration."""

from typing import Any, Dict, List

from .convergence_monitor import ConvergenceMonitor
from .lr_scheduler import AdaptiveDecayLRScheduler
from .optimization_config import OptimizationConfig


class AdaptiveLRController:
    """Round-wise LR controller backed by convergence-aware policy."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.monitor = ConvergenceMonitor(config)
        self.scheduler = AdaptiveDecayLRScheduler(config)

        self.current_lr = float(config.initial_lr)
        self._active_reason = "initial"
        self._active_change_ratio = 1.0
        self.history: List[Dict[str, Any]] = []

    @classmethod
    def from_dict(
        cls,
        optimization_config: Dict[str, Any],
        fallback_learning_rate: float,
    ) -> "AdaptiveLRController":
        """Build controller from YAML configuration sections."""
        config = OptimizationConfig.from_dict(
            optimization_config=optimization_config,
            fallback_learning_rate=fallback_learning_rate,
        )
        return cls(config=config)

    def is_enabled(self) -> bool:
        """Return whether adaptive behavior is enabled."""
        return bool(self.config.enabled)

    def get_round_context(self, round_num: int) -> Dict[str, Any]:
        """Return optimization metadata associated with current round LR."""
        return {
            "round_num": int(round_num),
            "learning_rate": float(self.current_lr),
            "lr_adjustment_reason": self._active_reason,
            "lr_change_ratio": float(self._active_change_ratio),
        }

    def update_after_round(
        self,
        round_num: int,
        global_loss: float,
        global_accuracy: float,
    ) -> Dict[str, Any]:
        """Ingest evaluated round metrics and compute LR for next round."""
        if not self.is_enabled():
            decision = {
                "round_num": int(round_num),
                "previous_lr": float(self.current_lr),
                "new_lr": float(self.current_lr),
                "lr_change_ratio": 1.0,
                "reason": "adaptive_disabled",
                "monitor": {},
            }
            self.history.append(decision)
            return decision

        monitor_state = self.monitor.update(
            global_loss=float(global_loss),
            global_accuracy=float(global_accuracy),
        )
        previous_lr = float(self.current_lr)
        new_lr, reason = self.scheduler.compute_next_lr(
            current_lr=previous_lr,
            monitor_state=monitor_state,
        )

        self.current_lr = float(new_lr)
        self._active_reason = reason
        self._active_change_ratio = (
            float(new_lr / previous_lr) if previous_lr > 0 else 1.0
        )

        decision = {
            "round_num": int(round_num),
            "previous_lr": previous_lr,
            "new_lr": float(new_lr),
            "lr_change_ratio": float(self._active_change_ratio),
            "reason": reason,
            "monitor": monitor_state,
        }
        self.history.append(decision)
        return decision
