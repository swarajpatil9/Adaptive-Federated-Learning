"""Convergence monitoring based on smoothed loss and accuracy trends."""

from typing import Dict

from .optimization_config import OptimizationConfig
from .optimization_utils import make_window, moving_average, stddev


class ConvergenceMonitor:
    """Track convergence health from recent round-level metrics."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.loss_window = make_window(config.window_size)
        self.accuracy_window = make_window(config.window_size)
        self._prev_loss_ma = None
        self._prev_accuracy_ma = None

    def update(self, global_loss: float, global_accuracy: float) -> Dict[str, float]:
        """Update windows and produce smoothed convergence diagnostics."""
        self.loss_window.append(float(global_loss))
        self.accuracy_window.append(float(global_accuracy))

        loss_ma = moving_average(self.loss_window)
        accuracy_ma = moving_average(self.accuracy_window)

        if self._prev_loss_ma is None:
            loss_improvement = 0.0
        else:
            loss_improvement = self._prev_loss_ma - loss_ma

        if self._prev_accuracy_ma is None:
            accuracy_improvement = 0.0
        else:
            accuracy_improvement = accuracy_ma - self._prev_accuracy_ma

        self._prev_loss_ma = loss_ma
        self._prev_accuracy_ma = accuracy_ma

        loss_stability_std = stddev(self.loss_window)
        is_unstable = loss_stability_std > self.config.instability_std_threshold
        is_plateau = (
            abs(loss_improvement) <= self.config.plateau_threshold
            and abs(accuracy_improvement) <= self.config.accuracy_improvement_threshold
        )

        return {
            "loss_ma": float(loss_ma),
            "accuracy_ma": float(accuracy_ma),
            "loss_improvement": float(loss_improvement),
            "accuracy_improvement": float(accuracy_improvement),
            "loss_stability_std": float(loss_stability_std),
            "is_unstable": float(1.0 if is_unstable else 0.0),
            "is_plateau": float(1.0 if is_plateau else 0.0),
            "window_size": float(len(self.loss_window)),
        }
