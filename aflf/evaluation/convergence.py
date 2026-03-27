"""Convergence tracking for federated learning rounds."""

from typing import Dict, List


class ConvergenceTracker:
    """Track convergence speed signals from global accuracy and loss."""

    def __init__(
        self,
        window_size: int = 5,
        accuracy_delta_threshold: float = 0.001,
        loss_delta_threshold: float = 0.001,
    ):
        self.window_size = max(2, int(window_size))
        self.accuracy_delta_threshold = float(accuracy_delta_threshold)
        self.loss_delta_threshold = float(loss_delta_threshold)

        self.rounds: List[int] = []
        self.accuracies: List[float] = []
        self.losses: List[float] = []

    def update(self, round_num: int, global_accuracy: float, global_loss: float) -> Dict[str, float]:
        """Update tracker with one new round and return convergence statistics."""
        self.rounds.append(int(round_num))
        self.accuracies.append(float(global_accuracy))
        self.losses.append(float(global_loss))

        return {
            'accuracy_improvement_rate': self.accuracy_improvement_rate(),
            'loss_decrease_rate': self.loss_decrease_rate(),
            'rounds_to_convergence_estimate': self.rounds_to_convergence_estimate(),
            'has_converged': float(self.has_converged()),
        }

    def _recent_pair(self, values: List[float]) -> List[float]:
        if len(values) < 2:
            return []
        window = values[-self.window_size :]
        if len(window) < 2:
            return []
        return window

    def accuracy_improvement_rate(self) -> float:
        """Estimate per-round accuracy improvement over recent window."""
        window = self._recent_pair(self.accuracies)
        if not window:
            return 0.0
        return (window[-1] - window[0]) / float(len(window) - 1)

    def loss_decrease_rate(self) -> float:
        """Estimate per-round loss decrease over recent window."""
        window = self._recent_pair(self.losses)
        if not window:
            return 0.0
        return (window[0] - window[-1]) / float(len(window) - 1)

    def rounds_to_convergence_estimate(self) -> float:
        """
        Rough estimate of rounds until convergence by recent slope.

        A low recent slope indicates near convergence and returns 0.
        """
        if len(self.accuracies) < self.window_size:
            return float('inf')

        slope = self.accuracy_improvement_rate()
        if slope <= self.accuracy_delta_threshold:
            return 0.0

        # Target is to reach a very small slope near the configured threshold.
        return max(1.0, slope / self.accuracy_delta_threshold)

    def has_converged(self) -> bool:
        """Check convergence from both accuracy and loss recent-rate criteria."""
        if len(self.accuracies) < self.window_size or len(self.losses) < self.window_size:
            return False

        return (
            abs(self.accuracy_improvement_rate()) <= self.accuracy_delta_threshold
            and abs(self.loss_decrease_rate()) <= self.loss_delta_threshold
        )
