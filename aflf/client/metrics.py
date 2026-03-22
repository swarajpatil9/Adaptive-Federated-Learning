"""
Metrics tracking for federated learning clients.

Provides lightweight, efficient tracking of training and validation metrics
during local client training.
"""

from typing import Dict, Optional


class MetricsTracker:
    """
    Track training/validation metrics during local training.

    Accumulates batch-level metrics and computes epoch-level averages.
    Designed to be lightweight and efficient for FL clients.

    Example:
        >>> tracker = MetricsTracker()
        >>> # During training
        >>> for batch in train_loader:
        >>>     loss, acc = train_step(batch)
        >>>     tracker.update(loss=loss, accuracy=acc, num_samples=len(batch))
        >>> # Get epoch metrics
        >>> metrics = tracker.get_metrics()
        >>> print(f"Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}")
    """

    def __init__(self):
        """Initialize empty metrics tracker."""
        self.reset()

    def reset(self):
        """Reset all tracked metrics to zero."""
        self._loss_sum = 0.0
        self._correct = 0
        self._total = 0
        self._custom_metrics: Dict[str, float] = {}
        self._custom_counts: Dict[str, int] = {}

    def update(
        self,
        loss: float,
        accuracy: Optional[float] = None,
        num_samples: int = 1,
        **custom_metrics: float,
    ):
        """
        Update metrics with batch results.

        Args:
            loss: Batch loss (will be weighted by num_samples)
            accuracy: Optional batch accuracy (0-1 or 0-100)
            num_samples: Number of samples in batch (for weighted averaging)
            **custom_metrics: Additional metrics to track (e.g., f1_score=0.85)

        Example:
            >>> tracker.update(loss=0.5, accuracy=0.92, num_samples=32)
            >>> tracker.update(loss=0.6, num_samples=16, perplexity=1.82)
        """
        # Track loss (weighted by samples)
        self._loss_sum += loss * num_samples
        self._total += num_samples

        # Track accuracy if provided
        if accuracy is not None:
            # Handle both 0-1 and 0-100 accuracy formats
            if accuracy > 1.0:
                accuracy = accuracy / 100.0
            self._correct += accuracy * num_samples

        # Track custom metrics
        for name, value in custom_metrics.items():
            if name not in self._custom_metrics:
                self._custom_metrics[name] = 0.0
                self._custom_counts[name] = 0
            self._custom_metrics[name] += value * num_samples
            self._custom_counts[name] += num_samples

    def get_metrics(self) -> Dict[str, float]:
        """
        Compute average metrics over all accumulated batches.

        Returns:
            Dictionary with metric names and values:
                - loss: Average loss
                - accuracy: Average accuracy (0-1)
                - num_samples: Total number of samples
                - [custom metrics]: Average values

        Example:
            >>> metrics = tracker.get_metrics()
            >>> print(metrics)
            {'loss': 0.542, 'accuracy': 0.879, 'num_samples': 1024}
        """
        if self._total == 0:
            return {
                'loss': 0.0,
                'accuracy': 0.0,
                'num_samples': 0,
            }

        metrics = {
            'loss': self._loss_sum / self._total,
            'accuracy': self._correct / self._total,
            'num_samples': self._total,
        }

        # Add custom metrics
        for name, value_sum in self._custom_metrics.items():
            count = self._custom_counts[name]
            metrics[name] = value_sum / count if count > 0 else 0.0

        return metrics

    @property
    def num_samples(self) -> int:
        """Get total number of samples tracked."""
        return self._total

    @property
    def loss(self) -> float:
        """Get current average loss."""
        return self._loss_sum / self._total if self._total > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Get current average accuracy."""
        return self._correct / self._total if self._total > 0 else 0.0

    def __repr__(self) -> str:
        """String representation of tracker state."""
        metrics = self.get_metrics()
        return (
            f"MetricsTracker(loss={metrics['loss']:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"samples={metrics['num_samples']})"
        )
