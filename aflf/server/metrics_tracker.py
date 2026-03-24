"""
Metrics tracking and export for federated learning.

Tracks metrics across rounds and exports to:
- JSON (structured data)
- CSV (spreadsheet format)
- TensorBoard (visualization)
"""

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from .round_manager import RoundState


class MetricsTracker:
    """
    Tracks and exports federated learning metrics.

    Accumulates metrics across rounds and provides export
    functionality for analysis and visualization.

    Features:
    - Round-level metrics tracking
    - Client-level statistics
    - JSON/CSV export
    - TensorBoard integration (optional)

    Example:
        >>> tracker = MetricsTracker(
        ...     experiment_name="mnist_fedavg",
        ...     output_dir="results"
        ... )
        >>> tracker.record_round(round_state, metrics)
        >>> tracker.export_json()
        >>> tracker.export_csv()
    """

    def __init__(
        self,
        experiment_name: str = "federated_learning",
        output_dir: str = "results",
        enable_tensorboard: bool = False,
        tensorboard_dir: Optional[str] = None,
    ):
        """
        Initialize metrics tracker.

        Args:
            experiment_name: Name of experiment
            output_dir: Directory for output files
            enable_tensorboard: Whether to log to TensorBoard
            tensorboard_dir: TensorBoard log directory (defaults to output_dir/tensorboard)
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metrics storage
        self.round_metrics: List[Dict] = []
        self.client_metrics: Dict[int, List[Dict]] = {}
        self.summary_stats: Dict = {}

        # Timestamps
        self.start_time = time.time()
        self.round_times: List[float] = []

        # TensorBoard
        self.enable_tensorboard = enable_tensorboard
        self.tensorboard_writer = None

        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                if tensorboard_dir is None:
                    tensorboard_dir = self.output_dir / "tensorboard" / experiment_name

                self.tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_dir))
                print(f"TensorBoard logging enabled: {tensorboard_dir}")
            except ImportError:
                print(
                    "Warning: TensorBoard not available. Install with: pip install tensorboard"
                )
                self.enable_tensorboard = False

    def record_round(
        self, round_state: RoundState, metrics: Dict, additional_data: Optional[Dict] = None
    ) -> None:
        """
        Record metrics for a round.

        Args:
            round_state: Round state object
            metrics: Dictionary of aggregated metrics
            additional_data: Additional data to record (optional)
        """
        round_num = round_state.round_num

        # Compile round metrics
        round_data = {
            'round': round_num,
            'timestamp': time.time(),
            'elapsed_time': time.time() - self.start_time,
            'round_duration': round_state.duration,
            'num_selected': len(round_state.selected_clients),
            'num_participating': len(round_state.participating_clients),
            'num_dropped': len(round_state.dropped_clients),
            'participation_rate': round_state.participation_rate,
            'failure_rate': round_state.failure_rate,
        }

        # Add metrics
        round_data.update(metrics)

        # Add additional data
        if additional_data:
            round_data.update(additional_data)

        self.round_metrics.append(round_data)
        self.round_times.append(round_state.duration)

        # Log to TensorBoard
        if self.enable_tensorboard and self.tensorboard_writer:
            self._log_to_tensorboard(round_num, round_data)

    def record_client_result(
        self, round_num: int, client_id: int, result_dict: Dict
    ) -> None:
        """
        Record individual client result.

        Args:
            round_num: Round number
            client_id: Client ID
            result_dict: Client result dictionary
        """
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = []

        client_data = {
            'round': round_num,
            'client_id': client_id,
            'timestamp': time.time(),
        }
        client_data.update(result_dict)

        self.client_metrics[client_id].append(client_data)

    def compute_summary(self) -> Dict:
        """
        Compute summary statistics across all rounds.

        Returns:
            Dictionary with summary statistics
        """
        if not self.round_metrics:
            return {}

        num_rounds = len(self.round_metrics)
        total_time = time.time() - self.start_time

        summary = {
            'experiment_name': self.experiment_name,
            'num_rounds': num_rounds,
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'avg_round_time': sum(self.round_times) / len(self.round_times),
            'min_round_time': min(self.round_times),
            'max_round_time': max(self.round_times),
        }

        # Average metrics across rounds
        metric_keys = [
            'avg_train_loss',
            'avg_train_accuracy',
            'participation_rate',
            'failure_rate',
        ]

        for key in metric_keys:
            values = [r[key] for r in self.round_metrics if key in r]
            if values:
                summary[f'{key}_mean'] = sum(values) / len(values)
                summary[f'{key}_min'] = min(values)
                summary[f'{key}_max'] = max(values)

        # Final round metrics
        if self.round_metrics:
            last_round = self.round_metrics[-1]
            summary['final_train_loss'] = last_round.get('avg_train_loss', None)
            summary['final_train_accuracy'] = last_round.get('avg_train_accuracy', None)

        self.summary_stats = summary
        return summary

    def export_json(self, filename: Optional[str] = None) -> str:
        """
        Export metrics to JSON file.

        Args:
            filename: Output filename (default: {experiment_name}_metrics.json)

        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_name}_metrics_{timestamp}.json"

        filepath = self.output_dir / filename

        # Compute summary if not already done
        if not self.summary_stats:
            self.compute_summary()

        export_data = {
            'experiment_name': self.experiment_name,
            'summary': self.summary_stats,
            'round_metrics': self.round_metrics,
            'client_metrics': self.client_metrics,
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Exported metrics to: {filepath}")
        return str(filepath)

    def export_csv(self, filename: Optional[str] = None) -> str:
        """
        Export round metrics to CSV file.

        Args:
            filename: Output filename (default: {experiment_name}_rounds.csv)

        Returns:
            Path to exported file
        """
        if not self.round_metrics:
            print("No metrics to export")
            return ""

        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_name}_rounds_{timestamp}.csv"

        filepath = self.output_dir / filename

        # Get all unique keys across all rounds
        all_keys = set()
        for metrics in self.round_metrics:
            all_keys.update(metrics.keys())

        # Sort keys for consistent column order
        fieldnames = sorted(all_keys)

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.round_metrics)

        print(f"Exported CSV to: {filepath}")
        return str(filepath)

    def export_client_csv(self, filename: Optional[str] = None) -> str:
        """
        Export client-level metrics to CSV file.

        Args:
            filename: Output filename (default: {experiment_name}_clients.csv)

        Returns:
            Path to exported file
        """
        if not self.client_metrics:
            print("No client metrics to export")
            return ""

        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_name}_clients_{timestamp}.csv"

        filepath = self.output_dir / filename

        # Flatten client metrics
        all_client_data = []
        for client_id, metrics_list in self.client_metrics.items():
            all_client_data.extend(metrics_list)

        if not all_client_data:
            return ""

        # Get all keys
        all_keys = set()
        for data in all_client_data:
            all_keys.update(data.keys())

        fieldnames = sorted(all_keys)

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_client_data)

        print(f"Exported client CSV to: {filepath}")
        return str(filepath)

    def _log_to_tensorboard(self, round_num: int, metrics: Dict) -> None:
        """
        Log metrics to TensorBoard.

        Args:
            round_num: Round number
            metrics: Metrics dictionary
        """
        if not self.tensorboard_writer:
            return

        # Log scalar metrics
        scalar_metrics = {
            'Loss/train': 'avg_train_loss',
            'Accuracy/train': 'avg_train_accuracy',
            'Loss/val': 'avg_val_loss',
            'Accuracy/val': 'avg_val_accuracy',
            'Participation/rate': 'participation_rate',
            'Participation/num_clients': 'num_participating',
            'Failure/rate': 'failure_rate',
            'Timing/round_duration': 'round_duration',
        }

        for tag, key in scalar_metrics.items():
            if key in metrics and metrics[key] is not None:
                self.tensorboard_writer.add_scalar(tag, metrics[key], round_num)

        # Flush to disk
        self.tensorboard_writer.flush()

    def close(self) -> None:
        """Close TensorBoard writer if open."""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class ProgressTracker:
    """
    Tracks training progress and computes convergence metrics.

    Useful for determining when to stop training or adjust hyperparameters.

    Example:
        >>> tracker = ProgressTracker()
        >>> for round_num in range(100):
        ...     tracker.update(loss=0.5, accuracy=0.85)
        ...     if tracker.has_converged():
        ...         break
    """

    def __init__(
        self,
        convergence_window: int = 5,
        convergence_threshold: float = 0.001,
    ):
        """
        Initialize progress tracker.

        Args:
            convergence_window: Number of rounds to check for convergence
            convergence_threshold: Threshold for metric change (convergence if < threshold)
        """
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold

        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = []

    def update(self, loss: float, accuracy: float) -> None:
        """
        Update with new metrics.

        Args:
            loss: Current loss
            accuracy: Current accuracy
        """
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)

    def has_converged(self, metric: str = 'loss') -> bool:
        """
        Check if training has converged.

        Convergence is detected if the metric has changed by less than
        threshold over the last window rounds.

        Args:
            metric: Metric to check ('loss' or 'accuracy')

        Returns:
            True if converged, False otherwise
        """
        history = self.loss_history if metric == 'loss' else self.accuracy_history

        if len(history) < self.convergence_window:
            return False

        # Get last window values
        recent = history[-self.convergence_window :]

        # Check if variance is below threshold
        mean_val = sum(recent) / len(recent)
        variance = sum((x - mean_val) ** 2 for x in recent) / len(recent)
        std_dev = variance**0.5

        return std_dev < self.convergence_threshold

    def get_improvement(self, window: int = 5) -> Dict[str, float]:
        """
        Get improvement over last N rounds.

        Args:
            window: Number of rounds to look back

        Returns:
            Dictionary with improvement deltas
        """
        if len(self.loss_history) < window:
            return {'loss_improvement': 0.0, 'accuracy_improvement': 0.0}

        loss_improvement = self.loss_history[-window] - self.loss_history[-1]
        accuracy_improvement = self.accuracy_history[-1] - self.accuracy_history[-window]

        return {
            'loss_improvement': loss_improvement,
            'accuracy_improvement': accuracy_improvement,
        }

    def get_best(self) -> Dict[str, float]:
        """
        Get best metrics seen so far.

        Returns:
            Dictionary with best loss and accuracy
        """
        if not self.loss_history:
            return {'best_loss': float('inf'), 'best_accuracy': 0.0}

        return {
            'best_loss': min(self.loss_history),
            'best_accuracy': max(self.accuracy_history),
        }
