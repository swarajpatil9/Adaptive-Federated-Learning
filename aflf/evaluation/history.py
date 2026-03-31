"""Training history data structures and export helpers."""

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ClientMetrics:
    """Client-level metrics captured for one training round."""

    round_num: int
    client_id: int
    train_accuracy: float
    train_loss: float
    num_samples: int
    training_time: float
    val_accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    privacy_enabled: bool = False
    privacy_overhead_time: float = 0.0
    communication_original_bytes: float = 0.0
    communication_compressed_bytes: float = 0.0
    communication_reduction_percentage: float = 0.0
    communication_precision: str = 'float32'
    communication_sparsification_enabled: bool = False
    communication_sparsity_ratio: float = 0.0
    clip_applied: bool = False
    clip_factor: float = 1.0
    noise_scale: float = 0.0


@dataclass
class RoundMetrics:
    """Structured round-level metric bundle used across evaluation outputs."""

    round_num: int
    global_accuracy: float
    global_loss: float
    client_accuracy_mean: float
    client_accuracy_variance: float
    round_time: float
    total_training_time: float
    participation_rate: float
    num_selected_clients: int
    num_participating_clients: int
    model_size_bytes: int
    communication_cost_bytes: int
    communication_cost_mb: float
    client_training_time_mean: float
    client_training_time_min: float
    client_training_time_max: float
    accuracy_improvement_rate: float
    loss_decrease_rate: float
    rounds_to_convergence_estimate: float
    communication_compressed_cost_bytes: float = 0.0
    communication_saved_bytes: float = 0.0
    communication_reduction_percentage: float = 0.0
    communication_precision_mode: str = 'float32'
    communication_sparsification_enabled: float = 0.0
    learning_rate: float = 0.0
    lr_change_ratio: float = 1.0
    lr_adjustment_reason: str = 'static'
    privacy_enabled_fraction: float = 0.0
    privacy_overhead_time_mean: float = 0.0
    privacy_overhead_time_total: float = 0.0
    privacy_noise_scale_mean: float = 0.0
    privacy_clip_applied_fraction: float = 0.0
    privacy_accuracy_drop_estimate: float = 0.0
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None


class TrainingHistory:
    """Container for round-level and client-level training history."""

    def __init__(self, experiment_name: str = 'federated_learning'):
        self.experiment_name = experiment_name
        self.start_time = time.time()
        self.round_metrics: List[RoundMetrics] = []
        self.client_metrics: List[ClientMetrics] = []

    def add_round_metrics(self, metrics: RoundMetrics) -> None:
        """Append one round metrics record."""
        self.round_metrics.append(metrics)

    def add_client_metrics(self, metrics: List[ClientMetrics]) -> None:
        """Append per-client records for one round."""
        self.client_metrics.extend(metrics)

    def to_dict(self) -> Dict[str, object]:
        """Export complete history as a JSON-serializable dictionary."""
        return {
            'experiment_name': self.experiment_name,
            'round_metrics': [asdict(entry) for entry in self.round_metrics],
            'client_metrics': [asdict(entry) for entry in self.client_metrics],
            'summary': self.summary(),
        }

    def summary(self) -> Dict[str, float]:
        """Compute aggregate summary statistics for completed rounds."""
        if not self.round_metrics:
            return {
                'num_rounds': 0,
                'total_training_time': 0.0,
            }

        total_time = time.time() - self.start_time
        last_round = self.round_metrics[-1]

        return {
            'num_rounds': len(self.round_metrics),
            'total_training_time': total_time,
            'avg_round_time': sum(record.round_time for record in self.round_metrics)
            / len(self.round_metrics),
            'final_global_accuracy': last_round.global_accuracy,
            'final_global_loss': last_round.global_loss,
            'final_participation_rate': last_round.participation_rate,
            'total_communication_mb': sum(
                record.communication_cost_mb for record in self.round_metrics
            ),
        }

    def export_json(self, output_path: str) -> str:
        """Write full experiment history to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return str(path)

    def export_round_csv(self, output_path: str) -> str:
        """Write round-level metrics to CSV file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        row_dicts = [asdict(entry) for entry in self.round_metrics]
        if not row_dicts:
            with open(path, 'w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['round_num'])
            return str(path)

        fields = sorted({key for row in row_dicts for key in row.keys()})
        with open(path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row_dicts)
        return str(path)

    def export_client_csv(self, output_path: str) -> str:
        """Write client-level metrics to CSV file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        row_dicts = [asdict(entry) for entry in self.client_metrics]
        if not row_dicts:
            with open(path, 'w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['round_num', 'client_id'])
            return str(path)

        fields = sorted({key for row in row_dicts for key in row.keys()})
        with open(path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row_dicts)
        return str(path)
