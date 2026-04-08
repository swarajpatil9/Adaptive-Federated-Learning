"""
Utility helpers for the federated training loop.
"""

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class FederatedTrainingConfig:
    """Configuration for baseline federated training."""

    num_rounds: int = 10
    clients_per_round: int = 10
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    evaluation_frequency: int = 1
    optimizer: str = 'sgd'
    momentum: float = 0.0
    weight_decay: float = 0.0
    criterion: str = 'cross_entropy'
    privacy_enabled: bool = False
    clip_norm: float = 1.0
    noise_multiplier: float = 0.05
    device: str = 'cpu'
    seed: int = 42
    verbose: bool = False
    communication: Dict[str, Any] = None
    optimization: Dict[str, Any] = None

    def to_client_train_config(self) -> Dict[str, Any]:
        """Convert to client-local training kwargs."""
        return {
            'epochs': self.local_epochs,
            'lr': self.learning_rate,
            'optimizer': self.optimizer,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'criterion': self.criterion,
            'privacy': {
                'privacy_enabled': self.privacy_enabled,
                'clip_norm': self.clip_norm,
                'noise_multiplier': self.noise_multiplier,
            },
            'communication': self.communication,
        }

    def __post_init__(self) -> None:
        """Initialize optional dict fields safely."""
        if self.communication is None:
            self.communication = {}
        if self.optimization is None:
            self.optimization = {}

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary representation."""
        return asdict(self)


@dataclass
class RoundTrainingRecord:
    """Tracked metrics for one federated round."""

    round_num: int
    global_loss: float
    global_accuracy: float
    avg_train_loss: float
    avg_train_accuracy: float
    selected_clients: int
    participating_clients: int
    failed_clients: int
    participation_rate: float
    round_duration: float


class TrainingProgressTracker:
    """Collect and summarize federated round-level progress."""

    def __init__(self):
        self.records = []
        self.start_time = time.time()

    def add_record(self, record: RoundTrainingRecord) -> None:
        """Add one round record."""
        self.records.append(record)

    def summary(self) -> Dict[str, Any]:
        """Return aggregate summary for completed training."""
        if not self.records:
            return {
                'num_rounds': 0,
                'total_training_time': 0.0,
            }

        total_time = time.time() - self.start_time
        last = self.records[-1]

        return {
            'num_rounds': len(self.records),
            'total_training_time': total_time,
            'avg_round_time': sum(record.round_duration for record in self.records)
            / len(self.records),
            'final_global_loss': last.global_loss,
            'final_global_accuracy': last.global_accuracy,
            'final_avg_train_loss': last.avg_train_loss,
            'final_avg_train_accuracy': last.avg_train_accuracy,
            'avg_participation_rate': sum(
                record.participation_rate for record in self.records
            )
            / len(self.records),
        }


def format_round_log(record: RoundTrainingRecord) -> str:
    """Format one compact console log line for a training round."""
    return (
        f"[Round {record.round_num:03d}] "
        f"global_acc={record.global_accuracy:.4f} "
        f"global_loss={record.global_loss:.4f} "
        f"train_acc={record.avg_train_accuracy:.4f} "
        f"train_loss={record.avg_train_loss:.4f} "
        f"clients={record.participating_clients}/{record.selected_clients} "
        f"failed={record.failed_clients} "
        f"duration={record.round_duration:.2f}s"
    )


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file into a dictionary."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def build_federated_config(config: Dict[str, Any]) -> FederatedTrainingConfig:
    """
    Build FederatedTrainingConfig from raw dictionary config.

    Supports either:
    - flat fields under federated/training/evaluation blocks, or
    - direct fields under training for simple scripts.
    """
    federated = config.get('federated', {})
    training = config.get('training', config)
    evaluation = config.get('evaluation', {})
    privacy = config.get('privacy', {})
    communication = config.get('communication', {})
    optimization = config.get('optimization', {})

    return FederatedTrainingConfig(
        num_rounds=int(federated.get('num_rounds', 10)),
        clients_per_round=int(federated.get('clients_per_round', 10)),
        local_epochs=int(training.get('epochs', training.get('local_epochs', 1))),
        batch_size=int(training.get('batch_size', 32)),
        learning_rate=float(training.get('lr', training.get('learning_rate', 0.01))),
        evaluation_frequency=int(evaluation.get('frequency', 1)),
        optimizer=str(training.get('optimizer', 'sgd')),
        momentum=float(training.get('momentum', 0.0)),
        weight_decay=float(training.get('weight_decay', 0.0)),
        criterion=str(training.get('criterion', 'cross_entropy')),
        privacy_enabled=bool(privacy.get('privacy_enabled', False)),
        clip_norm=float(privacy.get('clip_norm', 1.0)),
        noise_multiplier=float(privacy.get('noise_multiplier', 0.05)),
        device=str(training.get('device', federated.get('device', 'cpu'))),
        seed=int(config.get('seed', 42)),
        verbose=bool(training.get('verbose', False)),
        communication=communication,
        optimization=optimization,
    )


def ensure_dir(path: str) -> None:
    """Create directory path if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
