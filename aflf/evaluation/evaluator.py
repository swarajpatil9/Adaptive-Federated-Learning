"""Centralized evaluation interfaces for federated learning."""

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..client.client_utils import get_criterion, get_device
from ..metrics.accuracy import compute_classification_metrics
from ..metrics.loss import compute_average_loss
from .convergence import ConvergenceTracker
from .evaluation_utils import format_round_summary
from .history import RoundMetrics, TrainingHistory
from .metrics_tracker import MetricsTracker


class GlobalEvaluator:
    """Evaluate a global model on a centralized held-out dataset."""

    def __init__(self, test_loader: DataLoader, device: Optional[str] = None):
        self.test_loader = test_loader
        self.device = get_device(device)

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        criterion_name: str = 'cross_entropy',
        include_optional_metrics: bool = False,
    ) -> Dict[str, float]:
        """Evaluate global model and return standardized metric dictionary."""
        criterion = get_criterion(criterion_name)

        was_training = model.training
        model = model.to(self.device)
        model.eval()

        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []

        for data, target in self.test_loader:
            data = data.to(self.device)
            target = target.to(self.device)

            output = model(data)
            loss = criterion(output, target)

            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = output.argmax(dim=1)
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(target.detach().cpu())

        if was_training:
            model.train()

        if total_samples == 0:
            metrics: Dict[str, float] = {
                'global_loss': 0.0,
                'global_accuracy': 0.0,
                'num_eval_samples': 0.0,
            }
            if include_optional_metrics:
                metrics.update({'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0})
            return metrics

        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)
        classification_metrics = compute_classification_metrics(
            predictions=predictions,
            targets=targets,
            include_optional=include_optional_metrics,
        )

        metrics = {
            'global_loss': compute_average_loss(total_loss, total_samples),
            'global_accuracy': float(classification_metrics['accuracy']),
            'num_eval_samples': float(total_samples),
        }

        if include_optional_metrics:
            metrics['precision'] = float(classification_metrics['precision'])
            metrics['recall'] = float(classification_metrics['recall'])
            metrics['f1_score'] = float(classification_metrics['f1_score'])

        return metrics


class EvaluationManager:
    """Phase 7 evaluation coordinator for federated training rounds."""

    def __init__(
        self,
        test_loader: DataLoader,
        device: Optional[str] = None,
        experiment_name: str = 'federated_learning',
        output_dir: str = 'results/metrics',
        include_optional_metrics: bool = True,
        bytes_per_parameter: int = 4,
    ):
        self.global_evaluator = GlobalEvaluator(test_loader=test_loader, device=device)
        self.metrics_tracker = MetricsTracker(bytes_per_parameter=bytes_per_parameter)
        self.history = TrainingHistory(experiment_name=experiment_name)
        self.convergence_tracker = ConvergenceTracker()

        self.include_optional_metrics = include_optional_metrics
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()

    def evaluate_round(
        self,
        round_num: int,
        model: nn.Module,
        server_round: Dict[str, Any],
        round_time: float,
        criterion_name: str = 'cross_entropy',
    ) -> RoundMetrics:
        """Evaluate one round, track history, and return structured metrics."""
        global_metrics = self.global_evaluator.evaluate(
            model=model,
            criterion_name=criterion_name,
            include_optional_metrics=self.include_optional_metrics,
        )

        client_metrics = self.metrics_tracker.collect_client_metrics(
            round_num=round_num,
            results=list(server_round.get('results', [])),
        )

        convergence_metrics = self.convergence_tracker.update(
            round_num=round_num,
            global_accuracy=float(global_metrics.get('global_accuracy', 0.0)),
            global_loss=float(global_metrics.get('global_loss', 0.0)),
        )

        round_metrics = self.metrics_tracker.build_round_metrics(
            round_num=round_num,
            global_metrics=global_metrics,
            server_round=server_round,
            client_metrics=client_metrics,
            convergence_metrics=convergence_metrics,
            model=model,
            total_training_time=time.time() - self.start_time,
            round_time=round_time,
        )

        self.history.add_round_metrics(round_metrics)
        self.history.add_client_metrics(client_metrics)
        return round_metrics

    def round_metrics_dict(self, round_metrics: RoundMetrics) -> Dict[str, Any]:
        """Convert round metrics object into dictionary form."""
        return asdict(round_metrics)

    def format_round_report(self, round_metrics: Any) -> str:
        """Format one compact round-level metrics line for console logs."""
        if isinstance(round_metrics, dict):
            return format_round_summary(round_metrics)
        return format_round_summary(asdict(round_metrics))

    def export_logs(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """Export round/client CSV and JSON experiment logs."""
        target_dir = Path(output_dir) if output_dir else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        base_name = f"{self.experiment_name}_phase7_{timestamp}"

        round_csv = self.history.export_round_csv(str(target_dir / f"{base_name}_rounds.csv"))
        client_csv = self.history.export_client_csv(str(target_dir / f"{base_name}_clients.csv"))
        experiment_json = self.history.export_json(
            str(target_dir / f"{base_name}_experiment.json")
        )

        return {
            'round_csv': round_csv,
            'client_csv': client_csv,
            'experiment_json': experiment_json,
        }
