"""Centralized metric tracking and round-statistics collector."""

from dataclasses import asdict
from typing import Any, Dict, List

import torch.nn as nn

from ..metrics.communication import (
    estimate_model_size_bytes,
    estimate_round_communication_bytes,
)
from ..metrics.timing import summarize_client_training_time
from .evaluation_utils import safe_mean, safe_variance
from .history import ClientMetrics, RoundMetrics


class MetricsTracker:
    """
    Central metrics framework for federated training evaluation.

    Responsibilities:
    - Collect per-client metrics from round results
    - Compute client distribution statistics (mean and variance)
    - Estimate communication costs from model parameter count
    - Build structured round-level metric objects
    """

    def __init__(self, bytes_per_parameter: int = 4):
        self.bytes_per_parameter = int(bytes_per_parameter)

    def collect_client_metrics(self, round_num: int, results: List[Any]) -> List[ClientMetrics]:
        """Extract client-level metrics from client training result objects."""
        metrics: List[ClientMetrics] = []
        for result in results:
            privacy_metadata = getattr(result, 'privacy_metadata', {}) or {}
            metrics.append(
                ClientMetrics(
                    round_num=int(round_num),
                    client_id=int(getattr(result, 'client_id', -1)),
                    train_accuracy=float(getattr(result, 'train_accuracy', 0.0)),
                    train_loss=float(getattr(result, 'train_loss', 0.0)),
                    num_samples=int(getattr(result, 'num_samples', 0)),
                    training_time=float(getattr(result, 'training_time', 0.0)),
                    val_accuracy=getattr(result, 'val_accuracy', None),
                    val_loss=getattr(result, 'val_loss', None),
                    privacy_enabled=bool(getattr(result, 'privacy_enabled', False)),
                    privacy_overhead_time=float(
                        getattr(result, 'privacy_overhead_time', 0.0)
                    ),
                    clip_applied=bool(privacy_metadata.get('clip_applied', False)),
                    clip_factor=float(privacy_metadata.get('clip_factor', 1.0)),
                    noise_scale=float(privacy_metadata.get('noise_std', 0.0)),
                )
            )
        return metrics

    def build_round_metrics(
        self,
        round_num: int,
        global_metrics: Dict[str, float],
        server_round: Dict[str, Any],
        client_metrics: List[ClientMetrics],
        convergence_metrics: Dict[str, float],
        model: nn.Module,
        total_training_time: float,
        round_time: float,
    ) -> RoundMetrics:
        """Create structured round metrics from all metric sub-components."""
        num_participating = int(server_round.get('num_participating', len(client_metrics)))

        client_accuracies = [item.train_accuracy for item in client_metrics]
        client_times = [item.training_time for item in client_metrics]
        privacy_enabled_flags = [item.privacy_enabled for item in client_metrics]
        privacy_overheads = [item.privacy_overhead_time for item in client_metrics]
        privacy_noise_scales = [item.noise_scale for item in client_metrics]
        privacy_clip_flags = [item.clip_applied for item in client_metrics]

        model_size_bytes = estimate_model_size_bytes(
            model=model,
            bytes_per_parameter=self.bytes_per_parameter,
        )
        communication_bytes, communication_mb = estimate_round_communication_bytes(
            model_size_bytes=model_size_bytes,
            num_participating_clients=num_participating,
            include_downlink=True,
            include_uplink=True,
        )

        timing_summary = summarize_client_training_time(client_times)

        privacy_enabled_fraction = safe_mean([1.0 if flag else 0.0 for flag in privacy_enabled_flags])
        privacy_clip_applied_fraction = safe_mean(
            [1.0 if flag else 0.0 for flag in privacy_clip_flags]
        )
        privacy_overhead_time_mean = safe_mean(privacy_overheads)
        privacy_overhead_time_total = sum(privacy_overheads)
        privacy_noise_scale_mean = safe_mean(privacy_noise_scales)

        global_accuracy = float(global_metrics.get('global_accuracy', 0.0))
        client_accuracy_mean = safe_mean(client_accuracies)
        privacy_accuracy_drop_estimate = max(0.0, client_accuracy_mean - global_accuracy)

        return RoundMetrics(
            round_num=int(round_num),
            global_accuracy=global_accuracy,
            global_loss=float(global_metrics.get('global_loss', 0.0)),
            client_accuracy_mean=client_accuracy_mean,
            client_accuracy_variance=safe_variance(client_accuracies),
            round_time=float(round_time),
            total_training_time=float(total_training_time),
            participation_rate=float(server_round.get('participation_rate', 0.0)),
            num_selected_clients=int(server_round.get('num_selected', 0)),
            num_participating_clients=num_participating,
            model_size_bytes=model_size_bytes,
            communication_cost_bytes=communication_bytes,
            communication_cost_mb=communication_mb,
            client_training_time_mean=float(
                timing_summary.get('client_training_time_mean', 0.0)
            ),
            client_training_time_min=float(
                timing_summary.get('client_training_time_min', 0.0)
            ),
            client_training_time_max=float(
                timing_summary.get('client_training_time_max', 0.0)
            ),
            accuracy_improvement_rate=float(
                convergence_metrics.get('accuracy_improvement_rate', 0.0)
            ),
            loss_decrease_rate=float(convergence_metrics.get('loss_decrease_rate', 0.0)),
            rounds_to_convergence_estimate=float(
                convergence_metrics.get('rounds_to_convergence_estimate', float('inf'))
            ),
            privacy_enabled_fraction=float(privacy_enabled_fraction),
            privacy_overhead_time_mean=float(privacy_overhead_time_mean),
            privacy_overhead_time_total=float(privacy_overhead_time_total),
            privacy_noise_scale_mean=float(privacy_noise_scale_mean),
            privacy_clip_applied_fraction=float(privacy_clip_applied_fraction),
            privacy_accuracy_drop_estimate=float(privacy_accuracy_drop_estimate),
            precision=global_metrics.get('precision'),
            recall=global_metrics.get('recall'),
            f1_score=global_metrics.get('f1_score'),
        )

    @staticmethod
    def to_dict(round_metrics: RoundMetrics) -> Dict[str, Any]:
        """Serialize round metrics dataclass as dictionary."""
        metrics_dict = asdict(round_metrics)
        return metrics_dict
