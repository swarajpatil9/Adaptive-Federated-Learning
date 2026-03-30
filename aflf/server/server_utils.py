"""
Server utility functions for federated learning.

Provides helper functions for:
- Model weight serialization/deserialization
- Client result processing
- Metrics aggregation
- Logging utilities
"""

from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn

from ..client.client import TrainingResult


def get_model_parameters(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    """
    Extract model parameters as OrderedDict.

    Args:
        model: PyTorch model

    Returns:
        OrderedDict mapping parameter names to tensors
    """
    return OrderedDict(
        [(name, param.data.clone()) for name, param in model.state_dict().items()]
    )


def set_model_parameters(
    model: nn.Module, parameters: OrderedDict[str, torch.Tensor]
) -> None:
    """
    Load parameters into model.

    Args:
        model: PyTorch model
        parameters: OrderedDict of parameters
    """
    model.load_state_dict(parameters, strict=True)


def compute_weighted_average_metrics(
    results: List[TrainingResult],
) -> Dict[str, float]:
    """
    Compute weighted average of metrics across clients.

    Metrics are weighted by number of samples per client.

    Args:
        results: List of training results from clients

    Returns:
        Dictionary with averaged metrics
    """
    if not results:
        return {}

    total_samples = sum(r.num_samples for r in results)

    if total_samples == 0:
        return {}

    avg_train_loss = (
        sum(r.train_loss * r.num_samples for r in results) / total_samples
    )
    avg_train_accuracy = (
        sum(r.train_accuracy * r.num_samples for r in results) / total_samples
    )

    # Handle optional validation metrics
    val_results = [r for r in results if r.val_loss is not None]
    if val_results:
        val_samples = sum(r.num_samples for r in val_results)
        avg_val_loss = (
            sum(r.val_loss * r.num_samples for r in val_results) / val_samples
        )
        avg_val_accuracy = (
            sum(r.val_accuracy * r.num_samples for r in val_results) / val_samples
        )
    else:
        avg_val_loss = None
        avg_val_accuracy = None

    avg_training_time = sum(r.training_time for r in results) / len(results)
    avg_privacy_overhead_time = sum(
        float(getattr(r, 'privacy_overhead_time', 0.0)) for r in results
    ) / len(results)
    privacy_enabled_fraction = sum(
        1.0 if bool(getattr(r, 'privacy_enabled', False)) else 0.0
        for r in results
    ) / len(results)

    return {
        'num_clients': len(results),
        'total_samples': total_samples,
        'avg_train_loss': avg_train_loss,
        'avg_train_accuracy': avg_train_accuracy,
        'avg_val_loss': avg_val_loss,
        'avg_val_accuracy': avg_val_accuracy,
        'avg_training_time': avg_training_time,
        'avg_privacy_overhead_time': avg_privacy_overhead_time,
        'privacy_enabled_fraction': privacy_enabled_fraction,
    }


def format_round_summary(round_num: int, metrics: Dict) -> str:
    """
    Format round metrics as readable string.

    Args:
        round_num: Round number
        metrics: Dictionary of metrics

    Returns:
        Formatted string
    """
    lines = [f"Round {round_num} Summary:"]
    lines.append(f"  Clients: {metrics.get('num_clients', 0)}")
    lines.append(f"  Total samples: {metrics.get('total_samples', 0)}")

    if 'avg_train_loss' in metrics:
        lines.append(f"  Avg train loss: {metrics['avg_train_loss']:.4f}")
    if 'avg_train_accuracy' in metrics:
        lines.append(f"  Avg train accuracy: {metrics['avg_train_accuracy']:.4f}")

    if metrics.get('avg_val_loss') is not None:
        lines.append(f"  Avg val loss: {metrics['avg_val_loss']:.4f}")
    if metrics.get('avg_val_accuracy') is not None:
        lines.append(f"  Avg val accuracy: {metrics['avg_val_accuracy']:.4f}")

    if 'avg_training_time' in metrics:
        lines.append(f"  Avg training time: {metrics['avg_training_time']:.2f}s")

    return '\n'.join(lines)
