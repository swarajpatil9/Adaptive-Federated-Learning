"""Utility functions for evaluation and metric formatting."""

from typing import Iterable


def safe_mean(values: Iterable[float]) -> float:
    """Compute mean with empty-input safety."""
    sequence = [float(value) for value in values]
    if not sequence:
        return 0.0
    return sum(sequence) / len(sequence)


def safe_variance(values: Iterable[float]) -> float:
    """Compute population variance with empty-input safety."""
    sequence = [float(value) for value in values]
    if not sequence:
        return 0.0
    mean_value = safe_mean(sequence)
    return sum((value - mean_value) ** 2 for value in sequence) / len(sequence)


def format_megabytes(num_bytes: int) -> float:
    """Convert bytes to megabytes using binary megabytes."""
    if num_bytes <= 0:
        return 0.0
    return num_bytes / float(1024 * 1024)


def format_round_summary(round_metrics: dict) -> str:
    """Generate a compact single-line round summary for console logging."""
    privacy_enabled_fraction = round_metrics.get('privacy_enabled_fraction', 0.0)
    privacy_noise = round_metrics.get('privacy_noise_scale_mean', 0.0)
    privacy_overhead = round_metrics.get('privacy_overhead_time_mean', 0.0)
    communication_reduction = round_metrics.get('communication_reduction_percentage', 0.0)
    communication_precision = round_metrics.get('communication_precision_mode', 'float32')
    return (
        f"[Round {round_metrics.get('round_num', 0):03d}] "
        f"global_acc={round_metrics.get('global_accuracy', 0.0):.4f} "
        f"global_loss={round_metrics.get('global_loss', 0.0):.4f} "
        f"clients={round_metrics.get('num_participating_clients', 0)}/"
        f"{round_metrics.get('num_selected_clients', 0)} "
        f"round_time={round_metrics.get('round_time', 0.0):.2f}s "
        f"comm={round_metrics.get('communication_cost_mb', 0.0):.2f}MB "
        f"comm_reduction={communication_reduction:.2f}% "
        f"comm_precision={communication_precision} "
        f"privacy={privacy_enabled_fraction:.2f} "
        f"noise={privacy_noise:.4f} "
        f"privacy_overhead={privacy_overhead:.4f}s"
    )
