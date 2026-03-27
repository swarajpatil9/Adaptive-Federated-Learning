"""Timing aggregation helpers for round-level analysis."""

from typing import Dict, Iterable


def summarize_client_training_time(training_times: Iterable[float]) -> Dict[str, float]:
    """Return summary stats for per-client local training durations."""
    values = [float(value) for value in training_times]

    if not values:
        return {
            'client_training_time_mean': 0.0,
            'client_training_time_min': 0.0,
            'client_training_time_max': 0.0,
        }

    return {
        'client_training_time_mean': sum(values) / len(values),
        'client_training_time_min': min(values),
        'client_training_time_max': max(values),
    }
