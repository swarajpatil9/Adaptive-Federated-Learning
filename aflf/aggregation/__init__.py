"""
Federated aggregation strategies.

Main API:
    AggregationStrategy - Abstract base class
    FedAvg - Baseline weighted averaging strategy

Example:
    >>> from aflf.aggregation import FedAvg
    >>>
    >>> aggregator = FedAvg()
    >>> new_weights = aggregator.aggregate(client_results, global_weights)
"""

from .aggregation_base import AggregationStrategy, FedAvgPlaceholder
from .aggregation_utils import (
    normalize_sample_weights,
    validate_training_results,
    weighted_average_state_dicts,
)
from .fedavg import FedAvg

__all__ = [
    'AggregationStrategy',
    'FedAvg',
    'FedAvgPlaceholder',
    'validate_training_results',
    'normalize_sample_weights',
    'weighted_average_state_dicts',
]
