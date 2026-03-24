"""
Federated aggregation strategies.

Main API:
    AggregationStrategy - Abstract base class

NOTE: Phase 5 only defines interfaces.
Concrete implementations (FedAvg, etc.) will be added in Phase 6.

Example (Phase 6):
    >>> from aflf.aggregation import FedAvg
    >>>
    >>> aggregator = FedAvg()
    >>> new_weights = aggregator.aggregate(client_results, global_weights)
"""

from .aggregation_base import AggregationStrategy, FedAvgPlaceholder

__all__ = [
    'AggregationStrategy',
    'FedAvgPlaceholder',
]
