"""
Client selection strategies for federated learning.

Main API:
    SelectionStrategy - Abstract base class
    RandomSelection - Random client selection
    DataAwareSelection - Data-aware selection
    FairnessSelection - Fairness-based selection

Example:
    >>> from aflf.selection import RandomSelection
    >>>
    >>> strategy = RandomSelection(seed=42)
    >>> selected = strategy.select(
    ...     available_clients=[0, 1, 2, 3, 4],
    ...     num_clients=2,
    ...     round_num=0
    ... )
"""

from .selection_strategy import (
    DataAwareSelection,
    FairnessSelection,
    RandomSelection,
    SelectionStrategy,
)

__all__ = [
    'SelectionStrategy',
    'RandomSelection',
    'DataAwareSelection',
    'FairnessSelection',
]
