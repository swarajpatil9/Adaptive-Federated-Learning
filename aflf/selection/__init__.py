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
    BaseSelectionStrategy,
    DataAwareSelection,
    FairnessSelection,
    RandomSelection,
    SelectionResult,
    SelectionStrategy,
)
from .dynamic_selection import DynamicSelectionStrategy
from .ranking import ClientRanker
from .scoring import ClientScorer, ScoringWeights
from .selection_utils import SelectionPolicy, SelectionPolicyManager

__all__ = [
    'BaseSelectionStrategy',
    'SelectionStrategy',
    'SelectionResult',
    'RandomSelection',
    'DataAwareSelection',
    'FairnessSelection',
    'DynamicSelectionStrategy',
    'ClientScorer',
    'ScoringWeights',
    'ClientRanker',
    'SelectionPolicy',
    'SelectionPolicyManager',
]
