"""
Abstract base class for federated aggregation strategies.

This module defines the interface that all aggregation strategies must implement.
Aggregation strategies compute the global model from client updates.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List

import torch

from ..client.client import TrainingResult


class AggregationStrategy(ABC):
    """
    Abstract base class for federated aggregation strategies.

    Aggregation strategies take client training results and produce
    updated global model weights. Different strategies implement different
    mathematical operations.

    Common strategies:
    - FedAvg: Weighted average by dataset size
    - FedProx: FedAvg with proximal term
    - FedNova: Variance reduction via normalized averaging
    - FedOpt: Server-side optimization (Adam, etc.)
    - Trimmed Mean: Robust aggregation (outlier removal)
    """

    @abstractmethod
    def aggregate(
        self, results: List[TrainingResult], global_weights: OrderedDict[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        """
        Aggregate client updates into global model.

        This is the core operation of federated learning. It takes training
        results from multiple clients and produces updated global weights.

        Args:
            results: List of training results from participating clients.
                     Each result contains:
                     - client_id: Client identifier
                     - weights: Updated model weights
                     - num_samples: Number of training samples
                     - Additional metrics (loss, accuracy, etc.)
            global_weights: Current global model weights (before aggregation).
                           Some strategies (FedProx) may use this for computation.

        Returns:
            Updated global model weights as OrderedDict

        Example:
            >>> # In Phase 6, this will be implemented as:
            >>> aggregator = FedAvg()
            >>> new_weights = aggregator.aggregate(client_results, global_weights)

        Note:
            Implementations must ensure:
            - Output weights have same keys as input results[i].weights
            - Output weights are on CPU (for general compatibility)
            - Numeric stability (especially for small num_samples)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get strategy name.

        Returns:
            Human-readable strategy name (e.g., "FedAvg", "FedProx")
        """
        pass

class FedAvgPlaceholder(AggregationStrategy):
    """
    Placeholder for FedAvg aggregation.

    This is a stub that will be implemented in Phase 6.
    It exists to allow server code to reference the interface.

    DO NOT USE in production code yet.
    """

    def aggregate(
        self, results: List[TrainingResult], global_weights: OrderedDict[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        """Not implemented yet - Phase 6."""
        raise NotImplementedError(
            "FedAvg aggregation will be implemented in Phase 6. "
            "Server orchestration (Phase 5) is now complete."
        )

    def get_name(self) -> str:
        """Return strategy name."""
        return "FedAvg (Placeholder)"
