"""
FedAvg aggregation strategy.

Implements the standard weighted averaging baseline used in most
federated learning research.
"""

from collections import OrderedDict
from typing import List

import torch

from ..client.client import TrainingResult
from .aggregation_base import AggregationStrategy
from .aggregation_utils import validate_training_results, weighted_average_state_dicts


class FedAvg(AggregationStrategy):
    """
    Federated Averaging (FedAvg).

    Given client updates w_k and local sample counts n_k, the global update is:
        w = sum_k (n_k / sum_j n_j) * w_k

    This sample-size weighted baseline is the canonical starting point for
    federated optimization experiments.
    """

    def aggregate(
        self,
        results: List[TrainingResult],
        global_weights: OrderedDict[str, torch.Tensor],
    ) -> OrderedDict[str, torch.Tensor]:
        """
        Aggregate client weights using dataset-size weighted averaging.

        Args:
            results: Client training results from the round
            global_weights: Current global weights (unused by vanilla FedAvg,
                accepted to keep interface consistency)

        Returns:
            New global model weights
        """
        del global_weights

        validate_training_results(results)

        client_weights = [result.weights for result in results]
        client_sample_counts = [result.num_samples for result in results]

        return weighted_average_state_dicts(client_weights, client_sample_counts)

    def get_name(self) -> str:
        """Return strategy name."""
        return "FedAvg"
