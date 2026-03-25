"""
Utilities for aggregation strategies.

Contains reusable validation and weighted averaging helpers used by
FedAvg and future robust aggregation strategies.
"""

from collections import OrderedDict
from typing import Iterable, List

import torch

from ..client.client import TrainingResult


def validate_training_results(results: List[TrainingResult]) -> None:
    """
    Validate that client training results can be aggregated.

    Args:
        results: Client training results for a round

    Raises:
        ValueError: If results are empty or invalid for aggregation
    """
    if not results:
        raise ValueError("Cannot aggregate empty client results")

    total_samples = sum(result.num_samples for result in results)
    if total_samples <= 0:
        raise ValueError("Cannot aggregate when total sample count is zero")

    reference_keys = set(results[0].weights.keys())
    for result in results[1:]:
        if set(result.weights.keys()) != reference_keys:
            raise ValueError(
                "All client updates must have identical parameter keys for aggregation"
            )


def normalize_sample_weights(sample_counts: Iterable[int]) -> List[float]:
    """
    Convert sample counts into normalized weights that sum to 1.

    Args:
        sample_counts: Number of samples used by each client

    Returns:
        List of normalized weights

    Raises:
        ValueError: If the total sample count is zero or negative
    """
    counts = [int(count) for count in sample_counts]
    total_samples = sum(counts)

    if total_samples <= 0:
        raise ValueError("Sample counts must sum to a positive number")

    return [count / total_samples for count in counts]


def weighted_average_state_dicts(
    state_dicts: List[OrderedDict[str, torch.Tensor]],
    sample_counts: List[int],
) -> OrderedDict[str, torch.Tensor]:
    """
    Compute a weighted average over model state dictionaries.

    The averaging weight for each client is proportional to its dataset size.

    Args:
        state_dicts: List of client model parameters
        sample_counts: Sample counts aligned with state_dicts

    Returns:
        Aggregated model parameters
    """
    if not state_dicts:
        raise ValueError("state_dicts must contain at least one client update")

    if len(state_dicts) != len(sample_counts):
        raise ValueError("state_dicts and sample_counts must have the same length")

    normalized_weights = normalize_sample_weights(sample_counts)

    aggregated = OrderedDict()
    reference = state_dicts[0]

    for param_name in reference.keys():
        reference_tensor = reference[param_name]

        if torch.is_floating_point(reference_tensor):
            accumulator = torch.zeros_like(reference_tensor, dtype=torch.float64)
            for client_state, weight in zip(state_dicts, normalized_weights):
                accumulator += client_state[param_name].to(torch.float64) * weight
            aggregated[param_name] = accumulator.to(reference_tensor.dtype)
        else:
            accumulator = torch.zeros_like(reference_tensor, dtype=torch.float64)
            for client_state, weight in zip(state_dicts, normalized_weights):
                accumulator += client_state[param_name].to(torch.float64) * weight
            aggregated[param_name] = torch.round(accumulator).to(reference_tensor.dtype)

    return aggregated
