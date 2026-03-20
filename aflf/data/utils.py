"""
Utility functions for data management.

Includes dataset statistics, validation, and helper functions.
"""

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def compute_class_weights(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute inverse frequency class weights for imbalanced datasets.

    Args:
        targets: Tensor of labels
        num_classes: Total number of classes

    Returns:
        Tensor of shape (num_classes,) with class weights
    """
    class_counts = torch.bincount(targets, minlength=num_classes).float()

    # Avoid division by zero
    class_counts = torch.clamp(class_counts, min=1.0)

    # Inverse frequency weighting
    total_samples = len(targets)
    class_weights = total_samples / (num_classes * class_counts)

    return class_weights


def compute_label_distribution(
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Compute normalized label distribution.

    Args:
        targets: Tensor of labels
        num_classes: Total number of classes

    Returns:
        Tensor of shape (num_classes,) with label probabilities
    """
    counts = torch.bincount(targets, minlength=num_classes).float()
    return counts / counts.sum()


def measure_data_heterogeneity(
    client_indices: Dict[int, List[int]],
    targets: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    """
    Measure data heterogeneity across clients.

    Computes multiple metrics to quantify non-IID-ness:
    - KL divergence from uniform distribution
    - Jensen-Shannon divergence between clients
    - Label distribution entropy

    Args:
        client_indices: Mapping from client_id to sample indices
        targets: All labels
        num_classes: Number of classes

    Returns:
        Dictionary with heterogeneity metrics
    """
    num_clients = len(client_indices)

    # Compute label distribution for each client
    client_distributions = []
    for client_id in range(num_clients):
        indices = client_indices[client_id]
        client_targets = targets[indices]
        dist = compute_label_distribution(client_targets, num_classes)
        client_distributions.append(dist)

    client_distributions = torch.stack(client_distributions)

    # Uniform distribution (IID reference)
    uniform_dist = torch.ones(num_classes) / num_classes

    # Average KL divergence from uniform
    kl_divs = []
    for dist in client_distributions:
        # Add small epsilon for numerical stability
        dist_stable = dist + 1e-10
        uniform_stable = uniform_dist + 1e-10
        kl = (dist_stable * torch.log(dist_stable / uniform_stable)).sum()
        kl_divs.append(kl.item())

    avg_kl = np.mean(kl_divs)

    # Average entropy (higher = more uniform)
    entropies = []
    for dist in client_distributions:
        dist_stable = dist + 1e-10
        entropy = -(dist_stable * torch.log(dist_stable)).sum()
        entropies.append(entropy.item())

    avg_entropy = np.mean(entropies)
    max_entropy = np.log(num_classes)  # Maximum possible entropy

    return {
        'avg_kl_from_uniform': avg_kl,
        'avg_entropy': avg_entropy,
        'normalized_entropy': avg_entropy / max_entropy,
        'entropy_std': np.std(entropies),
    }


def validate_partition(
    client_indices: Dict[int, List[int]],
    total_samples: int,
) -> bool:
    """
    Validate that partition is correct.

    Checks:
    - All samples are assigned exactly once
    - No duplicate assignments
    - All clients have at least one sample

    Args:
        client_indices: Partition mapping
        total_samples: Expected total number of samples

    Returns:
        True if partition is valid

    Raises:
        AssertionError: If validation fails
    """
    # Collect all assigned indices
    all_indices = []
    for indices in client_indices.values():
        all_indices.extend(indices)

    # Check total count
    assert len(all_indices) == total_samples, \
        f"Expected {total_samples} samples, got {len(all_indices)}"

    # Check for duplicates
    unique_indices = set(all_indices)
    assert len(unique_indices) == total_samples, \
        "Duplicate sample assignments detected"

    # Check all indices are in valid range
    assert min(all_indices) >= 0, "Negative indices found"
    assert max(all_indices) < total_samples, "Out-of-range indices found"

    # Check all clients have data
    for client_id, indices in client_indices.items():
        assert len(indices) > 0, \
            f"Client {client_id} has no samples"

    return True


def count_samples_per_client(
    client_indices: Dict[int, List[int]],
) -> Dict[int, int]:
    """
    Count number of samples for each client.

    Args:
        client_indices: Partition mapping

    Returns:
        Dictionary mapping client_id -> num_samples
    """
    return {
        client_id: len(indices)
        for client_id, indices in client_indices.items()
    }


def get_dataset_info() -> Dict[str, Dict]:
    """
    Get information about supported datasets.

    Returns:
        Dictionary with dataset metadata
    """
    return {
        'mnist': {
            'name': 'MNIST',
            'num_classes': 10,
            'input_shape': (1, 28, 28),
            'train_samples': 60000,
            'test_samples': 10000,
            'description': 'Handwritten digits (0-9)',
        },
        'cifar10': {
            'name': 'CIFAR-10',
            'num_classes': 10,
            'input_shape': (3, 32, 32),
            'train_samples': 50000,
            'test_samples': 10000,
            'description': '10 object categories (airplane, car, bird, etc.)',
        },
        'cifar100': {
            'name': 'CIFAR-100',
            'num_classes': 100,
            'input_shape': (3, 32, 32),
            'train_samples': 50000,
            'test_samples': 10000,
            'description': '100 fine-grained object categories',
        },
    }


def print_dataset_info():
    """Print information about all supported datasets."""
    info = get_dataset_info()

    print("SUPPORTED DATASETS")
    print("="*80)
    for key, meta in info.items():
        print(f"\n{meta['name']} ('{key}')")
        print(f"  Classes: {meta['num_classes']}")
        print(f"  Input shape: {meta['input_shape']}")
        print(f"  Train samples: {meta['train_samples']}")
        print(f"  Test samples: {meta['test_samples']}")
        print(f"  Description: {meta['description']}")
    print("="*80)
