"""
Visualization utilities for data distributions.

Creates plots and visualizations for:
- Label distribution across clients
- Sample count distributions
- Heterogeneity analysis
"""

from typing import Dict, List, Optional

import numpy as np


def print_distribution_heatmap(
    client_indices: Dict[int, List[int]],
    targets,
    num_classes: int,
    num_clients_to_show: Optional[int] = None,
):
    """
    Print ASCII heatmap of label distribution across clients.

    Args:
        client_indices: Client partition mapping
        targets: All labels
        num_classes: Number of classes
        num_clients_to_show: Number of clients to display (None = all)
    """
    num_clients = len(client_indices)
    if num_clients_to_show is None:
        num_clients_to_show = num_clients

    print("\nLABEL DISTRIBUTION HEATMAP")
    print("="*80)
    print(f"{'Client':>8} | ", end="")
    for c in range(num_classes):
        print(f"{c:>5}", end=" ")
    print(f"| {'Total':>6}")
    print("-"*80)

    # Compute and display distributions
    for client_id in range(min(num_clients, num_clients_to_show)):
        indices = client_indices[client_id]
        client_targets = targets[indices]

        # Count per class
        counts = np.bincount(client_targets.numpy(), minlength=num_classes)

        print(f"{client_id:>8} | ", end="")
        for count in counts:
            if count == 0:
                print("    .", end=" ")
            else:
                print(f"{count:>5}", end=" ")
        print(f"| {len(indices):>6}")

    if num_clients > num_clients_to_show:
        print(f"... ({num_clients - num_clients_to_show} more clients)")

    print("="*80)


def print_sample_count_histogram(
    client_indices: Dict[int, List[int]],
    bins: int = 10,
):
    """
    Print ASCII histogram of samples per client.

    Args:
        client_indices: Client partition mapping
        bins: Number of histogram bins
    """
    sample_counts = [len(indices) for indices in client_indices.values()]

    print("\nSAMPLE COUNT DISTRIBUTION")
    print("="*80)

    min_count = min(sample_counts)
    max_count = max(sample_counts)
    mean_count = np.mean(sample_counts)
    std_count = np.std(sample_counts)

    print(f"Min: {min_count}, Max: {max_count}, "
          f"Mean: {mean_count:.1f}, Std: {std_count:.1f}")
    print()

    # Create histogram
    hist, bin_edges = np.histogram(sample_counts, bins=bins)

    # Print histogram
    max_bar_length = 50
    max_freq = max(hist)

    for i in range(len(hist)):
        bin_start = int(bin_edges[i])
        bin_end = int(bin_edges[i + 1])
        freq = hist[i]

        bar_length = int(max_bar_length * freq / max_freq) if max_freq > 0 else 0
        bar = "█" * bar_length

        print(f"{bin_start:>6} - {bin_end:<6} | {bar} {freq}")

    print("="*80)


def compare_distributions_side_by_side(
    partitions: List[Dict],
    partition_names: List[str],
    targets,
    num_classes: int,
    client_id: int = 0,
):
    """
    Compare label distributions for a specific client across different partitioning strategies.

    Args:
        partitions: List of client_indices dictionaries
        partition_names: Names for each partition
        targets: All labels
        num_classes: Number of classes
        client_id: Which client to compare
    """
    print(f"\nLABEL DISTRIBUTION COMPARISON (Client {client_id})")
    print("="*80)

    print(f"{'Label':<10} | ", end="")
    for name in partition_names:
        print(f"{name:<20} | ", end="")
    print()
    print("-"*80)

    # Compute distributions
    distributions = []
    for partition in partitions:
        indices = partition[client_id]
        client_targets = targets[indices]
        counts = np.bincount(client_targets.numpy(), minlength=num_classes)
        distributions.append(counts)

    # Print row by row
    for label in range(num_classes):
        print(f"Class {label:<4} | ", end="")
        for counts in distributions:
            count = counts[label]
            pct = 100.0 * count / counts.sum() if counts.sum() > 0 else 0
            print(f"{count:>5} ({pct:>5.1f}%)     | ", end="")
        print()

    print("="*80)
