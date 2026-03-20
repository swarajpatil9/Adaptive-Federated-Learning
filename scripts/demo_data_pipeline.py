"""
Demo script for the data pipeline (Phase 2).

This script demonstrates:
1. Loading federated datasets
2. Partitioning data across clients
3. Creating client DataLoaders
4. Computing and visualizing statistics
5. Comparing IID vs Non-IID distributions

Usage:
    python scripts/demo_data_pipeline.py
    python scripts/demo_data_pipeline.py --dataset cifar10 --num-clients 50
    python scripts/demo_data_pipeline.py --strategy dirichlet --alpha 0.1
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from aflf.data import (
    FederatedDataModule,
    measure_data_heterogeneity,
    print_dataset_info,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demo: Federated Data Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10", "cifar100"],
        help="Dataset to use",
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=10,
        help="Number of federated clients",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="iid",
        choices=["iid", "dirichlet", "pathological"],
        help="Partitioning strategy",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet concentration parameter (for dirichlet strategy)",
    )

    parser.add_argument(
        "--shards-per-client",
        type=int,
        default=2,
        help="Number of shards per client (for pathological strategy)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--save-partition",
        type=str,
        default=None,
        help="Path to save partition metadata",
    )

    parser.add_argument(
        "--show-all-clients",
        action="store_true",
        help="Show label distribution for all clients (not just first 10)",
    )

    return parser.parse_args()


def demo_basic_usage(args):
    """Demonstrate basic data module usage."""
    print("\n" + "="*80)
    print("DEMO: BASIC DATA MODULE USAGE")
    print("="*80)

    # Prepare partition kwargs
    partition_kwargs = {}
    if args.strategy == "dirichlet":
        partition_kwargs['alpha'] = args.alpha
    elif args.strategy == "pathological":
        partition_kwargs['shards_per_client'] = args.shards_per_client

    # Create federated data module
    print("\n[1] Creating FederatedDataModule...")
    data_module = FederatedDataModule(
        dataset_name=args.dataset,
        data_root="./data",
        num_clients=args.num_clients,
        partition_strategy=args.strategy,
        batch_size=args.batch_size,
        seed=args.seed,
        download=True,
        **partition_kwargs,
    )
    print("✓ Data module created successfully")

    # Print summary
    print("\n[2] Dataset Summary:")
    data_module.print_summary()

    # Visualize distribution
    print("\n[3] Label Distribution Across Clients:")
    num_to_show = args.num_clients if args.show_all_clients else 10
    data_module.visualize_distribution(num_clients_to_show=num_to_show)

    return data_module


def demo_client_loaders(data_module, num_clients_to_test: int = 3):
    """Demonstrate creating and using client DataLoaders."""
    print("\n" + "="*80)
    print("DEMO: CLIENT DATALOADERS")
    print("="*80)

    for client_id in range(num_clients_to_test):
        print(f"\n[Client {client_id}]")

        # Get client DataLoader
        loader = data_module.get_client_loader(client_id, shuffle=False)

        # Get first batch
        batch_x, batch_y = next(iter(loader))

        print(f"  DataLoader batches: {len(loader)}")
        print(f"  First batch shape: {batch_x.shape}")
        print(f"  First batch labels: {batch_y.tolist()[:10]}...")

        # Client statistics
        stats = data_module.get_client_statistics(client_id)
        print(f"  Total samples: {stats['num_samples']}")
        print(f"  Unique labels: {stats['unique_labels']}")


def demo_heterogeneity_metrics(data_module):
    """Demonstrate heterogeneity measurement."""
    print("\n" + "="*80)
    print("DEMO: DATA HETEROGENEITYMETRICS")
    print("="*80)

    targets = data_module.train_dataset.get_targets()
    client_indices = data_module.partitioner.client_indices

    metrics = measure_data_heterogeneity(
        client_indices=client_indices,
        targets=targets,
        num_classes=data_module.num_classes,
    )

    print("\nHeterogeneity Metrics:")
    print(f"  Avg KL divergence from uniform: {metrics['avg_kl_from_uniform']:.4f}")
    print(f"  Avg entropy: {metrics['avg_entropy']:.4f}")
    print(f"  Normalized entropy: {metrics['normalized_entropy']:.4f}")
    print(f"  Entropy std: {metrics['entropy_std']:.4f}")

    print("\nInterpretation:")
    if metrics['normalized_entropy'] > 0.9:
        print("  → Data is close to IID (low heterogeneity)")
    elif metrics['normalized_entropy'] > 0.7:
        print("  → Moderate heterogeneity")
    else:
        print("  → High heterogeneity (strongly non-IID)")


def demo_test_loader(data_module):
    """Demonstrate centralized test loader."""
    print("\n" + "="*80)
    print("DEMO: CENTRALIZED TEST LOADER")
    print("="*80)

    test_loader = data_module.get_test_loader()

    # Get first batch
    batch_x, batch_y = next(iter(test_loader))

    print(f"\nTest DataLoader:")
    print(f"  Total batches: {len(test_loader)}")
    print(f"  Batch shape: {batch_x.shape}")
    print(f"  Labels shape: {batch_y.shape}")

    stats = data_module.get_statistics()['test']
    print(f"  Total test samples: {stats['num_samples']}")


def demo_save_partition(data_module, save_path: Optional[str]):
    """Demonstrate saving partition for reproducibility."""
    if save_path is None:
        return

    print("\n" + "="*80)
    print("DEMO: SAVING PARTITION METADATA")
    print("="*80)

    data_module.save_partition(save_path)
    print(f"✓ Partition metadata saved to: {save_path}")
    print("  This can be loaded later to reproduce exact data splits")


def compare_strategies():
    """Compare IID vs Non-IID partitioning side-by-side."""
    print("\n" + "="*80)
    print("COMPARISON: IID vs NON-IID PARTITIONING")
    print("="*80)

    strategies = [
        ("IID", "iid", {}),
        ("Dirichlet (α=1.0)", "dirichlet", {'alpha': 1.0}),
        ("Dirichlet (α=0.5)", "dirichlet", {'alpha': 0.5}),
        ("Dirichlet (α=0.1)", "dirichlet", {'alpha': 0.1}),
        ("Pathological (2 shards)", "pathological", {'shards_per_client': 2}),
    ]

    results = []

    for name, strategy, kwargs in strategies:
        print(f"\n[{name}]")

        # Create data module
        dm = FederatedDataModule(
            dataset_name="mnist",
            data_root="./data",
            num_clients=10,
            partition_strategy=strategy,
            batch_size=32,
            seed=42,
            download=True,
            **kwargs,
        )

        # Compute metrics
        targets = dm.train_dataset.get_targets()
        metrics = measure_data_heterogeneity(
            client_indices=dm.partitioner.client_indices,
            targets=targets,
            num_classes=dm.num_classes,
        )

        # Get partition stats
        partition_stats = dm.partitioner.get_all_statistics()
        ulpc = partition_stats['unique_labels_per_client']

        results.append({
            'name': name,
            'kl': metrics['avg_kl_from_uniform'],
            'entropy': metrics['normalized_entropy'],
            'unique_labels': ulpc['mean'],
        })

        print(f"  KL divergence: {metrics['avg_kl_from_uniform']:.4f}")
        print(f"  Normalized entropy: {metrics['normalized_entropy']:.4f}")
        print(f"  Avg unique labels per client: {ulpc['mean']:.1f}")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Strategy':<30} {'KL Div':<10} {'Entropy':<10} {'Labels/Client':<15}")
    print("-"*80)
    for r in results:
        print(f"{r['name']:<30} {r['kl']:<10.4f} {r['entropy']:<10.4f} {r['unique_labels']:<15.1f}")


def main():
    """Main demo execution."""
    args = parse_args()

    print("="*80)
    print("FEDERATED LEARNING DATA PIPELINE DEMO (PHASE 2)")
    print("="*80)

    # Show available datasets
    print_dataset_info()

    # Run demos
    data_module = demo_basic_usage(args)
    demo_client_loaders(data_module, num_clients_to_test=3)
    demo_heterogeneity_metrics(data_module)
    demo_test_loader(data_module)
    demo_save_partition(data_module, args.save_partition)

    # Optional: Compare strategies
    if args.strategy == "iid" and args.dataset == "mnist" and args.num_clients == 10:
        print("\n" + "="*80)
        print("BONUS: Running comparison across all strategies...")
        print("="*80)
        compare_strategies()

    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nKey takeaways:")
    print("  ✓ Data pipeline is fully functional")
    print("  ✓ IID and Non-IID partitioning strategies implemented")
    print("  ✓ Client DataLoaders ready for federated training")
    print("  ✓ Reproducibility ensured via seeding")
    print("\nNext phase: Model architectures and client training")


if __name__ == "__main__":
    main()
