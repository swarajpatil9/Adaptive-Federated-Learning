"""
Quick integration test for Phase 2 data pipeline.

Runs a minimal test to verify all components work together.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from aflf.data import (
    FederatedDataModule,
    measure_data_heterogeneity,
    validate_partition,
)


def test_phase2_integration():
    """Quick integration test for Phase 2."""
    print("="*80)
    print("PHASE 2 INTEGRATION TEST")
    print("="*80)

    # Test 1: IID partitioning
    print("\n[Test 1] IID Partitioning...")
    dm_iid = FederatedDataModule(
        dataset_name="mnist",
        data_root="./data",
        num_clients=5,
        partition_strategy="iid",
        batch_size=32,
        seed=42,
        download=True,
    )

    # Validate partition
    validate_partition(
        dm_iid.partitioner.client_indices,
        total_samples=len(dm_iid.train_dataset),
    )
    print("  ✓ IID partition valid")

    # Test client loader
    loader = dm_iid.get_client_loader(0)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (32, 1, 28, 28)
    assert batch_y.shape == (32,)
    print("  ✓ Client DataLoader works")

    # Test test loader
    test_loader = dm_iid.get_test_loader()
    batch_x, batch_y = next(iter(test_loader))
    assert batch_x.shape[1:] == (1, 28, 28)
    print("  ✓ Test DataLoader works")

    # Test 2: Non-IID Dirichlet partitioning
    print("\n[Test 2] Non-IID Dirichlet Partitioning...")
    dm_noniid = FederatedDataModule(
        dataset_name="mnist",
        data_root="./data",
        num_clients=5,
        partition_strategy="dirichlet",
        alpha=0.5,
        batch_size=32,
        seed=42,
        download=True,
    )

    validate_partition(
        dm_noniid.partitioner.client_indices,
        total_samples=len(dm_noniid.train_dataset),
    )
    print("  ✓ Dirichlet partition valid")

    # Measure heterogeneity
    targets = dm_noniid.train_dataset.get_targets()
    metrics = measure_data_heterogeneity(
        dm_noniid.partitioner.client_indices,
        targets,
        dm_noniid.num_classes,
    )
    assert metrics['avg_kl_from_uniform'] >= 0
    print(f"  ✓ Heterogeneity measured (KL: {metrics['avg_kl_from_uniform']:.4f})")

    # Test 3: Pathological partitioning
    print("\n[Test 3] Pathological Partitioning...")
    dm_pathological = FederatedDataModule(
        dataset_name="mnist",
        data_root="./data",
        num_clients=5,
        partition_strategy="pathological",
        shards_per_client=2,
        batch_size=32,
        seed=42,
        download=True,
    )

    validate_partition(
        dm_pathological.partitioner.client_indices,
        total_samples=len(dm_pathological.train_dataset),
    )
    print("  ✓ Pathological partition valid")

    # Verify clients have limited labels
    for client_id in range(5):
        stats = dm_pathological.get_client_statistics(client_id)
        assert stats['unique_labels'] <= 2
    print("  ✓ Clients have ≤2 unique labels")

    # Test 4: Statistics and reproducibility
    print("\n[Test 4] Statistics and Reproducibility...")
    stats = dm_iid.get_statistics()
    assert 'dataset_name' in stats
    assert 'train' in stats
    assert 'test' in stats
    print("  ✓ Statistics computed")

    # Test 5: Multiple datasets
    print("\n[Test 5] Multiple Datasets...")
    for dataset_name in ["mnist", "cifar10"]:
        dm = FederatedDataModule(
            dataset_name=dataset_name,
            data_root="./data",
            num_clients=3,
            partition_strategy="iid",
            seed=42,
            download=True,
        )
        loader = dm.get_client_loader(0)
        batch_x, _ = next(iter(loader))
        assert batch_x.shape[0] > 0
        print(f"  ✓ {dataset_name.upper()} works")

    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
    print("\nPhase 2 data pipeline is fully operational!")


if __name__ == "__main__":
    try:
        test_phase2_integration()
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
