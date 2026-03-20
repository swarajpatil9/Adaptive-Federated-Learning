"""
Unit tests for data partitioning strategies.

Tests:
- IID partitioning correctness
- Dirichlet partitioning properties
- Pathological partitioning
- Edge cases and validation
"""

import pytest
import torch

from aflf.data import DataPartitioner
from aflf.data.utils import validate_partition


@pytest.fixture
def simple_targets():
    """Create simple targets for testing."""
    # 100 samples, 10 classes, balanced
    targets = torch.arange(100) % 10
    return targets


@pytest.fixture
def imbalanced_targets():
    """Create imbalanced targets for testing."""
    # Class 0: 50 samples, Class 1: 30 samples, Class 2: 20 samples
    targets = torch.cat([
        torch.zeros(50, dtype=torch.long),
        torch.ones(30, dtype=torch.long),
        torch.full((20,), 2, dtype=torch.long),
    ])
    return targets


class TestIIDPartitioning:
    """Test IID partitioning strategy."""

    def test_basic_iid_partition(self, simple_targets):
        """Test basic IID partitioning."""
        partitioner = DataPartitioner(
            targets=simple_targets,
            num_clients=10,
            partition_strategy="iid",
            seed=42,
        )

        # Verify all samples are assigned
        assert validate_partition(
            partitioner.client_indices,
            total_samples=len(simple_targets),
        )

        # Each client should have approximately 10 samples
        for client_id in range(10):
            indices = partitioner.get_client_indices(client_id)
            assert len(indices) == 10

    def test_iid_uneven_split(self, simple_targets):
        """Test IID with uneven number of clients."""
        partitioner = DataPartitioner(
            targets=simple_targets,
            num_clients=7,  # Doesn't divide evenly into 100
            partition_strategy="iid",
            seed=42,
        )

        assert validate_partition(
            partitioner.client_indices,
            total_samples=len(simple_targets),
        )

        # Last client gets extra samples
        sample_counts = [
            len(partitioner.get_client_indices(i)) for i in range(7)
        ]
        assert sum(sample_counts) == 100
        assert sample_counts[-1] >= sample_counts[0]

    def test_iid_reproducibility(self, simple_targets):
        """Test that same seed produces same partition."""
        p1 = DataPartitioner(
            targets=simple_targets,
            num_clients=5,
            partition_strategy="iid",
            seed=42,
        )

        p2 = DataPartitioner(
            targets=simple_targets,
            num_clients=5,
            partition_strategy="iid",
            seed=42,
        )

        # Should produce identical partitions
        for client_id in range(5):
            assert p1.get_client_indices(client_id) == p2.get_client_indices(client_id)


class TestDirichletPartitioning:
    """Test Dirichlet-based non-IID partitioning."""

    def test_basic_dirichlet_partition(self, simple_targets):
        """Test basic Dirichlet partitioning."""
        partitioner = DataPartitioner(
            targets=simple_targets,
            num_clients=10,
            partition_strategy="dirichlet",
            alpha=0.5,
            seed=42,
        )

        # Verify all samples are assigned
        assert validate_partition(
            partitioner.client_indices,
            total_samples=len(simple_targets),
        )

    def test_dirichlet_creates_heterogeneity(self, simple_targets):
        """Test that Dirichlet creates heterogeneous distributions."""
        partitioner = DataPartitioner(
            targets=simple_targets,
            num_clients=10,
            partition_strategy="dirichlet",
            alpha=0.1,  # Low alpha = high heterogeneity
            seed=42,
        )

        # Check that clients have different label distributions
        label_dists = []
        for client_id in range(10):
            stats = partitioner.get_client_statistics(client_id)
            label_dists.append(tuple(stats['label_distribution']))

        # Not all distributions should be the same
        unique_dists = len(set(label_dists))
        assert unique_dists > 1, "All clients have identical distributions"

        # With low alpha, some clients should have <10 unique labels
        unique_labels = [
            partitioner.get_client_statistics(i)['unique_labels']
            for i in range(10)
        ]
        assert min(unique_labels) < 10, "Expected some clients with fewer than all labels"

    def test_dirichlet_alpha_effect(self, simple_targets):
        """Test that alpha controls heterogeneity level."""
        # High alpha (closer to IID)
        p_high = DataPartitioner(
            targets=simple_targets,
            num_clients=10,
            partition_strategy="dirichlet",
            alpha=10.0,
            seed=42,
        )

        # Low alpha (more heterogeneous)
        p_low = DataPartitioner(
            targets=simple_targets,
            num_clients=10,
            partition_strategy="dirichlet",
            alpha=0.1,
            seed=42,
        )

        # High alpha should have more unique labels per client on average
        unique_high = [
            p_high.get_client_statistics(i)['unique_labels']
            for i in range(10)
        ]
        unique_low = [
            p_low.get_client_statistics(i)['unique_labels']
            for i in range(10)
        ]

        avg_unique_high = sum(unique_high) / len(unique_high)
        avg_unique_low = sum(unique_low) / len(unique_low)

        assert avg_unique_high >= avg_unique_low, \
            "High alpha should result in more labels per client"


class TestPathologicalPartitioning:
    """Test pathological non-IID partitioning."""

    def test_basic_pathological_partition(self, simple_targets):
        """Test basic pathological partitioning."""
        partitioner = DataPartitioner(
            targets=simple_targets,
            num_clients=10,
            partition_strategy="pathological",
            shards_per_client=2,
            seed=42,
        )

        # Verify all samples are assigned
        assert validate_partition(
            partitioner.client_indices,
            total_samples=len(simple_targets),
        )

    def test_pathological_shards_per_client(self, simple_targets):
        """Test that clients have at most shards_per_client unique labels."""
        partitioner = DataPartitioner(
            targets=simple_targets,
            num_clients=10,
            partition_strategy="pathological",
            shards_per_client=2,
            seed=42,
        )

        # Each client should have at most 2 unique labels
        # (may have fewer if shards come from same class due to sorting)
        for client_id in range(10):
            stats = partitioner.get_client_statistics(client_id)
            assert stats['unique_labels'] <= 2, \
                f"Client {client_id} has {stats['unique_labels']} labels, expected ≤2"


class TestPartitionerStatistics:
    """Test statistics computation."""

    def test_client_statistics(self, simple_targets):
        """Test per-client statistics computation."""
        partitioner = DataPartitioner(
            targets=simple_targets,
            num_clients=5,
            partition_strategy="iid",
            seed=42,
        )

        stats = partitioner.get_client_statistics(0)

        assert 'client_id' in stats
        assert 'num_samples' in stats
        assert 'label_distribution' in stats
        assert 'unique_labels' in stats
        assert stats['num_samples'] > 0
        assert len(stats['label_distribution']) == 10

    def test_all_statistics(self, simple_targets):
        """Test overall partition statistics."""
        partitioner = DataPartitioner(
            targets=simple_targets,
            num_clients=5,
            partition_strategy="iid",
            seed=42,
        )

        stats = partitioner.get_all_statistics()

        assert stats['num_clients'] == 5
        assert stats['total_samples'] == 100
        assert 'samples_per_client' in stats
        assert 'unique_labels_per_client' in stats
        assert 'client_statistics' in stats
        assert len(stats['client_statistics']) == 5


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_client(self, simple_targets):
        """Test partitioning with single client."""
        partitioner = DataPartitioner(
            targets=simple_targets,
            num_clients=1,
            partition_strategy="iid",
            seed=42,
        )

        indices = partitioner.get_client_indices(0)
        assert len(indices) == 100

    def test_more_clients_than_samples(self):
        """Test handling when num_clients > num_samples."""
        targets = torch.tensor([0, 1, 2, 3, 4])  # Only 5 samples

        partitioner = DataPartitioner(
            targets=targets,
            num_clients=10,  # More clients than samples
            partition_strategy="iid",
            seed=42,
        )

        # Some clients will have 0 samples
        sample_counts = [
            len(partitioner.get_client_indices(i))
            for i in range(10)
        ]
        assert sum(sample_counts) == 5
        assert min(sample_counts) == 0

    def test_invalid_strategy(self, simple_targets):
        """Test error handling for invalid strategy."""
        with pytest.raises(ValueError, match="Unknown partition strategy"):
            DataPartitioner(
                targets=simple_targets,
                num_clients=5,
                partition_strategy="invalid_strategy",
                seed=42,
            )

    def test_invalid_client_id(self, simple_targets):
        """Test error handling for invalid client_id."""
        partitioner = DataPartitioner(
            targets=simple_targets,
            num_clients=5,
            partition_strategy="iid",
            seed=42,
        )

        with pytest.raises(ValueError, match="Invalid client_id"):
            partitioner.get_client_indices(10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
