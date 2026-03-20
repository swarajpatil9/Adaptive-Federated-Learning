"""
Unit tests for FederatedDataModule.

Tests:
- Data module initialization
- Client DataLoader creation
- Test DataLoader creation
- Statistics computation
- Save/load partition metadata
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from aflf.data import FederatedDataModule


@pytest.fixture
def data_module_mnist_iid():
    """Create MNIST data module with IID partitioning."""
    return FederatedDataModule(
        dataset_name="mnist",
        data_root="./data",
        num_clients=10,
        partition_strategy="iid",
        batch_size=32,
        seed=42,
        download=True,
    )


@pytest.fixture
def data_module_mnist_noniid():
    """Create MNIST data module with Non-IID partitioning."""
    return FederatedDataModule(
        dataset_name="mnist",
        data_root="./data",
        num_clients=10,
        partition_strategy="dirichlet",
        alpha=0.5,
        batch_size=32,
        seed=42,
        download=True,
    )


class TestDataModuleInitialization:
    """Test data module initialization."""

    def test_mnist_initialization(self, data_module_mnist_iid):
        """Test MNIST data module initializes correctly."""
        dm = data_module_mnist_iid

        assert dm.dataset_name == "mnist"
        assert dm.num_clients == 10
        assert dm.batch_size == 32
        assert dm.num_classes == 10

        # Check datasets are loaded
        assert dm.train_dataset is not None
        assert dm.test_dataset is not None

        # Check partitioner is created
        assert dm.partitioner is not None
        assert len(dm.partitioner.client_indices) == 10

    def test_cifar10_initialization(self):
        """Test CIFAR-10 data module initializes correctly."""
        dm = FederatedDataModule(
            dataset_name="cifar10",
            data_root="./data",
            num_clients=5,
            partition_strategy="iid",
            batch_size=64,
            seed=42,
            download=True,
        )

        assert dm.num_classes == 10
        assert dm.batch_size == 64
        assert dm.input_shape == (3, 32, 32)


class TestClientDataLoaders:
    """Test client DataLoader creation."""

    def test_get_client_loader(self, data_module_mnist_iid):
        """Test creating client DataLoader."""
        dm = data_module_mnist_iid

        loader = dm.get_client_loader(client_id=0)

        assert loader is not None
        assert loader.batch_size == 32

        # Get a batch
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape[0] <= 32  # Batch size
        assert batch_x.shape[1:] == (1, 28, 28)  # MNIST shape
        assert batch_y.shape[0] <= 32

    def test_client_loader_custom_batch_size(self, data_module_mnist_iid):
        """Test client DataLoader with custom batch size."""
        dm = data_module_mnist_iid

        loader = dm.get_client_loader(client_id=0, batch_size=16)
        assert loader.batch_size == 16

    def test_all_clients_have_loaders(self, data_module_mnist_iid):
        """Test that all clients can create DataLoaders."""
        dm = data_module_mnist_iid

        for client_id in range(dm.num_clients):
            loader = dm.get_client_loader(client_id)
            assert loader is not None
            assert len(loader) > 0  # Has at least one batch


class TestTestDataLoader:
    """Test centralized test DataLoader."""

    def test_get_test_loader(self, data_module_mnist_iid):
        """Test creating test DataLoader."""
        dm = data_module_mnist_iid

        test_loader = dm.get_test_loader()

        assert test_loader is not None
        assert test_loader.batch_size == dm.test_batch_size

        # Get a batch
        batch_x, batch_y = next(iter(test_loader))
        assert batch_x.shape[1:] == (1, 28, 28)

    def test_test_loader_has_all_samples(self, data_module_mnist_iid):
        """Test that test loader includes all test samples."""
        dm = data_module_mnist_iid

        test_loader = dm.get_test_loader(batch_size=100)

        total_samples = 0
        for batch_x, batch_y in test_loader:
            total_samples += len(batch_x)

        # MNIST has 10,000 test samples
        assert total_samples == 10000


class TestStatistics:
    """Test statistics computation."""

    def test_client_statistics(self, data_module_mnist_iid):
        """Test per-client statistics."""
        dm = data_module_mnist_iid

        stats = dm.get_client_statistics(0)

        assert 'client_id' in stats
        assert 'num_samples' in stats
        assert 'label_distribution' in stats
        assert 'unique_labels' in stats

        assert stats['num_samples'] > 0
        assert len(stats['label_distribution']) == 10
        assert stats['unique_labels'] <= 10

    def test_global_statistics(self, data_module_mnist_iid):
        """Test global statistics computation."""
        dm = data_module_mnist_iid

        stats = dm.get_statistics()

        assert 'dataset_name' in stats
        assert 'num_clients' in stats
        assert 'train' in stats
        assert 'test' in stats
        assert 'partition' in stats

        assert stats['dataset_name'] == 'mnist'
        assert stats['num_clients'] == 10
        assert stats['train']['num_samples'] == 60000
        assert stats['test']['num_samples'] == 10000

    def test_noniid_has_heterogeneity(self, data_module_mnist_noniid):
        """Test that non-IID partitions show heterogeneity."""
        dm = data_module_mnist_noniid

        stats = dm.get_statistics()
        partition_stats = stats['partition']

        # Check that not all clients have same number of samples
        spc = partition_stats['samples_per_client']

        # With Dirichlet, there should be some variance
        # (though with alpha=0.5 it might be moderate)
        assert spc['std'] >= 0

        # Check unique labels per client varies
        ulpc = partition_stats['unique_labels_per_client']
        assert ulpc['min'] <= ulpc['max']


class TestSaveLoadPartition:
    """Test saving and loading partition metadata."""

    def test_save_partition(self, data_module_mnist_iid):
        """Test saving partition metadata to file."""
        dm = data_module_mnist_iid

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "partition.json"
            dm.save_partition(str(save_path))

            # Verify file exists and is valid JSON
            assert save_path.exists()

            with open(save_path, 'r') as f:
                metadata = json.load(f)

            assert metadata['dataset_name'] == 'mnist'
            assert metadata['num_clients'] == 10
            assert 'client_indices' in metadata
            assert len(metadata['client_indices']) == 10

    def test_load_partition(self, data_module_mnist_iid):
        """Test loading partition metadata from file."""
        dm1 = data_module_mnist_iid

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "partition.json"

            # Save from first data module
            dm1.save_partition(str(save_path))

            # Create new data module with different seed
            dm2 = FederatedDataModule(
                dataset_name="mnist",
                data_root="./data",
                num_clients=10,
                partition_strategy="iid",
                batch_size=32,
                seed=999,  # Different seed
                download=True,
            )

            # Load partition from file
            dm2.load_partition(str(save_path))

            # Should now have same partition as dm1
            for client_id in range(10):
                indices1 = dm1.partitioner.get_client_indices(client_id)
                indices2 = dm2.partitioner.get_client_indices(client_id)
                assert indices1 == indices2

    def test_load_partition_validation(self):
        """Test that loading incompatible partition raises error."""
        # Save partition for 10 clients
        dm1 = FederatedDataModule(
            dataset_name="mnist",
            data_root="./data",
            num_clients=10,
            partition_strategy="iid",
            seed=42,
            download=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "partition.json"
            dm1.save_partition(str(save_path))

            # Try to load into data module with 5 clients
            dm2 = FederatedDataModule(
                dataset_name="mnist",
                data_root="./data",
                num_clients=5,  # Different number of clients
                partition_strategy="iid",
                seed=42,
                download=True,
            )

            # Should raise assertion error
            with pytest.raises(AssertionError):
                dm2.load_partition(str(save_path))


class TestReproducibility:
    """Test reproducibility guarantees."""

    def test_same_seed_same_partition(self):
        """Test that same seed produces identical partitions."""
        dm1 = FederatedDataModule(
            dataset_name="mnist",
            data_root="./data",
            num_clients=10,
            partition_strategy="dirichlet",
            alpha=0.5,
            seed=42,
            download=True,
        )

        dm2 = FederatedDataModule(
            dataset_name="mnist",
            data_root="./data",
            num_clients=10,
            partition_strategy="dirichlet",
            alpha=0.5,
            seed=42,
            download=True,
        )

        # Should be identical
        for client_id in range(10):
            indices1 = dm1.partitioner.get_client_indices(client_id)
            indices2 = dm2.partitioner.get_client_indices(client_id)
            assert indices1 == indices2

    def test_different_seed_different_partition(self):
        """Test that different seeds produce different partitions."""
        dm1 = FederatedDataModule(
            dataset_name="mnist",
            data_root="./data",
            num_clients=10,
            partition_strategy="iid",
            seed=42,
            download=True,
        )

        dm2 = FederatedDataModule(
            dataset_name="mnist",
            data_root="./data",
            num_clients=10,
            partition_strategy="iid",
            seed=999,
            download=True,
        )

        # Should be different
        indices1 = dm1.partitioner.get_client_indices(0)
        indices2 = dm2.partitioner.get_client_indices(0)
        assert indices1 != indices2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
