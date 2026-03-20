"""
Main interface for federated data management.

The FederatedDataModule class orchestrates:
- Dataset loading
- Client partitioning
- DataLoader creation
- Statistics tracking
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from .base import FederatedDataset
from .datasets import create_dataset
from .partitioner import DataPartitioner


class FederatedDataModule:
    """
    Main data management interface for federated learning.

    This class handles:
    - Loading datasets (train/test)
    - Partitioning training data across clients
    - Creating per-client DataLoaders
    - Computing and caching statistics
    - Saving/loading partition metadata for reproducibility
    """

    def __init__(
        self,
        dataset_name: str,
        data_root: str = "./data",
        num_clients: int = 10,
        partition_strategy: str = "iid",
        batch_size: int = 32,
        test_batch_size: Optional[int] = None,
        num_workers: int = 0,
        seed: int = 42,
        download: bool = True,
        **partition_kwargs,
    ):
        """
        Initialize federated data module.

        Args:
            dataset_name: Name of dataset ('mnist', 'cifar10', 'cifar100')
            data_root: Root directory for dataset storage
            num_clients: Number of federated clients
            partition_strategy: Partitioning strategy ('iid', 'dirichlet', 'pathological')
            batch_size: Batch size for client training
            test_batch_size: Batch size for testing (defaults to batch_size)
            num_workers: Number of DataLoader workers
            seed: Random seed for reproducibility
            download: If True, download dataset if not present
            **partition_kwargs: Additional arguments for partitioner
                - alpha: Dirichlet concentration (default: 0.5)
                - shards_per_client: For pathological splits (default: 2)
                - min_samples_per_client: Minimum samples per client (default: 10)
        """
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.num_clients = num_clients
        self.partition_strategy = partition_strategy
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size or batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.partition_kwargs = partition_kwargs

        # Load datasets
        self.train_dataset = create_dataset(
            name=dataset_name,
            root=data_root,
            train=True,
            download=download,
            augment=True,
        )

        self.test_dataset = create_dataset(
            name=dataset_name,
            root=data_root,
            train=False,
            download=download,
            augment=False,
        )

        # Partition training data
        self.partitioner = DataPartitioner(
            targets=self.train_dataset.get_targets(),
            num_clients=num_clients,
            partition_strategy=partition_strategy,
            seed=seed,
            **partition_kwargs,
        )

        # Cache for statistics
        self._statistics: Optional[Dict] = None

    def get_client_loader(
        self,
        client_id: int,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> DataLoader:
        """
        Get DataLoader for a specific client's training data.

        Args:
            client_id: Client ID (0 to num_clients-1)
            batch_size: Batch size (defaults to self.batch_size)
            shuffle: If True, shuffle data
            drop_last: If True, drop last incomplete batch

        Returns:
            PyTorch DataLoader for this client
        """
        indices = self.partitioner.get_client_indices(client_id)
        subset = Subset(self.train_dataset.dataset, indices)

        loader = DataLoader(
            subset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=drop_last,
            pin_memory=torch.cuda.is_available(),
        )

        return loader

    def get_test_loader(
        self,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        """
        Get DataLoader for centralized test set.

        In FL, the test set is typically centralized on the server
        for fair evaluation across all clients.

        Args:
            batch_size: Batch size (defaults to self.test_batch_size)

        Returns:
            PyTorch DataLoader for test data
        """
        loader = DataLoader(
            self.test_dataset.dataset,
            batch_size=batch_size or self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        return loader

    def get_client_statistics(self, client_id: int) -> Dict:
        """
        Get data statistics for a specific client.

        Args:
            client_id: Client ID

        Returns:
            Dictionary with client-specific statistics
        """
        return self.partitioner.get_client_statistics(client_id)

    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics for the entire federated setup.

        Returns:
            Dictionary with dataset and partition statistics
        """
        if self._statistics is None:
            train_stats = self.train_dataset.get_statistics()
            partition_stats = self.partitioner.get_all_statistics()
            test_stats = self.test_dataset.get_statistics()

            self._statistics = {
                'dataset_name': self.dataset_name,
                'num_clients': self.num_clients,
                'batch_size': self.batch_size,
                'seed': self.seed,
                'train': train_stats,
                'test': test_stats,
                'partition': partition_stats,
            }

        return self._statistics

    def save_partition(self, save_path: str):
        """
        Save partition metadata for reproducibility.

        Saves client_id -> sample_indices mapping and statistics
        to a JSON file.

        Args:
            save_path: Path to save partition metadata
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            'dataset_name': self.dataset_name,
            'num_clients': self.num_clients,
            'partition_strategy': self.partition_strategy,
            'seed': self.seed,
            'partition_kwargs': self.partition_kwargs,
            'client_indices': {
                str(k): v for k, v in self.partitioner.client_indices.items()
            },
            'statistics': self.get_statistics(),
        }

        with open(save_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Partition saved to: {save_path}")

    def load_partition(self, load_path: str):
        """
        Load partition metadata from file.

        Warning: This will override the current partition.
        Useful for reproducing exact experimental setups.

        Args:
            load_path: Path to partition metadata file
        """
        with open(load_path, 'r') as f:
            metadata = json.load(f)

        # Verify compatibility
        assert metadata['dataset_name'] == self.dataset_name, \
            "Dataset name mismatch"
        assert metadata['num_clients'] == self.num_clients, \
            "Number of clients mismatch"

        # Load client indices
        self.partitioner.client_indices = {
            int(k): v for k, v in metadata['client_indices'].items()
        }

        print(f"Partition loaded from: {load_path}")

    def print_summary(self):
        """Print human-readable summary of the federated data setup."""
        stats = self.get_statistics()

        print("="*80)
        print("FEDERATED DATA MODULE SUMMARY")
        print("="*80)
        print(f"Dataset: {self.dataset_name.upper()}")
        print(f"Number of clients: {self.num_clients}")
        print(f"Partition strategy: {self.partition_strategy}")
        if self.partition_kwargs:
            print(f"Strategy parameters: {self.partition_kwargs}")
        print(f"Random seed: {self.seed}")
        print()

        print("TRAINING DATA")
        print(f"  Total samples: {stats['train']['num_samples']}")
        print(f"  Number of classes: {stats['train']['num_classes']}")
        print()

        print("TEST DATA")
        print(f"  Total samples: {stats['test']['num_samples']}")
        print()

        print("PARTITION STATISTICS")
        spc = stats['partition']['samples_per_client']
        print(f"  Samples per client: {spc['min']} - {spc['max']} "
              f"(mean: {spc['mean']:.1f}, std: {spc['std']:.1f})")

        ulpc = stats['partition']['unique_labels_per_client']
        print(f"  Unique labels per client: {ulpc['min']} - {ulpc['max']} "
              f"(mean: {ulpc['mean']:.1f})")
        print("="*80)

    def visualize_distribution(self, num_clients_to_show: int = 10):
        """
        Print ASCII visualization of label distribution across clients.

        Args:
            num_clients_to_show: Number of clients to display (default: 10)
        """
        stats = self.get_statistics()
        client_stats = stats['partition']['client_statistics']

        print("\nCLIENT LABEL DISTRIBUTION")
        print("="*80)

        # Show first N clients
        for client_stat in client_stats[:num_clients_to_show]:
            client_id = client_stat['client_id']
            label_dist = client_stat['label_distribution']
            num_samples = client_stat['num_samples']

            print(f"Client {client_id:3d} ({num_samples:5d} samples): ", end="")

            # ASCII bar chart
            max_count = max(label_dist) if max(label_dist) > 0 else 1
            for label, count in enumerate(label_dist):
                if count > 0:
                    bar_length = int(20 * count / max_count)
                    bar = "█" * bar_length
                    print(f"[{label}:{bar:<20} {count:4d}] ", end="")
            print()

        if self.num_clients > num_clients_to_show:
            print(f"... ({self.num_clients - num_clients_to_show} more clients)")

        print("="*80)

    @property
    def num_classes(self) -> int:
        """Get number of classes in the dataset."""
        return self.train_dataset.num_classes

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Get shape of a single input sample.

        Returns:
            Tuple representing (C, H, W) for images
        """
        sample, _ = self.train_dataset.dataset[0]
        return tuple(sample.shape)
