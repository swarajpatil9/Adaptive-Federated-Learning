"""
Data partitioning strategies for federated learning.

Implements:
- IID: Random uniform splits
- Non-IID Dirichlet: Label skew via Dirichlet distribution
- Pathological Non-IID: Fixed shards per client (FedAvg paper)
"""

from typing import Dict, List, Optional

import numpy as np
import torch


class DataPartitioner:
    """
    Partitions dataset across federated learning clients.

    Supports multiple partitioning strategies to simulate
    different data heterogeneity scenarios.
    """

    def __init__(
        self,
        targets: torch.Tensor,
        num_clients: int,
        partition_strategy: str = "iid",
        seed: int = 42,
        **strategy_kwargs,
    ):
        """
        Initialize data partitioner.

        Args:
            targets: Tensor of shape (N,) containing all labels
            num_clients: Number of federated clients
            partition_strategy: One of ['iid', 'dirichlet', 'pathological']
            seed: Random seed for reproducibility
            **strategy_kwargs: Additional arguments for specific strategies
                - alpha: Dirichlet concentration parameter (for 'dirichlet')
                - shards_per_client: Number of shards per client (for 'pathological')
                - min_samples_per_client: Minimum samples each client must have
        """
        self.targets = targets
        self.num_clients = num_clients
        self.partition_strategy = partition_strategy
        self.seed = seed
        self.strategy_kwargs = strategy_kwargs

        self.num_samples = len(targets)
        self.num_classes = int(targets.max().item()) + 1

        # Set random seed for reproducibility
        self.rng = np.random.RandomState(seed)

        # Partition the data
        self.client_indices = self._partition()

    def _partition(self) -> Dict[int, List[int]]:
        """
        Partition data according to the selected strategy.

        Returns:
            Dictionary mapping client_id -> list of sample indices
        """
        if self.partition_strategy == "iid":
            return self._partition_iid()
        elif self.partition_strategy == "dirichlet":
            return self._partition_dirichlet()
        elif self.partition_strategy == "pathological":
            return self._partition_pathological()
        else:
            raise ValueError(
                f"Unknown partition strategy: {self.partition_strategy}. "
                f"Available: ['iid', 'dirichlet', 'pathological']"
            )

    def _partition_iid(self) -> Dict[int, List[int]]:
        """
        IID partitioning: random uniform split across clients.

        Each client receives approximately equal number of samples
        with similar class distributions.

        Returns:
            client_id -> sample_indices mapping
        """
        # Shuffle all indices
        indices = self.rng.permutation(self.num_samples).tolist()

        # Split into roughly equal chunks
        samples_per_client = self.num_samples // self.num_clients

        client_indices = {}
        for client_id in range(self.num_clients):
            start = client_id * samples_per_client
            if client_id == self.num_clients - 1:
                # Last client gets remaining samples
                end = self.num_samples
            else:
                end = start + samples_per_client

            client_indices[client_id] = indices[start:end]

        return client_indices

    def _partition_dirichlet(self) -> Dict[int, List[int]]:
        """
        Non-IID partitioning via Dirichlet distribution.

        Uses Dirichlet(α) to control label skew across clients.
        Lower α → more heterogeneous (skewed label distributions).
        Higher α → more homogeneous (closer to IID).

        Implementation follows:
        Li et al. "Federated Optimization in Heterogeneous Networks" (FedProx)

        Returns:
            client_id -> sample_indices mapping
        """
        alpha = self.strategy_kwargs.get('alpha', 0.5)
        min_samples = self.strategy_kwargs.get('min_samples_per_client', 10)

        # Group indices by class
        class_indices = [
            np.where(self.targets.numpy() == c)[0]
            for c in range(self.num_classes)
        ]

        # Shuffle each class's indices
        for indices in class_indices:
            self.rng.shuffle(indices)

        # Initialize client indices
        client_indices = {i: [] for i in range(self.num_clients)}

        # For each class, split using Dirichlet distribution
        for c_idx, indices in enumerate(class_indices):
            # Draw from Dirichlet(α) to get client proportions
            proportions = self.rng.dirichlet(
                alpha=[alpha] * self.num_clients
            )

            # Convert proportions to sample counts
            proportions = (proportions * len(indices)).astype(int)

            # Adjust for rounding errors
            proportions[-1] = len(indices) - proportions[:-1].sum()

            # Assign samples to clients
            start_idx = 0
            for client_id in range(self.num_clients):
                end_idx = start_idx + proportions[client_id]
                client_indices[client_id].extend(
                    indices[start_idx:end_idx].tolist()
                )
                start_idx = end_idx

        # Shuffle each client's indices
        for client_id in client_indices:
            self.rng.shuffle(client_indices[client_id])

        # Ensure minimum samples per client
        client_indices = self._enforce_min_samples(client_indices, min_samples)

        return client_indices

    def _partition_pathological(self) -> Dict[int, List[int]]:
        """
        Pathological non-IID partitioning (FedAvg paper method).

        Sort data by label, divide into shards, assign n shards per client.
        Each shard receives data from a contiguous set of classes.
        Example: With 2 shards per client on MNIST, each client gets data
        from exactly 2 classes.

        From: McMahan et al. "Communication-Efficient Learning of
        Deep Networks from Decentralized Data" (FedAvg)

        Returns:
            client_id -> sample_indices mapping
        """
        shards_per_client = self.strategy_kwargs.get('shards_per_client', 2)

        # Get unique classes
        unique_labels = torch.unique(self.targets).numpy()
        num_classes = len(unique_labels)

        # Ensure reasonable configuration
        if shards_per_client > num_classes:
            raise ValueError(
                f"shards_per_client ({shards_per_client}) cannot exceed "
                f"num_classes ({num_classes})"
            )

        # Total number of shards
        num_shards = self.num_clients * shards_per_client

        # Divide classes among shards (as evenly as possible)
        # Each shard gets a contiguous set of classes
        classes_per_shard = max(1, num_classes // num_shards)

        shards = [[] for _ in range(num_shards)]

        # Assign classes to shards
        for shard_id in range(num_shards):
            # Determine which classes this shard should contain
            start_class_idx = shard_id * classes_per_shard
            if shard_id == num_shards - 1:
                # Last shard gets remaining classes
                end_class_idx = num_classes
            else:
                end_class_idx = start_class_idx + classes_per_shard

            # Collect all samples from these classes
            for class_idx in range(start_class_idx, end_class_idx):
                class_label = unique_labels[class_idx]
                class_mask = self.targets == class_label
                class_indices = torch.where(class_mask)[0].numpy().tolist()
                shards[shard_id].extend(class_indices)

            # Shuffle samples within each shard
            self.rng.shuffle(shards[shard_id])

        # Now randomly assign shards to clients
        # Each client gets shards_per_client consecutive shards
        shard_assignment = self.rng.permutation(num_shards).tolist()

        client_indices = {}
        for client_id in range(self.num_clients):
            client_data = []
            for s in range(shards_per_client):
                shard_idx = shard_assignment[client_id * shards_per_client + s]
                client_data.extend(shards[shard_idx])

            # Final shuffle of client's data
            self.rng.shuffle(client_data)
            client_indices[client_id] = client_data

        return client_indices

    def _enforce_min_samples(
        self,
        client_indices: Dict[int, List[int]],
        min_samples: int,
    ) -> Dict[int, List[int]]:
        """
        Ensure each client has minimum number of samples.

        Redistributes samples from clients with many samples
        to clients with too few.

        Args:
            client_indices: Current partition
            min_samples: Minimum samples per client

        Returns:
            Adjusted partition
        """
        # Find clients below threshold
        below_threshold = [
            cid for cid, indices in client_indices.items()
            if len(indices) < min_samples
        ]

        if not below_threshold:
            return client_indices

        # Find clients above average (can donate samples)
        avg_samples = self.num_samples // self.num_clients
        above_avg = [
            cid for cid, indices in client_indices.items()
            if len(indices) > avg_samples
        ]

        # Redistribute
        for client_id in below_threshold:
            needed = min_samples - len(client_indices[client_id])

            for donor_id in above_avg:
                if needed <= 0:
                    break

                donor_samples = client_indices[donor_id]
                can_donate = len(donor_samples) - avg_samples

                if can_donate > 0:
                    transfer = min(needed, can_donate)
                    # Transfer samples
                    transferred = donor_samples[-transfer:]
                    client_indices[donor_id] = donor_samples[:-transfer]
                    client_indices[client_id].extend(transferred)
                    needed -= transfer

        return client_indices

    def get_client_indices(self, client_id: int) -> List[int]:
        """
        Get sample indices for a specific client.

        Args:
            client_id: Client ID (0 to num_clients-1)

        Returns:
            List of sample indices for this client
        """
        if client_id not in self.client_indices:
            raise ValueError(
                f"Invalid client_id: {client_id}. "
                f"Must be in range [0, {self.num_clients})"
            )
        return self.client_indices[client_id]

    def get_client_statistics(self, client_id: int) -> Dict:
        """
        Get statistics for a specific client's data.

        Args:
            client_id: Client ID

        Returns:
            Dictionary with num_samples and label_distribution
        """
        indices = self.get_client_indices(client_id)
        client_targets = self.targets[indices]

        # Compute class distribution
        class_counts = torch.bincount(
            client_targets,
            minlength=self.num_classes
        )

        return {
            'client_id': client_id,
            'num_samples': len(indices),
            'label_distribution': class_counts.tolist(),
            'unique_labels': (class_counts > 0).sum().item(),
        }

    def get_all_statistics(self) -> Dict:
        """
        Get statistics for all clients.

        Returns:
            Dictionary with overall partition statistics
        """
        client_stats = [
            self.get_client_statistics(i)
            for i in range(self.num_clients)
        ]

        sample_counts = [s['num_samples'] for s in client_stats]
        unique_labels = [s['unique_labels'] for s in client_stats]

        return {
            'num_clients': self.num_clients,
            'total_samples': self.num_samples,
            'partition_strategy': self.partition_strategy,
            'strategy_kwargs': self.strategy_kwargs,
            'samples_per_client': {
                'min': min(sample_counts),
                'max': max(sample_counts),
                'mean': np.mean(sample_counts),
                'std': np.std(sample_counts),
            },
            'unique_labels_per_client': {
                'min': min(unique_labels),
                'max': max(unique_labels),
                'mean': np.mean(unique_labels),
            },
            'client_statistics': client_stats,
        }
