"""
Data module for federated learning.

Provides:
- Dataset loading and preprocessing
- Client data partitioning (IID and Non-IID)
- DataLoader creation for clients
- Reproducibility utilities
"""

from .base import FederatedDataset
from .datasets import (
    CIFAR10Dataset,
    CIFAR100Dataset,
    MNISTDataset,
    create_dataset,
)
from .federated_data import FederatedDataModule
from .partitioner import DataPartitioner
from .transforms import (
    get_cifar10_transforms,
    get_cifar100_transforms,
    get_mnist_transforms,
    get_transforms,
)
from .utils import (
    compute_class_weights,
    compute_label_distribution,
    get_dataset_info,
    measure_data_heterogeneity,
    print_dataset_info,
    validate_partition,
)

__all__ = [
    # Base classes
    'FederatedDataset',
    'DataPartitioner',
    'FederatedDataModule',

    # Concrete datasets
    'MNISTDataset',
    'CIFAR10Dataset',
    'CIFAR100Dataset',
    'create_dataset',

    # Transforms
    'get_mnist_transforms',
    'get_cifar10_transforms',
    'get_cifar100_transforms',
    'get_transforms',

    # Utilities
    'compute_class_weights',
    'compute_label_distribution',
    'measure_data_heterogeneity',
    'validate_partition',
    'get_dataset_info',
    'print_dataset_info',
]
