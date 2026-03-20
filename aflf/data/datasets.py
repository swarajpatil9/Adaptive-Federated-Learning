"""
Concrete implementations of federated datasets.

Includes standard FL benchmarks:
- MNIST
- CIFAR-10
- CIFAR-100
- FEMNIST (planned)
"""

from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .base import FederatedDataset


class MNISTDataset(FederatedDataset):
    """
    MNIST dataset for federated learning.

    - 60,000 training images, 10,000 test images
    - 28x28 grayscale images
    - 10 classes (digits 0-9)
    - Commonly used in FL papers as baseline
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = True,
        augment: bool = False,
    ):
        super().__init__(root, train, download)
        self.augment = augment
        self._num_classes = 10

    def load_data(self) -> Dataset:
        """Load MNIST dataset."""
        transform = self.get_transforms()
        dataset = datasets.MNIST(
            root=self.root,
            train=self.train,
            download=self.download,
            transform=transform,
        )
        return dataset

    def get_transforms(self) -> transforms.Compose:
        """Get MNIST preprocessing transforms."""
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
        ]

        # Optional augmentation for training
        if self.train and self.augment:
            transform_list = [
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ] + transform_list

        return transforms.Compose(transform_list)

    @property
    def num_classes(self) -> int:
        return self._num_classes


class CIFAR10Dataset(FederatedDataset):
    """
    CIFAR-10 dataset for federated learning.

    - 50,000 training images, 10,000 test images
    - 32x32 RGB images
    - 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
    - More challenging than MNIST, standard FL benchmark
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = True,
        augment: bool = True,
    ):
        super().__init__(root, train, download)
        self.augment = augment
        self._num_classes = 10

    def load_data(self) -> Dataset:
        """Load CIFAR-10 dataset."""
        transform = self.get_transforms()
        dataset = datasets.CIFAR10(
            root=self.root,
            train=self.train,
            download=self.download,
            transform=transform,
        )
        return dataset

    def get_transforms(self) -> transforms.Compose:
        """Get CIFAR-10 preprocessing transforms."""
        if self.train and self.augment:
            # Standard CIFAR-10 augmentation from ResNet paper
            transform_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010),
                ),
            ]
        else:
            # Test transforms: no augmentation
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010),
                ),
            ]

        return transforms.Compose(transform_list)

    @property
    def num_classes(self) -> int:
        return self._num_classes


class CIFAR100Dataset(FederatedDataset):
    """
    CIFAR-100 dataset for federated learning.

    - 50,000 training images, 10,000 test images
    - 32x32 RGB images
    - 100 fine-grained classes
    - More challenging than CIFAR-10, useful for testing with many classes
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = True,
        augment: bool = True,
    ):
        super().__init__(root, train, download)
        self.augment = augment
        self._num_classes = 100

    def load_data(self) -> Dataset:
        """Load CIFAR-100 dataset."""
        transform = self.get_transforms()
        dataset = datasets.CIFAR100(
            root=self.root,
            train=self.train,
            download=self.download,
            transform=transform,
        )
        return dataset

    def get_transforms(self) -> transforms.Compose:
        """Get CIFAR-100 preprocessing transforms."""
        if self.train and self.augment:
            transform_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5071, 0.4867, 0.4408),
                    std=(0.2675, 0.2565, 0.2761),
                ),
            ]
        else:
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5071, 0.4867, 0.4408),
                    std=(0.2675, 0.2565, 0.2761),
                ),
            ]

        return transforms.Compose(transform_list)

    @property
    def num_classes(self) -> int:
        return self._num_classes


# Factory function for easy dataset creation
def create_dataset(
    name: str,
    root: str,
    train: bool = True,
    download: bool = True,
    augment: Optional[bool] = None,
) -> FederatedDataset:
    """
    Factory function to create datasets by name.

    Args:
        name: Dataset name ('mnist', 'cifar10', 'cifar100')
        root: Root directory for data storage
        train: If True, load training set
        download: If True, download if not present
        augment: If True, apply data augmentation (None = use dataset default)

    Returns:
        FederatedDataset instance

    Raises:
        ValueError: If dataset name is not recognized
    """
    name = name.lower()

    dataset_map = {
        'mnist': MNISTDataset,
        'cifar10': CIFAR10Dataset,
        'cifar100': CIFAR100Dataset,
    }

    if name not in dataset_map:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available: {list(dataset_map.keys())}"
        )

    dataset_class = dataset_map[name]

    # Use default augmentation if not specified
    kwargs = {'root': root, 'train': train, 'download': download}
    if augment is not None:
        kwargs['augment'] = augment

    return dataset_class(**kwargs)
