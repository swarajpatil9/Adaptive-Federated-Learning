"""
Abstract base class for federated datasets.

All datasets must inherit from FederatedDataset and implement
the required interface methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FederatedDataset(ABC):
    """
    Abstract base class for federated learning datasets.

    All FL datasets must implement this interface to ensure
    compatibility with the partitioning and data loading pipeline.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = True,
    ):
        """
        Initialize federated dataset.

        Args:
            root: Root directory for dataset storage
            train: If True, load training set; else test set
            download: If True, download dataset if not present
        """
        self.root = root
        self.train = train
        self.download = download
        self._dataset: Optional[Dataset] = None
        self._num_classes: Optional[int] = None

    @abstractmethod
    def load_data(self) -> Dataset:
        """
        Load and return the PyTorch Dataset.

        This method should handle downloading, extracting, and
        preparing the dataset for use.

        Returns:
            PyTorch Dataset object
        """
        pass

    @abstractmethod
    def get_transforms(self) -> transforms.Compose:
        """
        Get the preprocessing transforms for this dataset.

        Returns:
            Composed torchvision transforms
        """
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """
        Get the number of classes in the dataset.

        Returns:
            Number of classification classes
        """
        pass

    @property
    def dataset(self) -> Dataset:
        """Get the loaded dataset (lazy loading)."""
        if self._dataset is None:
            self._dataset = self.load_data()
        return self._dataset

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.dataset)

    def get_targets(self) -> torch.Tensor:
        """
        Extract all labels/targets from the dataset.

        Returns:
            Tensor of shape (N,) containing all labels
        """
        dataset = self.dataset

        # Handle different dataset formats
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
            if isinstance(targets, list):
                targets = torch.tensor(targets)
            elif not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets)
            return targets
        elif hasattr(dataset, 'labels'):
            labels = dataset.labels
            if isinstance(labels, list):
                labels = torch.tensor(labels)
            elif not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)
            return labels
        else:
            # Fallback: iterate through dataset
            targets = []
            for i in range(len(dataset)):
                _, target = dataset[i]
                targets.append(target)
            return torch.tensor(targets)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute dataset statistics.

        Returns:
            Dictionary with statistics (num_samples, num_classes, class_distribution)
        """
        targets = self.get_targets()
        class_counts = torch.bincount(targets, minlength=self.num_classes)

        return {
            'num_samples': len(self),
            'num_classes': self.num_classes,
            'class_distribution': class_counts.tolist(),
            'min_class_count': class_counts.min().item(),
            'max_class_count': class_counts.max().item(),
            'mean_class_count': class_counts.float().mean().item(),
        }
