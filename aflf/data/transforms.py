"""
Standard preprocessing and augmentation transforms for FL datasets.

Provides pre-configured transform pipelines for common datasets,
following best practices from literature.
"""

from typing import Optional

from torchvision import transforms


def get_mnist_transforms(
    train: bool = True,
    augment: bool = False,
) -> transforms.Compose:
    """
    Get standard MNIST transforms.

    Args:
        train: If True, return training transforms; else test transforms
        augment: If True, apply data augmentation

    Returns:
        Composed transforms
    """
    transform_list = []

    if train and augment:
        # Light augmentation for MNIST
        transform_list.extend([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    return transforms.Compose(transform_list)


def get_cifar10_transforms(
    train: bool = True,
    augment: bool = True,
) -> transforms.Compose:
    """
    Get standard CIFAR-10 transforms.

    Follows the augmentation strategy from:
    He et al. "Deep Residual Learning for Image Recognition"

    Args:
        train: If True, return training transforms; else test transforms
        augment: If True, apply data augmentation

    Returns:
        Composed transforms
    """
    if train and augment:
        # Standard CIFAR-10 augmentation
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


def get_cifar100_transforms(
    train: bool = True,
    augment: bool = True,
) -> transforms.Compose:
    """
    Get standard CIFAR-100 transforms.

    Args:
        train: If True, return training transforms; else test transforms
        augment: If True, apply data augmentation

    Returns:
        Composed transforms
    """
    if train and augment:
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


def get_transforms(
    dataset_name: str,
    train: bool = True,
    augment: Optional[bool] = None,
) -> transforms.Compose:
    """
    Factory function to get transforms for any dataset.

    Args:
        dataset_name: Name of the dataset
        train: If True, return training transforms
        augment: If True, apply augmentation (None = use defaults)

    Returns:
        Composed transforms

    Raises:
        ValueError: If dataset name is not recognized
    """
    dataset_name = dataset_name.lower()

    # Default augmentation behavior
    if augment is None:
        augment = train  # Augment training by default, not test

    if dataset_name == 'mnist':
        return get_mnist_transforms(train=train, augment=augment)
    elif dataset_name == 'cifar10':
        return get_cifar10_transforms(train=train, augment=augment)
    elif dataset_name == 'cifar100':
        return get_cifar100_transforms(train=train, augment=augment)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
