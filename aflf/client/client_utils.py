"""
Utility functions for federated learning clients.

Provides weight management, device selection, and reproducibility utilities
that are commonly needed in client training.
"""

import random
from collections import OrderedDict
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn


def get_model_weights(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    """
    Extract model weights as CPU tensors.

    This is the standard format for transmitting weights in FL systems.
    Weights are detached and moved to CPU to avoid GPU memory issues.

    Args:
        model: PyTorch model

    Returns:
        OrderedDict mapping parameter names to CPU tensors

    Example:
        >>> model = SimpleCNN(num_classes=10)
        >>> weights = get_model_weights(model)
        >>> print(weights.keys())
        odict_keys(['conv1.weight', 'conv1.bias', ...])
    """
    return OrderedDict(
        (name, param.detach().cpu().clone())
        for name, param in model.state_dict().items()
    )


def set_model_weights(
    model: nn.Module,
    weights: Union[OrderedDict[str, torch.Tensor], Dict[str, torch.Tensor]],
) -> None:
    """
    Load weights into model.

    Handles device placement automatically. If model is on GPU,
    weights will be moved to GPU during load_state_dict().

    Args:
        model: PyTorch model
        weights: State dict with model parameters

    Example:
        >>> model = SimpleCNN(num_classes=10)
        >>> global_weights = get_model_weights(global_model)
        >>> set_model_weights(model, global_weights)
    """
    model.load_state_dict(weights, strict=True)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get PyTorch device for training.

    Auto-detects available hardware if device is None:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon GPU)
    3. CPU (fallback)

    Args:
        device: Device string ('cpu', 'cuda', 'mps', 'cuda:0')
                If None, auto-detects best available device

    Returns:
        torch.device object

    Example:
        >>> device = get_device()  # Auto-detect
        >>> print(device)
        device(type='cuda', index=0)

        >>> device = get_device('cpu')  # Force CPU
        >>> print(device)
        device(type='cpu')
    """
    if device is not None:
        return torch.device(device)

    # Auto-detect
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def set_reproducibility(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducible training.

    Sets seeds for:
    - Python random
    - NumPy random
    - PyTorch CPU and GPU

    Args:
        seed: Random seed
        deterministic: If True, use deterministic algorithms (slower but reproducible)

    Note:
        Deterministic mode may reduce performance but ensures
        exact reproducibility across runs.

    Example:
        >>> set_reproducibility(42)
        >>> # Now all random operations are deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For PyTorch >= 1.12
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True, warn_only=True)


def count_model_parameters(model: nn.Module) -> int:
    """
    Count total number of parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Total number of parameters (trainable + non-trainable)

    Example:
        >>> model = SimpleCNN(num_classes=10)
        >>> count = count_model_parameters(model)
        >>> print(f"Total parameters: {count:,}")
        Total parameters: 62,006
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count number of trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters

    Example:
        >>> model = SimpleCNN(num_classes=10)
        >>> # Freeze first layer
        >>> for param in model.conv1.parameters():
        >>>     param.requires_grad = False
        >>> trainable = count_trainable_parameters(model)
        >>> print(f"Trainable parameters: {trainable:,}")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "sgd",
    lr: float = 0.01,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Create optimizer for model.

    Supports common optimizers used in FL research:
    - SGD: Standard in FedAvg
    - Adam: Used in some adaptive FL methods
    - AdamW: Better weight decay handling

    Args:
        model: PyTorch model
        optimizer_name: Optimizer name ('sgd', 'adam', 'adamw')
        lr: Learning rate
        momentum: Momentum (for SGD)
        weight_decay: L2 regularization
        **kwargs: Additional optimizer arguments

    Returns:
        Configured optimizer

    Example:
        >>> model = SimpleCNN(num_classes=10)
        >>> optimizer = get_optimizer(model, 'sgd', lr=0.01, momentum=0.9)
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs,
        )
    elif optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs,
        )
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. "
            f"Supported: 'sgd', 'adam', 'adamw'"
        )


def get_criterion(criterion_name: str = "cross_entropy") -> nn.Module:
    """
    Create loss criterion.

    Args:
        criterion_name: Loss function name
            - 'cross_entropy': For classification
            - 'mse': For regression
            - 'bce': Binary cross-entropy

    Returns:
        Loss criterion

    Example:
        >>> criterion = get_criterion('cross_entropy')
        >>> loss = criterion(outputs, targets)
    """
    criterion_name = criterion_name.lower()

    if criterion_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif criterion_name == "mse":
        return nn.MSELoss()
    elif criterion_name == "bce":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(
            f"Unknown criterion: {criterion_name}. "
            f"Supported: 'cross_entropy', 'mse', 'bce'"
        )
