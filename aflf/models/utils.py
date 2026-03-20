"""
Utility functions for model management.

Provides tools for:
- Parameter counting and model size calculation
- Model initialization (deterministic seeding)
- Weight saving/loading
- Model inspection and comparison
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn


def count_parameters(
    model: nn.Module,
    trainable_only: bool = False,
) -> int:
    """
    Count total number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Total parameter count

    Example:
        >>> from aflf.models import SimpleCNN
        >>> model = SimpleCNN()
        >>> total = count_parameters(model)
        >>> trainable = count_parameters(model, trainable_only=True)
        >>> print(f"Total: {total:,}, Trainable: {trainable:,}")
        Total: 62,006, Trainable: 62,006
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate model size in megabytes.

    Assumes float32 parameters (4 bytes each).

    Args:
        model: PyTorch model

    Returns:
        Model size in MB

    Example:
        >>> from aflf.models import SimpleCNN
        >>> model = SimpleCNN()
        >>> size = get_model_size_mb(model)
        >>> print(f"Model size: {size:.2f} MB")
        Model size: 0.24 MB
    """
    num_params = count_parameters(model)
    size_bytes = num_params * 4  # 4 bytes per float32
    return size_bytes / (1024 ** 2)


def initialize_model(
    model: nn.Module,
    seed: Optional[int] = None,
    method: str = 'kaiming',
) -> nn.Module:
    """
    Initialize model weights deterministically.

    This ensures reproducibility in federated learning experiments.
    All clients start with the same initial weights.

    Args:
        model: PyTorch model to initialize
        seed: Random seed (if None, no seeding)
        method: Initialization method ('kaiming', 'xavier', 'normal')

    Returns:
        Model with initialized weights

    Example:
        >>> from aflf.models import SimpleCNN
        >>> model = SimpleCNN()
        >>> model = initialize_model(model, seed=42, method='kaiming')
        >>> # All runs with seed=42 will have identical initial weights
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    for name, param in model.named_parameters():
        if 'weight' in name:
            if method == 'kaiming':
                if len(param.shape) >= 2:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.normal_(param, mean=0.0, std=0.01)
            elif method == 'xavier':
                if len(param.shape) >= 2:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.normal_(param, mean=0.0, std=0.01)
            elif method == 'normal':
                nn.init.normal_(param, mean=0.0, std=0.01)
            else:
                raise ValueError(f"Unknown initialization method: {method}")
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)

    return model


def save_model_weights(
    model: nn.Module,
    save_path: Union[str, Path],
    include_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save model weights to file.

    Args:
        model: PyTorch model
        save_path: Path to save checkpoint
        include_optimizer: Whether to save optimizer state
        optimizer: Optimizer (required if include_optimizer=True)
        metadata: Additional metadata to save (e.g., epoch, accuracy)

    Example:
        >>> from aflf.models import SimpleCNN
        >>> model = SimpleCNN()
        >>> save_model_weights(model, 'checkpoints/model.pt')
        >>> # With metadata
        >>> save_model_weights(
        ...     model,
        ...     'checkpoints/model.pt',
        ...     metadata={'epoch': 10, 'accuracy': 0.95}
        ... )
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }

    if include_optimizer:
        if optimizer is None:
            raise ValueError("Optimizer must be provided if include_optimizer=True")
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if metadata is not None:
        checkpoint['metadata'] = metadata

    torch.save(checkpoint, save_path)


def load_model_weights(
    model: nn.Module,
    load_path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
) -> Dict:
    """
    Load model weights from file.

    Args:
        model: PyTorch model
        load_path: Path to checkpoint
        optimizer: Optimizer (will load state if found in checkpoint)
        strict: Whether to strictly enforce state dict keys match

    Returns:
        Dictionary with metadata (if any was saved)

    Example:
        >>> from aflf.models import SimpleCNN
        >>> model = SimpleCNN()
        >>> metadata = load_model_weights(model, 'checkpoints/model.pt')
        >>> print(metadata)
        {'epoch': 10, 'accuracy': 0.95}
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    checkpoint = torch.load(load_path, map_location='cpu')

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    # Load optimizer state if present
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Return metadata
    return checkpoint.get('metadata', {})


def compare_models(
    models: Dict[str, nn.Module],
    input_shape: tuple = (1, 3, 32, 32),
) -> None:
    """
    Compare multiple models side-by-side.

    Args:
        models: Dictionary mapping model names to instances
        input_shape: Input tensor shape for forward pass test

    Example:
        >>> from aflf.models import SimpleCNN, CNN, CNNLarge
        >>> models = {
        ...     'SimpleCNN': SimpleCNN(num_classes=10),
        ...     'CNN': CNN(num_classes=10),
        ...     'CNNLarge': CNNLarge(num_classes=10),
        ... }
        >>> compare_models(models)
    """
    print("="*100)
    print("MODEL COMPARISON")
    print("="*100)
    print(f"{'Model':<15} {'Parameters':<15} {'Size (MB)':<15} {'Output Shape':<20} {'Status':<10}")
    print("-"*100)

    for name, model in models.items():
        num_params = count_parameters(model)
        size_mb = get_model_size_mb(model)

        # Test forward pass
        try:
            with torch.no_grad():
                dummy_input = torch.randn(input_shape)
                output = model(dummy_input)
                output_shape = str(tuple(output.shape))
                status = "✓"
        except Exception as e:
            output_shape = f"Error: {str(e)[:30]}"
            status = "✗"

        print(
            f"{name:<15} "
            f"{num_params:<15,} "
            f"{size_mb:<15.2f} "
            f"{output_shape:<20} "
            f"{status:<10}"
        )

    print("="*100)


def get_parameter_histogram(
    model: nn.Module,
    num_bins: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Get histogram of parameter values.

    Useful for analyzing weight distributions and detecting
    gradient issues.

    Args:
        model: PyTorch model
        num_bins: Number of histogram bins

    Returns:
        Dictionary with histogram data for each layer

    Example:
        >>> from aflf.models import SimpleCNN
        >>> model = SimpleCNN()
        >>> histograms = get_parameter_histogram(model)
        >>> # Check weight distribution
        >>> conv1_hist = histograms['conv1.weight']
    """
    histograms = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            values = param.detach().cpu().numpy().flatten()
            hist, bin_edges = np.histogram(values, bins=num_bins)
            histograms[name] = {
                'hist': hist,
                'bin_edges': bin_edges,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            }

    return histograms


def freeze_layers(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
    freeze_all: bool = False,
) -> nn.Module:
    """
    Freeze specific layers or all layers.

    Useful for transfer learning or selective training in FL.

    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze (None = freeze all)
        freeze_all: If True, freeze all layers

    Returns:
        Model with frozen layers

    Example:
        >>> from aflf.models import CNN
        >>> model = CNN()
        >>> # Freeze only convolutional layers
        >>> model = freeze_layers(model, ['conv1', 'conv2', 'conv3', 'conv4'])
        >>> # Or freeze all
        >>> model = freeze_layers(model, freeze_all=True)
    """
    if freeze_all:
        for param in model.parameters():
            param.requires_grad = False
    elif layer_names is not None:
        for name, param in model.named_parameters():
            for layer_name in layer_names:
                if name.startswith(layer_name):
                    param.requires_grad = False
                    break

    return model


def print_trainable_parameters(model: nn.Module) -> None:
    """
    Print which parameters are trainable.

    Args:
        model: PyTorch model

    Example:
        >>> from aflf.models import SimpleCNN
        >>> model = SimpleCNN()
        >>> from aflf.models.utils import freeze_layers
        >>> model = freeze_layers(model, ['conv1'])
        >>> print_trainable_parameters(model)
    """
    print("="*80)
    print("TRAINABLE PARAMETERS")
    print("="*80)
    print(f"{'Layer':<40} {'Trainable':<12} {'Parameters':<15}")
    print("-"*80)

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        trainable = "Yes" if param.requires_grad else "No"
        if param.requires_grad:
            trainable_params += num_params

        print(f"{name:<40} {trainable:<12} {num_params:<15,}")

    print("-"*80)
    print(f"{'Total':<40} {'':<12} {total_params:<15,}")
    print(f"{'Trainable':<40} {'':<12} {trainable_params:<15,}")
    print(f"{'Frozen':<40} {'':<12} {total_params - trainable_params:<15,}")
    print("="*80)
