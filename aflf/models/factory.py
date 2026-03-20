"""
Model factory for creating models from configuration.

Enables config-driven model selection and hyperparameter search.
"""

from typing import Any, Dict, Optional

import torch.nn as nn

from .cnn import CNN, CNNLarge, SimpleCNN


# Registry of available models
MODEL_REGISTRY = {
    'simple_cnn': SimpleCNN,
    'cnn': CNN,
    'cnn_large': CNNLarge,
}


def create_model(
    model_name: str,
    num_classes: Optional[int] = None,
    **kwargs: Any,
) -> nn.Module:
    """
    Create a model by name.

    This is the main factory function for creating models from
    configuration files or command-line arguments.

    Args:
        model_name: Name of the model (see MODEL_REGISTRY)
        num_classes: Number of output classes (overrides model default)
        **kwargs: Additional model-specific arguments

    Returns:
        Instantiated model

    Raises:
        ValueError: If model_name is not recognized

    Example:
        >>> # Create SimpleCNN for MNIST
        >>> model = create_model('simple_cnn', num_classes=10)

        >>> # Create CNN for CIFAR-10 with batch norm
        >>> model = create_model('cnn', num_classes=10, use_batch_norm=True)

        >>> # Create CNNLarge for CIFAR-100
        >>> model = create_model('cnn_large', num_classes=100, dropout_rate=0.3)
    """
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {available}"
        )

    model_class = MODEL_REGISTRY[model_name]

    # Build kwargs
    model_kwargs = dict(kwargs)
    if num_classes is not None:
        model_kwargs['num_classes'] = num_classes

    return model_class(**model_kwargs)


def list_available_models() -> Dict[str, type]:
    """
    Get dictionary of all available models.

    Returns:
        Dictionary mapping model names to classes

    Example:
        >>> models = list_available_models()
        >>> print(models.keys())
        dict_keys(['simple_cnn', 'cnn', 'cnn_large'])
    """
    return dict(MODEL_REGISTRY)


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with model info (class, docstring, etc.)

    Raises:
        ValueError: If model_name is not recognized

    Example:
        >>> info = get_model_info('simple_cnn')
        >>> print(info['description'])
        Simple CNN for MNIST.
        >>> print(info['default_params'])
        62006
    """
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {available}"
        )

    model_class = MODEL_REGISTRY[model_name]

    # Create a default instance to get parameter count
    # Use defaults appropriate for each model
    if model_name == 'simple_cnn':
        default_instance = model_class(num_classes=10)
        default_dataset = 'MNIST'
    elif model_name == 'cnn':
        default_instance = model_class(num_classes=10)
        default_dataset = 'CIFAR-10'
    elif model_name == 'cnn_large':
        default_instance = model_class(num_classes=10)
        default_dataset = 'CIFAR-10/100'
    else:
        default_instance = model_class()
        default_dataset = 'Unknown'

    return {
        'name': model_name,
        'class': model_class.__name__,
        'description': model_class.__doc__.split('\n')[1].strip() if model_class.__doc__ else '',
        'default_params': default_instance.get_num_parameters(),
        'default_size_mb': default_instance.get_model_size_mb(),
        'default_dataset': default_dataset,
    }


def print_model_catalog():
    """
    Print catalog of all available models.

    Example:
        >>> print_model_catalog()
        ================================================================================
        AVAILABLE MODELS
        ================================================================================
        Name         Class        Parameters   Size (MB)   Dataset
        --------------------------------------------------------------------------------
        simple_cnn   SimpleCNN    62,006       0.24        MNIST
        cnn          CNN          122,570      0.47        CIFAR-10
        cnn_large    CNNLarge     1,234,826    4.71        CIFAR-10/100
        ================================================================================
    """
    print("="*80)
    print("AVAILABLE MODELS")
    print("="*80)
    print(f"{'Name':<13} {'Class':<13} {'Parameters':<13} {'Size (MB)':<12} {'Dataset':<15}")
    print("-"*80)

    for model_name in MODEL_REGISTRY:
        info = get_model_info(model_name)
        print(
            f"{info['name']:<13} "
            f"{info['class']:<13} "
            f"{info['default_params']:<13,} "
            f"{info['default_size_mb']:<12.2f} "
            f"{info['default_dataset']:<15}"
        )

    print("="*80)
