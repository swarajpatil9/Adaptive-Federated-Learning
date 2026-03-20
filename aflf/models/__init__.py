"""
Model module for federated learning.

Provides:
- BaseModel abstract class
- CNN implementations (SimpleCNN, CNN, CNNLarge)
- Model factory for config-driven creation
- Utilities for parameter handling and model inspection
"""

from .base import BaseModel
from .cnn import CNN, CNNLarge, SimpleCNN
from .factory import create_model, get_model_info, list_available_models
from .utils import (
    count_parameters,
    get_model_size_mb,
    initialize_model,
    load_model_weights,
    save_model_weights,
)

__all__ = [
    # Base class
    'BaseModel',

    # CNN models
    'SimpleCNN',
    'CNN',
    'CNNLarge',

    # Factory
    'create_model',
    'list_available_models',
    'get_model_info',

    # Utilities
    'count_parameters',
    'get_model_size_mb',
    'initialize_model',
    'save_model_weights',
    'load_model_weights',
]
