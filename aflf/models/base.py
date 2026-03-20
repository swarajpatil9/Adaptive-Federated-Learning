"""
Base model class for federated learning.

Defines the interface that all FL models must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for all federated learning models.

    This class defines the standard interface for models in FL:
    - Parameter extraction/loading for communication
    - Model metadata (size, num parameters)
    - Device management

    All FL models should inherit from this class and implement
    the required methods.
    """

    def __init__(self):
        """Initialize base model."""
        super().__init__()
        self._device = torch.device("cpu")

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor (logits)
        """
        pass

    def get_parameters(self) -> List[np.ndarray]:
        """
        Extract model parameters as list of numpy arrays.

        This is used for federated communication. Parameters are
        extracted in a deterministic order (state_dict order).

        Returns:
            List of parameter arrays (weights and biases)

        Example:
            >>> model = SimpleCNN()
            >>> params = model.get_parameters()
            >>> print(len(params))  # Number of parameter tensors
            10
        """
        return [param.detach().cpu().numpy() for param in self.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Load parameters from list of numpy arrays.

        This is used to load aggregated parameters from the server.
        Parameters must be in the same order as get_parameters().

        Args:
            parameters: List of parameter arrays

        Raises:
            ValueError: If number of parameters doesn't match

        Example:
            >>> model = SimpleCNN()
            >>> params = model.get_parameters()
            >>> # Simulate aggregation
            >>> agg_params = [p * 0.9 for p in params]
            >>> model.set_parameters(agg_params)
        """
        param_list = list(self.parameters())

        if len(parameters) != len(param_list):
            raise ValueError(
                f"Parameter count mismatch: got {len(parameters)} "
                f"but model has {len(param_list)} parameters"
            )

        with torch.no_grad():
            for param_tensor, new_param in zip(param_list, parameters):
                param_tensor.copy_(torch.from_numpy(new_param))

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        Count total number of parameters.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Total parameter count

        Example:
            >>> model = SimpleCNN()
            >>> print(model.get_num_parameters())
            62006
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_model_size(self) -> int:
        """
        Calculate model size in bytes.

        This is useful for estimating communication cost in FL.
        Assumes float32 parameters (4 bytes each).

        Returns:
            Model size in bytes

        Example:
            >>> model = SimpleCNN()
            >>> size_bytes = model.get_model_size()
            >>> size_mb = size_bytes / (1024 ** 2)
            >>> print(f"Model size: {size_mb:.2f} MB")
            Model size: 0.24 MB
        """
        num_params = self.get_num_parameters()
        return num_params * 4  # 4 bytes per float32

    def get_model_size_mb(self) -> float:
        """
        Get model size in megabytes.

        Returns:
            Model size in MB
        """
        return self.get_model_size() / (1024 ** 2)

    def to_device(self, device: torch.device) -> 'BaseModel':
        """
        Move model to device.

        Args:
            device: Target device (cpu, cuda, mps)

        Returns:
            Self (for chaining)
        """
        self._device = device
        return self.to(device)

    def get_device(self) -> torch.device:
        """
        Get current device.

        Returns:
            Current device
        """
        return self._device

    def get_layer_shapes(self) -> List[Tuple[str, Tuple[int, ...]]]:
        """
        Get shapes of all parameter tensors.

        Returns:
            List of (layer_name, shape) tuples

        Example:
            >>> model = SimpleCNN()
            >>> for name, shape in model.get_layer_shapes():
            ...     print(f"{name}: {shape}")
            conv1.weight: (32, 1, 3, 3)
            conv1.bias: (32,)
            ...
        """
        return [(name, tuple(param.shape)) for name, param in self.named_parameters()]

    def summary(self, input_shape: Tuple[int, ...] = None) -> str:
        """
        Generate model summary string.

        Args:
            input_shape: Input tensor shape (including batch dimension)

        Returns:
            Summary string with layer info and parameter counts
        """
        lines = []
        lines.append("="*80)
        lines.append(f"Model: {self.__class__.__name__}")
        lines.append("="*80)

        # Layer information
        lines.append(f"{'Layer':<30} {'Parameters':<20} {'Shape':<30}")
        lines.append("-"*80)

        total_params = 0
        trainable_params = 0

        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params

            shape_str = str(tuple(param.shape))
            lines.append(f"{name:<30} {num_params:<20,} {shape_str:<30}")

        lines.append("="*80)
        lines.append(f"Total parameters: {total_params:,}")
        lines.append(f"Trainable parameters: {trainable_params:,}")
        lines.append(f"Non-trainable parameters: {total_params - trainable_params:,}")
        lines.append(f"Model size: {self.get_model_size_mb():.4f} MB")
        lines.append("="*80)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        num_params = self.get_num_parameters()
        size_mb = self.get_model_size_mb()
        return (
            f"{self.__class__.__name__}("
            f"parameters={num_params:,}, "
            f"size={size_mb:.2f}MB"
            ")"
        )
