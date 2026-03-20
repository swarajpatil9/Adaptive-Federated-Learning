"""
CNN model implementations for federated learning.

Provides three model scales:
- SimpleCNN: Lightweight for MNIST (62K parameters)
- CNN: Medium for CIFAR-10 (122K parameters)
- CNNLarge: Larger for CIFAR-10/100 (1.2M parameters)

All models follow the LeNet-style architecture:
conv → pool → conv → pool → fc → fc

Design choices for FL:
- Small parameter count (communication efficiency)
- Fast training on CPU (edge device constraint)
- Proven architectures from FL literature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel


class SimpleCNN(BaseModel):
    """
    Simple CNN for MNIST.

    Architecture:
        Conv(1→32) → ReLU → MaxPool
        Conv(32→64) → ReLU → MaxPool
        Flatten
        FC(64*7*7 → 128) → ReLU → Dropout
        FC(128 → num_classes)

    Parameters: ~62K
    Input: (batch, 1, 28, 28)
    Output: (batch, num_classes)

    Used in: FedAvg paper, FedProx for MNIST experiments

    Args:
        num_classes: Number of output classes (default: 10 for MNIST)
        dropout_rate: Dropout probability (default: 0.5)

    Example:
        >>> model = SimpleCNN(num_classes=10)
        >>> x = torch.randn(32, 1, 28, 28)
        >>> logits = model(x)
        >>> print(logits.shape)
        torch.Size([32, 10])
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super().__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After 2 pooling layers: 28 → 14 → 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, 1, 28, 28)

        Returns:
            Logits (batch, num_classes)
        """
        # Conv block 1
        x = self.conv1(x)           # (batch, 32, 28, 28)
        x = F.relu(x)
        x = self.pool(x)            # (batch, 32, 14, 14)

        # Conv block 2
        x = self.conv2(x)           # (batch, 64, 14, 14)
        x = F.relu(x)
        x = self.pool(x)            # (batch, 64, 7, 7)

        # Flatten
        x = x.view(x.size(0), -1)   # (batch, 64*7*7)

        # FC layers
        x = self.fc1(x)             # (batch, 128)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)             # (batch, num_classes)

        return x


class CNN(BaseModel):
    """
    Medium CNN for CIFAR-10.

    Architecture:
        Conv(3→32) → ReLU → Conv(32→32) → ReLU → MaxPool
        Conv(32→64) → ReLU → Conv(64→64) → ReLU → MaxPool
        Flatten
        FC(64*8*8 → 128) → ReLU → Dropout
        FC(128 → num_classes)

    Parameters: ~122K
    Input: (batch, 3, 32, 32)
    Output: (batch, num_classes)

    Used in: FedProx, SCAFFOLD for CIFAR-10 experiments

    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        dropout_rate: Dropout probability (default: 0.5)
        use_batch_norm: Whether to use batch normalization (default: False)

    Example:
        >>> model = CNN(num_classes=10)
        >>> x = torch.randn(32, 3, 32, 32)
        >>> logits = model(x)
        >>> print(logits.shape)
        torch.Size([32, 10])
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Convolutional block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()

        # Convolutional block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After 2 pooling layers: 32 → 16 → 8
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, 3, 32, 32)

        Returns:
            Logits (batch, num_classes)
        """
        # Conv block 1
        x = self.conv1(x)           # (batch, 32, 32, 32)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)           # (batch, 32, 32, 32)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)            # (batch, 32, 16, 16)

        # Conv block 2
        x = self.conv3(x)           # (batch, 64, 16, 16)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)           # (batch, 64, 16, 16)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)            # (batch, 64, 8, 8)

        # Flatten
        x = x.view(x.size(0), -1)   # (batch, 64*8*8)

        # FC layers
        x = self.fc1(x)             # (batch, 128)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)             # (batch, num_classes)

        return x


class CNNLarge(BaseModel):
    """
    Larger CNN for CIFAR-10/100.

    Architecture:
        Conv(3→64) → ReLU → Conv(64→64) → ReLU → MaxPool
        Conv(64→128) → ReLU → Conv(128→128) → ReLU → MaxPool
        Conv(128→256) → ReLU → Conv(256→256) → ReLU → MaxPool
        Flatten
        FC(256*4*4 → 512) → ReLU → Dropout
        FC(512 → num_classes)

    Parameters: ~1.2M
    Input: (batch, 3, 32, 32)
    Output: (batch, num_classes)

    Note: This is larger than typical FL models. Use for:
    - Studying communication-accuracy trade-offs
    - CIFAR-100 (more classes need more capacity)
    - Comparing with lightweight models

    Args:
        num_classes: Number of output classes (10 or 100)
        dropout_rate: Dropout probability (default: 0.5)
        use_batch_norm: Whether to use batch normalization (default: True)

    Example:
        >>> model = CNNLarge(num_classes=100)  # CIFAR-100
        >>> x = torch.randn(32, 3, 32, 32)
        >>> logits = model(x)
        >>> print(logits.shape)
        torch.Size([32, 100])
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Convolutional block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()

        # Convolutional block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()

        # Convolutional block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256) if use_batch_norm else nn.Identity()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256) if use_batch_norm else nn.Identity()

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After 3 pooling layers: 32 → 16 → 8 → 4
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, 3, 32, 32)

        Returns:
            Logits (batch, num_classes)
        """
        # Conv block 1
        x = self.conv1(x)           # (batch, 64, 32, 32)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)           # (batch, 64, 32, 32)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)            # (batch, 64, 16, 16)

        # Conv block 2
        x = self.conv3(x)           # (batch, 128, 16, 16)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)           # (batch, 128, 16, 16)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)            # (batch, 128, 8, 8)

        # Conv block 3
        x = self.conv5(x)           # (batch, 256, 8, 8)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x)           # (batch, 256, 8, 8)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.pool(x)            # (batch, 256, 4, 4)

        # Flatten
        x = x.view(x.size(0), -1)   # (batch, 256*4*4)

        # FC layers
        x = self.fc1(x)             # (batch, 512)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)             # (batch, num_classes)

        return x
