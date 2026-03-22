"""
Local training logic for federated learning clients.

Implements the core SGD training loop executed on each client.
Designed to be stateless and reusable across different FL algorithms.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .client_utils import get_criterion, get_device, get_optimizer
from .metrics import MetricsTracker


class LocalTrainer:
    """
    Stateless local trainer for federated learning clients.

    Executes SGD-based training on client data without maintaining
    state between calls. This design allows the same trainer to be
    used across multiple FL rounds and different clients.

    Key design:
    - Stateless: No persistent model or optimizer storage
    - Reusable: Can train different models with different configs
    - Tested: Used in FedAvg, FedProx, and other FL algorithms

    Example:
        >>> trainer = LocalTrainer(device='cuda', verbose=True)
        >>> model = SimpleCNN(num_classes=10)
        >>> results = trainer.train(
        >>>     model=model,
        >>>     train_loader=train_loader,
        >>>     epochs=5,
        >>>     lr=0.01,
        >>> )
        >>> print(f"Final loss: {results['train_loss']:.4f}")
    """

    def __init__(
        self,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize local trainer.

        Args:
            device: Training device ('cpu', 'cuda', 'mps')
                    If None, auto-detects best available device
            verbose: If True, print training progress
        """
        self.device = get_device(device)
        self.verbose = verbose

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        epochs: int = 1,
        lr: float = 0.01,
        optimizer_name: str = "sgd",
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        criterion_name: str = "cross_entropy",
        val_loader: Optional[DataLoader] = None,
    ) -> dict:
        """
        Execute local training.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            epochs: Number of local epochs
            lr: Learning rate
            optimizer_name: Optimizer ('sgd', 'adam', 'adamw')
            momentum: Momentum for SGD
            weight_decay: L2 regularization
            criterion_name: Loss function ('cross_entropy', 'mse', 'bce')
            val_loader: Optional validation data loader

        Returns:
            Dictionary with training results:
                - train_loss: Final training loss
                - train_accuracy: Final training accuracy
                - val_loss: Validation loss (if val_loader provided)
                - val_accuracy: Validation accuracy (if val_loader provided)
                - num_samples: Total training samples

        Example:
            >>> results = trainer.train(
            >>>     model=model,
            >>>     train_loader=train_loader,
            >>>     epochs=5,
            >>>     lr=0.01,
            >>>     optimizer_name='sgd',
            >>>     momentum=0.9,
            >>> )
        """
        # Move model to device
        model = model.to(self.device)
        model.train()

        # Create optimizer and criterion
        optimizer = get_optimizer(
            model=model,
            optimizer_name=optimizer_name,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        criterion = get_criterion(criterion_name)

        # Training loop
        for epoch in range(epochs):
            epoch_metrics = self._train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                epoch=epoch,
                total_epochs=epochs,
            )

            if self.verbose:
                print(
                    f"  Epoch [{epoch+1}/{epochs}] - "
                    f"Loss: {epoch_metrics['loss']:.4f}, "
                    f"Acc: {epoch_metrics['accuracy']:.4f}"
                )

        # Get final training metrics
        final_train_metrics = epoch_metrics

        # Validation if provided
        val_loss = None
        val_accuracy = None
        if val_loader is not None:
            val_metrics = self.validate(model, val_loader, criterion)
            val_loss = val_metrics['loss']
            val_accuracy = val_metrics['accuracy']

            if self.verbose:
                print(
                    f"  Validation - "
                    f"Loss: {val_loss:.4f}, "
                    f"Acc: {val_accuracy:.4f}"
                )

        return {
            'train_loss': final_train_metrics['loss'],
            'train_accuracy': final_train_metrics['accuracy'],
            'num_samples': final_train_metrics['num_samples'],
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        }

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epoch: int,
        total_epochs: int,
    ) -> dict:
        """
        Train for one epoch.

        Args:
            model: Model to train
            train_loader: Training data
            optimizer: Optimizer
            criterion: Loss function
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of epochs

        Returns:
            Epoch metrics dictionary
        """
        model.train()
        tracker = MetricsTracker()

        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data = data.to(self.device)
            target = target.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Compute accuracy
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == target).sum().item()
            accuracy = correct / len(target)

            # Update metrics
            tracker.update(
                loss=loss.item(),
                accuracy=accuracy,
                num_samples=len(target),
            )

        return tracker.get_metrics()

    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
    ) -> dict:
        """
        Run validation.

        Args:
            model: Model to validate
            val_loader: Validation data loader
            criterion: Loss function (creates default if None)

        Returns:
            Validation metrics dictionary:
                - loss: Average validation loss
                - accuracy: Average validation accuracy
                - num_samples: Total validation samples

        Example:
            >>> metrics = trainer.validate(model, val_loader)
            >>> print(f"Val acc: {metrics['accuracy']:.4f}")
        """
        model.eval()

        if criterion is None:
            criterion = get_criterion('cross_entropy')

        tracker = MetricsTracker()

        for data, target in val_loader:
            # Move data to device
            data = data.to(self.device)
            target = target.to(self.device)

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Compute accuracy
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == target).sum().item()
            accuracy = correct / len(target)

            # Update metrics
            tracker.update(
                loss=loss.item(),
                accuracy=accuracy,
                num_samples=len(target),
            )

        return tracker.get_metrics()

    def __repr__(self) -> str:
        """String representation."""
        return f"LocalTrainer(device={self.device})"
