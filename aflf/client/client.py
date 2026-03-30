"""
Federated learning client implementation.

Provides the main client interface that orchestrates:
- Receiving global model
- Local training execution
- Weight extraction and return
- Metrics reporting
"""

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .client_utils import get_model_weights, set_model_weights
from .trainer import LocalTrainer
from ..privacy.dp_mechanism import PrivacyEngine
from ..privacy.privacy_config import PrivacyConfig


@dataclass
class TrainingResult:
    """
    Result of client training.

    This is the standard return format for FL clients. Contains
    all information needed by the server for aggregation.

    Attributes:
        client_id: Unique client identifier
        weights: Updated model weights (OrderedDict)
        num_samples: Number of training samples
        train_loss: Final training loss
        train_accuracy: Final training accuracy
        val_loss: Validation loss (None if no validation)
        val_accuracy: Validation accuracy (None if no validation)
        training_time: Training duration in seconds

    Example:
        >>> result = TrainingResult(
        >>>     client_id=0,
        >>>     weights=updated_weights,
        >>>     num_samples=600,
        >>>     train_loss=0.52,
        >>>     train_accuracy=0.85,
        >>>     val_loss=None,
        >>>     val_accuracy=None,
        >>>     training_time=12.5,
        >>> )
    """

    client_id: int
    weights: OrderedDict[str, torch.Tensor]
    num_samples: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float]
    val_accuracy: Optional[float]
    training_time: float
    privacy_enabled: bool = False
    privacy_overhead_time: float = 0.0
    privacy_metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """
        Convert result to dictionary.

        Useful for logging and serialization.
        Note: weights are excluded for brevity.

        Returns:
            Dictionary with all metadata (no weights)
        """
        return {
            'client_id': self.client_id,
            'num_samples': self.num_samples,
            'train_loss': self.train_loss,
            'train_accuracy': self.train_accuracy,
            'val_loss': self.val_loss,
            'val_accuracy': self.val_accuracy,
            'training_time': self.training_time,
            'privacy_enabled': self.privacy_enabled,
            'privacy_overhead_time': self.privacy_overhead_time,
            'privacy_metadata': self.privacy_metadata or {},
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TrainingResult("
            f"client_id={self.client_id}, "
            f"samples={self.num_samples}, "
            f"train_loss={self.train_loss:.4f}, "
            f"train_acc={self.train_accuracy:.4f}, "
            f"time={self.training_time:.2f}s)"
        )


class FederatedClient:
    """
    Federated learning client.

    Represents a single client in a federated learning system.
    Handles local training on private data and returns model updates.

    Key responsibilities:
    1. Maintain client identity (client_id)
    2. Manage local data (train/val)
    3. Execute local training using LocalTrainer
    4. Extract and return weight updates
    5. Track and report metrics

    Design principles:
    - Stateless training: Each train() call is independent
    - No server dependency: Client doesn't know about aggregation
    - Configurable: All hyperparameters passed at train time
    - Testable: Can be tested without full FL system

    Example:
        >>> # Create client with local data
        >>> client = FederatedClient(
        >>>     client_id=0,
        >>>     train_loader=client_0_train,
        >>>     val_loader=client_0_val,
        >>> )
        >>>
        >>> # Receive global model and train
        >>> result = client.train(
        >>>     global_model=global_model,
        >>>     config={
        >>>         'epochs': 5,
        >>>         'lr': 0.01,
        >>>         'optimizer': 'sgd',
        >>>     }
        >>> )
        >>>
        >>> # Send result back to server
        >>> print(result)
        >>> # Server will use result.weights for aggregation
    """

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize federated client.

        Args:
            client_id: Unique identifier for this client
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation
            epochs: Optional default local epochs (backward-compatible)
            device: Training device ('cpu', 'cuda', 'mps')
            verbose: If True, print training progress
        """
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.default_epochs = epochs
        self.verbose = verbose

        # Create local trainer
        self.trainer = LocalTrainer(device=device, verbose=verbose)

        # Compute dataset size (for weighted aggregation)
        self.num_train_samples = len(train_loader.dataset)
        self.num_val_samples = len(val_loader.dataset) if val_loader else 0

    def train(
        self,
        global_model: nn.Module,
        config: Optional[Dict] = None,
    ) -> TrainingResult:
        """
        Execute local training.

        This is the main method called by the FL server/orchestrator.
        Clones the global model, trains it locally, and returns updates.

        Args:
            global_model: Global model from server
            config: Training configuration with keys:
                - epochs (int): Local epochs (default: 1)
                - lr (float): Learning rate (default: 0.01)
                - optimizer (str): Optimizer name (default: 'sgd')
                - momentum (float): Momentum for SGD (default: 0.0)
                - weight_decay (float): L2 regularization (default: 0.0)
                - criterion (str): Loss function (default: 'cross_entropy')

        Returns:
            TrainingResult with updated weights and metrics

        Example:
            >>> result = client.train(
            >>>     global_model=model,
            >>>     config={
            >>>         'epochs': 5,
            >>>         'lr': 0.01,
            >>>         'optimizer': 'sgd',
            >>>         'momentum': 0.9,
            >>>     }
            >>> )
        """
        if config is None:
            config = {}

        # Extract config (with defaults)
        epochs = config.get('epochs', self.default_epochs if self.default_epochs is not None else 1)
        lr = config.get('lr', 0.01)
        optimizer_name = config.get('optimizer', 'sgd')
        momentum = config.get('momentum', 0.0)
        weight_decay = config.get('weight_decay', 0.0)
        criterion_name = config.get('criterion', 'cross_entropy')

        if self.verbose:
            print(f"\n[Client {self.client_id}] Starting training...")
            print(f"  Config: {config}")
            print(f"  Train samples: {self.num_train_samples}")

        # Clone global model for local training
        # This ensures we don't modify the global model
        local_model = self._clone_model(global_model)
        initial_weights = get_model_weights(local_model)

        # Track training time
        start_time = time.time()

        # Execute local training
        train_metrics = self.trainer.train(
            model=local_model,
            train_loader=self.train_loader,
            epochs=epochs,
            lr=lr,
            optimizer_name=optimizer_name,
            momentum=momentum,
            weight_decay=weight_decay,
            criterion_name=criterion_name,
            val_loader=self.val_loader,
        )

        training_time = time.time() - start_time

        # Extract updated weights
        updated_weights = get_model_weights(local_model)

        # Apply optional privacy protection at the client update boundary.
        privacy_config = PrivacyConfig.from_dict(config.get('privacy', config))
        privacy_engine = PrivacyEngine(privacy_config)
        privacy_result = privacy_engine.protect_weights(
            global_weights=initial_weights,
            local_weights=updated_weights,
        )
        merged_privacy_metadata = {
            **privacy_result.metadata,
            'secure_aggregation': {
                'secure_aggregation_enabled': privacy_config.secure_aggregation_enabled,
                'masking_applied': False,
                'protocol': 'preparation_only',
            },
        }

        # Create result
        result = TrainingResult(
            client_id=self.client_id,
            weights=privacy_result.protected_weights,
            num_samples=train_metrics['num_samples'],
            train_loss=train_metrics['train_loss'],
            train_accuracy=train_metrics['train_accuracy'],
            val_loss=train_metrics['val_loss'],
            val_accuracy=train_metrics['val_accuracy'],
            training_time=training_time,
            privacy_enabled=bool(privacy_result.metadata.get('privacy_enabled', False)),
            privacy_overhead_time=float(privacy_result.processing_time),
            privacy_metadata=merged_privacy_metadata,
        )

        if self.verbose:
            print(f"\n[Client {self.client_id}] Training complete!")
            print(f"  Train Loss: {result.train_loss:.4f}")
            print(f"  Train Acc: {result.train_accuracy:.4f}")
            if result.val_loss is not None:
                print(f"  Val Loss: {result.val_loss:.4f}")
                print(f"  Val Acc: {result.val_accuracy:.4f}")
            print(f"  Time: {result.training_time:.2f}s")

        return result

    def _clone_model(self, model: nn.Module) -> nn.Module:
        """
        Create a deep copy of the model for local training.

        Args:
            model: Model to clone

        Returns:
            Independent copy of the model
        """
        # Get model class and create new instance
        model_class = type(model)

        # Create new instance with same architecture
        # This works because our models have get_config() method
        if hasattr(model, 'get_config'):
            config = model.get_config()
            local_model = model_class(**config)
        else:
            # Fallback: create empty instance and copy weights
            local_model = model_class()

        # Copy weights
        local_model.load_state_dict(model.state_dict())

        return local_model

    @property
    def num_samples(self) -> int:
        """Get number of training samples."""
        return self.num_train_samples

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FederatedClient("
            f"id={self.client_id}, "
            f"train_samples={self.num_train_samples}, "
            f"val_samples={self.num_val_samples})"
        )
