"""
Client module for federated learning.

Provides all components needed for local client training:
- FederatedClient: Main client interface
- LocalTrainer: Training execution engine
- MetricsTracker: Metrics accumulation
- TrainingResult: Standard result format
- Utilities: Weight management, device selection, reproducibility

Example:
    >>> from aflf.client import FederatedClient
    >>> client = FederatedClient(
    >>>     client_id=0,
    >>>     train_loader=train_loader,
    >>>     val_loader=val_loader,
    >>> )
    >>> result = client.train(
    >>>     global_model=model,
    >>>     config={'epochs': 5, 'lr': 0.01}
    >>> )
"""

# Main classes
from .client import FederatedClient, TrainingResult
from .trainer import LocalTrainer
from .metrics import MetricsTracker

# Simulation features
from .simulation import (
    SimulatedFederatedClient,
    ClientSimulationConfig,
    ClientFailureException,
    ClientUnavailableException,
    create_heterogeneous_clients,
    compute_dataset_imbalance_metrics,
)

# Utilities
from .client_utils import (
    get_model_weights,
    set_model_weights,
    get_device,
    set_reproducibility,
    count_model_parameters,
    count_trainable_parameters,
    get_optimizer,
    get_criterion,
)

__all__ = [
    # Main client interface
    'FederatedClient',
    'TrainingResult',

    # Training components
    'LocalTrainer',
    'MetricsTracker',

    # Simulation
    'SimulatedFederatedClient',
    'ClientSimulationConfig',
    'ClientFailureException',
    'ClientUnavailableException',
    'create_heterogeneous_clients',
    'compute_dataset_imbalance_metrics',

    # Weight management
    'get_model_weights',
    'set_model_weights',

    # Device and reproducibility
    'get_device',
    'set_reproducibility',

    # Model utilities
    'count_model_parameters',
    'count_trainable_parameters',

    # Training utilities
    'get_optimizer',
    'get_criterion',
]
