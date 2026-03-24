"""
Federated learning server module.

Main API:
    FederatedServer - Main server class
    ClientManager - Client registry
    RoundManager - Round tracking
    Orchestrator - Round execution
    ServerLogger - Structured logging
    MetricsTracker - Metrics export
    ConsoleProgressLogger - Progress bar
    ProgressTracker - Convergence tracking

Example:
    >>> from aflf.server import FederatedServer, ServerLogger, MetricsTracker
    >>> from aflf.models import SimpleCNN
    >>>
    >>> # Initialize with logging
    >>> logger = ServerLogger(experiment_name="mnist_fedavg")
    >>> tracker = MetricsTracker(experiment_name="mnist_fedavg")
    >>>
    >>> server = FederatedServer(
    ...     model=SimpleCNN(num_classes=10),
    ...     num_clients_per_round=10,
    ...     server_logger=logger,
    ...     metrics_tracker=tracker
    ... )
    >>> server.register_client(client_id=0, dataset_size=600)
"""

from .client_manager import ClientManager, ClientMetadata
from .logger import ConsoleProgressLogger, ServerLogger
from .metrics_tracker import MetricsTracker, ProgressTracker
from .orchestrator import Orchestrator, RoundResult
from .round_manager import RoundManager, RoundState
from .server import FederatedServer
from .server_utils import (
    compute_weighted_average_metrics,
    format_round_summary,
    get_model_parameters,
    set_model_parameters,
)

__all__ = [
    'FederatedServer',
    'ClientManager',
    'ClientMetadata',
    'RoundManager',
    'RoundState',
    'Orchestrator',
    'RoundResult',
    'ServerLogger',
    'ConsoleProgressLogger',
    'MetricsTracker',
    'ProgressTracker',
    'get_model_parameters',
    'set_model_parameters',
    'compute_weighted_average_metrics',
    'format_round_summary',
]
