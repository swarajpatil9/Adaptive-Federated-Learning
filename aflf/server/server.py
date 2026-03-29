"""
Federated learning server implementation.

Main server class that coordinates federated learning:
- Global model management
- Client registration
- Round execution
- Aggregation coordination
"""

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..aggregation.aggregation_base import AggregationStrategy
from ..client.client import FederatedClient, TrainingResult
from ..selection.selection_strategy import RandomSelection, SelectionStrategy
from .client_manager import ClientManager
from .logger import ConsoleProgressLogger, ServerLogger
from .metrics_tracker import MetricsTracker, ProgressTracker
from .orchestrator import Orchestrator, RoundResult
from .round_manager import RoundManager, RoundState
from .server_utils import (
    compute_weighted_average_metrics,
    get_model_parameters,
    set_model_parameters,
)

logger = logging.getLogger(__name__)


class FederatedServer:
    """
    Federated learning server.

    Main coordinator for FL training. Manages:
    - Global model
    - Client registration and tracking
    - Round execution and orchestration
    - Aggregation (Phase 6)
    - Training history

    Example:
        >>> from aflf.models import SimpleCNN
        >>> from aflf.selection import RandomSelection
        >>>
        >>> server = FederatedServer(
        ...     model=SimpleCNN(num_classes=10),
        ...     selection_strategy=RandomSelection(seed=42),
        ...     num_clients_per_round=10
        ... )
        >>>
        >>> # Register clients
        >>> for client_id, client in enumerate(clients):
        ...     server.register_client(client_id, dataset_size=600)
        >>>
        >>> # Execute rounds
        >>> for round_num in range(10):
        ...     result = server.execute_round(
        ...         round_num=round_num,
        ...         clients=clients_dict
        ...     )
        >>>
        >>> # Get final model
        >>> global_model = server.get_global_model()
    """

    def __init__(
        self,
        model: nn.Module,
        selection_strategy: Optional[SelectionStrategy] = None,
        aggregation_strategy: Optional[AggregationStrategy] = None,
        num_clients_per_round: int = 10,
        device: str = "cpu",
        server_logger: Optional[ServerLogger] = None,
        metrics_tracker: Optional[MetricsTracker] = None,
        enable_progress_bar: bool = False,
    ):
        """
        Initialize federated server.

        Args:
            model: Global model (will be copied, not modified in place)
            selection_strategy: Strategy for selecting clients (default: RandomSelection)
            aggregation_strategy: Strategy for aggregation (default: None, Phase 6)
            num_clients_per_round: Number of clients per round
            device: Device for global model ('cpu' or 'cuda')
            server_logger: Optional ServerLogger for structured logging
            metrics_tracker: Optional MetricsTracker for metrics export
            enable_progress_bar: Whether to show console progress bar
        """
        # Initialize global model
        self.device = torch.device(device)
        self.global_model = model.to(self.device)

        # Initialize managers
        self.client_manager = ClientManager()
        self.round_manager = RoundManager()

        # Initialize selection strategy
        if selection_strategy is None:
            selection_strategy = RandomSelection()
        self.selection_strategy = selection_strategy

        # Aggregation strategy (Phase 6)
        self.aggregation_strategy = aggregation_strategy

        # Initialize orchestrator
        self.orchestrator = Orchestrator(
            client_manager=self.client_manager,
            round_manager=self.round_manager,
            selection_strategy=self.selection_strategy,
        )

        # Configuration
        self.num_clients_per_round = num_clients_per_round

        # Logging and metrics
        self.server_logger = server_logger
        self.metrics_tracker = metrics_tracker
        self.enable_progress_bar = enable_progress_bar
        self.progress_logger: Optional[ConsoleProgressLogger] = None

        logger.info(
            f"Initialized FederatedServer with selection={self.selection_strategy}"
        )

        # Log server initialization if logger provided
        if self.server_logger:
            self.server_logger.log_server_init(
                num_clients=0,  # Will be updated as clients register
                model_name=model.__class__.__name__,
                selection_strategy=str(self.selection_strategy),
                config={
                    'num_clients_per_round': num_clients_per_round,
                    'device': str(device),
                },
            )

    def register_client(
        self,
        client_id: int,
        dataset_size: int,
        is_available: bool = True,
        resource_score: float = 1.0,
    ) -> None:
        """
        Register a client with the server.

        Args:
            client_id: Unique client identifier
            dataset_size: Number of training samples
            is_available: Initial availability status

        Raises:
            ValueError: If client already registered
        """
        self.client_manager.register_client(
            client_id=client_id,
            dataset_size=dataset_size,
            is_available=is_available,
            resource_score=resource_score,
        )
        logger.info(f"Registered client {client_id} (dataset_size={dataset_size})")

    def execute_round(
        self,
        round_num: int,
        clients: Dict[int, FederatedClient],
        client_train_config: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        Execute a complete FL round.

        This is the main method for running federated learning.

        Workflow:
        1. Select clients (via orchestrator)
        2. Distribute global model (via orchestrator)
        3. Collect training results (via orchestrator)
        4. Aggregate results (Phase 6 - currently returns unaggregated data)
        5. Update global model (Phase 6)
        6. Record round metrics

        Args:
            round_num: Current round number
            clients: Dictionary mapping client_id to FederatedClient

        Returns:
            Dictionary with round results:
                - round_num: Round number
                - num_selected: Number of clients selected
                - num_participating: Number of clients that completed
                - num_failed: Number of clients that failed
                - participation_rate: Fraction of selected clients that participated
                - results: List of TrainingResults (for aggregation in Phase 6)
                - metrics: Aggregated metrics
                - duration: Round duration in seconds

        Raises:
            ValueError: If no clients available
        """
        logger.info(f"Executing round {round_num}")

        # Log round start
        if self.server_logger:
            self.server_logger.log_round_start(
                round_num=round_num,
                num_selected=self.num_clients_per_round,
            )

        # Execute round via orchestrator
        round_result = self.orchestrator.execute_round(
            round_num=round_num,
            clients=clients,
            global_model=self.global_model,
            num_clients_per_round=self.num_clients_per_round,
            client_train_config=client_train_config,
        )

        # Compute metrics
        if round_result.results:
            metrics = compute_weighted_average_metrics(round_result.results)
        else:
            logger.warning(f"Round {round_num}: No successful results")
            metrics = {}

        # Finalize round
        round_state = self.round_manager.end_round(metrics=metrics)

        # Log summary
        logger.info(
            f"Round {round_num} complete: "
            f"{len(round_result.results)}/{len(round_result.round_state.selected_clients)} "
            f"clients participated"
        )
        if metrics:
            logger.info(
                f"  Avg train loss: {metrics.get('avg_train_loss', 0):.4f}, "
                f"Avg train accuracy: {metrics.get('avg_train_accuracy', 0):.4f}"
            )

        # Log round end
        if self.server_logger:
            self.server_logger.log_round_end(round_state, metrics)

        # Track metrics
        if self.metrics_tracker:
            self.metrics_tracker.record_round(round_state, metrics)

            # Record individual client results
            for result in round_result.results:
                self.metrics_tracker.record_client_result(
                    round_num=round_num,
                    client_id=result.client_id,
                    result_dict=result.to_dict(),
                )

        # Update progress bar
        if self.progress_logger:
            self.progress_logger.update_round(
                round_num=round_num,
                metrics=metrics,
                num_participating=len(round_result.results),
                num_selected=len(round_result.round_state.selected_clients),
            )

        return {
            'round_num': round_num,
            'num_selected': len(round_result.round_state.selected_clients),
            'num_participating': len(round_result.results),
            'num_failed': len(round_result.failed_clients),
            'participation_rate': round_result.round_state.participation_rate,
            'selected_clients': round_result.round_state.selected_clients.copy(),
            'selection_scores': round_result.round_state.selection_scores.copy(),
            'selection_reasoning': round_result.round_state.selection_reasoning.copy(),
            'selection_policy': round_result.round_state.selection_policy,
            'participating_clients': round_state.participating_clients.copy(),
            'failed_clients': round_result.failed_clients.copy(),
            'results': round_result.results,  # For aggregation in Phase 6
            'metrics': metrics,
            'duration': round_result.round_state.duration,
        }

    def get_global_model(self) -> nn.Module:
        """
        Get current global model.

        Returns:
            Global model (reference, not copy)
        """
        return self.global_model

    def get_global_parameters(self) -> OrderedDict[str, torch.Tensor]:
        """
        Get global model parameters.

        Returns:
            OrderedDict of parameters
        """
        return get_model_parameters(self.global_model)

    def set_global_parameters(
        self, parameters: OrderedDict[str, torch.Tensor]
    ) -> None:
        """
        Update global model parameters.

        This will be used in Phase 6 after aggregation.

        Args:
            parameters: New global parameters
        """
        set_model_parameters(self.global_model, parameters)

    def get_round_history(self) -> List[RoundState]:
        """
        Get history of all rounds.

        Returns:
            List of RoundState objects
        """
        return self.round_manager.get_round_history()

    def get_client_summary(self) -> Dict:
        """
        Get summary of all clients.

        Returns:
            Dictionary with client statistics
        """
        return self.client_manager.get_summary_stats()

    def get_round_summary(self) -> Dict:
        """
        Get summary of all rounds.

        Returns:
            Dictionary with round statistics
        """
        return self.round_manager.get_summary_stats()

    def run_training(
        self,
        num_rounds: int,
        clients: Dict[int, FederatedClient],
        show_progress: bool = True,
        client_train_config: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        Run federated training for multiple rounds with progress tracking.

        Convenience method that handles progress bar and logging.

        Args:
            num_rounds: Number of rounds to train
            clients: Dictionary mapping client_id to FederatedClient
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with training summary

        Example:
            >>> summary = server.run_training(
            ...     num_rounds=50,
            ...     clients=clients_dict,
            ...     show_progress=True
            ... )
        """
        import time

        start_time = time.time()

        # Initialize progress logger if requested
        if show_progress and not self.enable_progress_bar:
            self.progress_logger = ConsoleProgressLogger(total_rounds=num_rounds)

        # Run rounds
        for round_num in range(num_rounds):
            self.execute_round(
                round_num=round_num,
                clients=clients,
                client_train_config=client_train_config,
            )

        # Finish progress bar
        if self.progress_logger:
            self.progress_logger.finish()
            self.progress_logger = None

        # Compute summary
        total_time = time.time() - start_time

        # Log training complete
        if self.server_logger:
            final_metrics = {}
            if self.round_manager.get_num_rounds() > 0:
                last_round = self.round_manager.get_round_history()[-1]
                final_metrics = last_round.metrics

            self.server_logger.log_training_complete(
                total_rounds=num_rounds,
                total_time=total_time,
                final_metrics=final_metrics,
            )

        # Export metrics
        if self.metrics_tracker:
            summary = self.metrics_tracker.compute_summary()
            return summary

        return {
            'num_rounds': num_rounds,
            'total_time': total_time,
            'avg_time_per_round': total_time / num_rounds,
        }

    def export_metrics(
        self, export_json: bool = True, export_csv: bool = True
    ) -> Dict[str, str]:
        """
        Export tracked metrics to files.

        Args:
            export_json: Whether to export JSON
            export_csv: Whether to export CSV

        Returns:
            Dictionary with paths to exported files

        Example:
            >>> paths = server.export_metrics()
            >>> print(paths['json'])  # Path to JSON file
        """
        if not self.metrics_tracker:
            print("No metrics tracker configured")
            return {}

        paths = {}

        if export_json:
            paths['json'] = self.metrics_tracker.export_json()

        if export_csv:
            paths['csv'] = self.metrics_tracker.export_csv()
            paths['client_csv'] = self.metrics_tracker.export_client_csv()

        return paths

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FederatedServer("
            f"clients={self.client_manager.get_num_clients()}, "
            f"rounds={self.round_manager.get_num_rounds()})"
        )
