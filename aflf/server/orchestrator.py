"""
Server orchestrator for federated learning.

Coordinates round execution:
- Client selection
- Model distribution
- Result collection
- Failure handling
"""

import logging
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch.nn as nn

from ..client.client import FederatedClient, TrainingResult
from ..client.simulation import (
    ClientFailureException,
    ClientUnavailableException,
)
from ..selection.selection_strategy import SelectionResult, SelectionStrategy
from .client_manager import ClientManager
from .round_manager import RoundManager, RoundState
from .server_utils import get_model_parameters

logger = logging.getLogger(__name__)


@dataclass
class RoundResult:
    """
    Result of orchestrating a single FL round.

    Contains all information needed by server:
    - Successful training results
    - Failed clients
    - Round metrics

    Attributes:
        round_num: Round number
        results: List of successful TrainingResults
        failed_clients: List of client IDs that failed
        failure_reasons: Mapping from client_id to failure reason
        round_state: Complete round state from RoundManager
    """

    round_num: int
    results: List[TrainingResult]
    failed_clients: List[int]
    failure_reasons: Dict[int, str]
    round_state: RoundState


class Orchestrator:
    """
    Orchestrator for federated learning rounds.

    Responsibilities:
    - Select clients using SelectionStrategy
    - Distribute global model to selected clients
    - Collect training results with failure handling
    - Track round state via RoundManager
    - Log round events

    The orchestrator is stateless - all state is managed by
    ClientManager and RoundManager.

    Example:
        >>> orchestrator = Orchestrator(
        ...     client_manager=manager,
        ...     round_manager=rounds,
        ...     selection_strategy=RandomSelection(seed=42)
        ... )
        >>> result = orchestrator.execute_round(
        ...     round_num=0,
        ...     clients=clients_dict,
        ...     global_model=model,
        ...     num_clients_per_round=10
        ... )
    """

    def __init__(
        self,
        client_manager: ClientManager,
        round_manager: RoundManager,
        selection_strategy: SelectionStrategy,
    ):
        """
        Initialize orchestrator.

        Args:
            client_manager: Client registry and metadata
            round_manager: Round state tracker
            selection_strategy: Strategy for selecting clients
        """
        self.client_manager = client_manager
        self.round_manager = round_manager
        self.selection_strategy = selection_strategy

    def execute_round(
        self,
        round_num: int,
        clients: Dict[int, FederatedClient],
        global_model: nn.Module,
        num_clients_per_round: int,
        client_train_config: Optional[Dict[str, Any]] = None,
    ) -> RoundResult:
        """
        Execute a complete FL round.

        Workflow:
        1. Select clients
        2. Start round tracking
        3. Distribute model to selected clients
        4. Collect results (with failure handling)
        5. Record participations/drops
        6. Finalize round

        Args:
            round_num: Current round number
            clients: Dictionary mapping client_id to FederatedClient
            global_model: Global model to distribute
            num_clients_per_round: Number of clients to select

        Returns:
            RoundResult with all round information

        Raises:
            ValueError: If no clients available or selection fails
        """
        logger.info(f"Starting round {round_num}")

        # Step 1: Select clients
        available_clients = self.client_manager.get_available_clients()
        if not available_clients:
            raise ValueError("No available clients for round")

        # Get client metadata for selection strategy
        client_metadata = {
            cid: self.client_manager.get_client_metadata(cid)
            for cid in available_clients
        }

        # Select clients
        num_to_select = min(num_clients_per_round, len(available_clients))
        selection_result = self.selection_strategy.select_with_details(
            available_clients=available_clients,
            num_clients=num_to_select,
            round_num=round_num,
            client_metadata=client_metadata,
        )
        if isinstance(selection_result, SelectionResult):
            selected_clients = selection_result.selected_client_ids
        else:
            # Safety fallback for custom legacy strategies.
            selected_clients = self.selection_strategy.select(
                available_clients=available_clients,
                num_clients=num_to_select,
                round_num=round_num,
                client_metadata=client_metadata,
            )
            selection_result = SelectionResult(selected_client_ids=selected_clients)

        self.client_manager.record_selection(
            round_num=round_num,
            selected_client_ids=selected_clients,
            available_clients=available_clients,
            scores=selection_result.client_scores,
        )

        logger.info(
            f"Round {round_num}: Selected {len(selected_clients)} clients from "
            f"{len(available_clients)} available"
        )

        # Step 2: Start round tracking
        try:
            round_state = self.round_manager.start_round(
                round_num=round_num,
                selected_clients=selected_clients,
                selection_scores=selection_result.client_scores,
                selection_reasoning=selection_result.selection_reasoning,
                selection_policy=selection_result.policy_name,
            )
        except TypeError:
            # Backward compatibility with RoundManager versions that do not
            # expose selection metadata fields.
            round_state = self.round_manager.start_round(
                round_num=round_num,
                selected_clients=selected_clients,
            )

        # Step 3: Distribute model and collect results
        results, failed_clients, failure_reasons = self._distribute_and_collect(
            round_num=round_num,
            selected_clients=selected_clients,
            clients=clients,
            global_model=global_model,
            client_train_config=client_train_config,
        )

        # Step 4: Record participations and drops
        for result in results:
            self.round_manager.record_participation(result.client_id)
            self.client_manager.update_from_result(result, round_num)

        for client_id, reason in failure_reasons.items():
            self.round_manager.record_drop(client_id, reason)
            self.client_manager.record_failure(client_id)

        logger.info(
            f"Round {round_num}: {len(results)} clients completed, "
            f"{len(failed_clients)} clients failed"
        )

        return RoundResult(
            round_num=round_num,
            results=results,
            failed_clients=failed_clients,
            failure_reasons=failure_reasons,
            round_state=round_state,
        )

    def _distribute_and_collect(
        self,
        round_num: int,
        selected_clients: List[int],
        clients: Dict[int, FederatedClient],
        global_model: nn.Module,
        client_train_config: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[TrainingResult], List[int], Dict[int, str]]:
        """
        Distribute model and collect results from selected clients.

        Handles failures gracefully - failing clients don't stop the round.

        Args:
            round_num: Current round number
            selected_clients: List of selected client IDs
            clients: Dictionary of all clients
            global_model: Global model to distribute

        Returns:
            Tuple of (successful_results, failed_client_ids, failure_reasons)
        """
        global_weights = get_model_parameters(global_model)

        results = []
        failed_clients = []
        failure_reasons = {}

        for client_id in selected_clients:
            if client_id not in clients:
                logger.warning(
                    f"Round {round_num}: Client {client_id} not in client dict, skipping"
                )
                failed_clients.append(client_id)
                failure_reasons[client_id] = "client_not_found"
                continue

            client = clients[client_id]

            try:
                # Distribute global model if client exposes explicit setter.
                if hasattr(client, 'set_parameters') and callable(client.set_parameters):
                    client.set_parameters(global_weights)

                # Train client with broad API compatibility.
                result = self._train_client(
                    client=client,
                    global_model=global_model,
                    client_train_config=client_train_config,
                )

                # Validate result
                if result.client_id != client_id:
                    logger.warning(
                        f"Round {round_num}: Client {client_id} returned result with "
                        f"mismatched ID {result.client_id}"
                    )
                    failed_clients.append(client_id)
                    failure_reasons[client_id] = "mismatched_client_id"
                    continue

                results.append(result)
                logger.debug(
                    f"Round {round_num}: Client {client_id} completed successfully "
                    f"(loss={result.train_loss:.4f}, acc={result.train_accuracy:.4f})"
                )

            except ClientFailureException as e:
                logger.warning(
                    f"Round {round_num}: Client {client_id} failed during training: {e}"
                )
                failed_clients.append(client_id)
                failure_reasons[client_id] = "training_failure"

            except ClientUnavailableException as e:
                logger.warning(
                    f"Round {round_num}: Client {client_id} unavailable: {e}"
                )
                failed_clients.append(client_id)
                failure_reasons[client_id] = "unavailable"

            except Exception as e:
                logger.error(
                    f"Round {round_num}: Client {client_id} raised unexpected error: {e}"
                )
                failed_clients.append(client_id)
                failure_reasons[client_id] = f"unexpected_error: {type(e).__name__}"

        return results, failed_clients, failure_reasons

    def _train_client(
        self,
        client: FederatedClient,
        global_model: nn.Module,
        client_train_config: Optional[Dict[str, Any]] = None,
    ) -> TrainingResult:
        """
        Train client while supporting multiple client interfaces.

        Supported patterns:
        - client.train(global_model=..., config=...)
        - client.train(global_model, config)
        - client.train(config=...)
        - client.train()
        """
        train_fn = client.train

        try:
            signature = inspect.signature(train_fn)
            param_names = list(signature.parameters.keys())
        except (TypeError, ValueError):
            param_names = []

        if 'global_model' in param_names:
            return train_fn(global_model=global_model, config=client_train_config)

        if len(param_names) >= 2:
            return train_fn(global_model, client_train_config)

        if 'config' in param_names:
            return train_fn(config=client_train_config)

        return train_fn()
