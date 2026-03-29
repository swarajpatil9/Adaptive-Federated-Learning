"""
Client manager for federated learning server.

Manages client registration, metadata tracking, and state queries.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..client.client import TrainingResult


@dataclass
class ClientMetadata:
    """
    Metadata tracked per client.

    Attributes:
        client_id: Unique client identifier
        dataset_size: Number of training samples
        is_available: Whether client is currently available
        last_accuracy: Most recent training accuracy (None if not trained yet)
        last_loss: Most recent training loss
        participation_count: Number of rounds participated
        failure_count: Number of times client failed
        total_training_time: Cumulative training time (seconds)
        last_round_participated: Last round number where client participated
    """

    client_id: int
    dataset_size: int
    is_available: bool = True
    last_accuracy: Optional[float] = None
    last_loss: Optional[float] = None
    last_performance: Optional[float] = None
    last_score: Optional[float] = None
    participation_count: int = 0
    selection_count: int = 0
    failure_count: int = 0
    total_training_time: float = 0.0
    average_training_time: float = 0.0
    selection_history: List[int] = field(default_factory=list)
    skipped_rounds: int = 0
    resource_score: float = 1.0
    last_selected_round: Optional[int] = None
    last_round_participated: Optional[int] = None


class ClientManager:
    """
    Manages client registry and metadata for federated server.

    Responsibilities:
    - Register clients with dataset sizes
    - Track client availability
    - Update client statistics after training
    - Query available/all clients
    - Track participation history

    Example:
        >>> manager = ClientManager()
        >>> manager.register_client(client_id=0, dataset_size=600)
        >>> manager.register_client(client_id=1, dataset_size=550)
        >>> available = manager.get_available_clients()
        >>> print(len(available))  # 2
    """

    def __init__(self):
        """Initialize client manager."""
        self._clients: Dict[int, ClientMetadata] = {}

    def register_client(
        self,
        client_id: int,
        dataset_size: int,
        is_available: bool = True,
        resource_score: float = 1.0,
    ) -> None:
        """
        Register a new client.

        Args:
            client_id: Unique client identifier
            dataset_size: Number of training samples client has
            is_available: Initial availability status

        Raises:
            ValueError: If client_id already registered
        """
        if client_id in self._clients:
            raise ValueError(f"Client {client_id} already registered")

        self._clients[client_id] = ClientMetadata(
            client_id=client_id,
            dataset_size=dataset_size,
            is_available=is_available,
            resource_score=resource_score,
        )

    def update_from_result(self, result: TrainingResult, round_num: int) -> None:
        """
        Update client metadata from training result.

        Args:
            result: Training result from client
            round_num: Current round number

        Raises:
            ValueError: If client not registered
        """
        client_id = result.client_id
        if client_id not in self._clients:
            raise ValueError(f"Client {client_id} not registered")

        metadata = self._clients[client_id]
        metadata.last_accuracy = result.train_accuracy
        metadata.last_loss = result.train_loss
        metadata.last_performance = result.train_accuracy
        metadata.participation_count += 1
        metadata.total_training_time += result.training_time
        metadata.average_training_time = (
            metadata.total_training_time / metadata.participation_count
        )
        metadata.last_round_participated = round_num

    def record_selection(
        self,
        round_num: int,
        selected_client_ids: List[int],
        available_clients: Optional[List[int]] = None,
        scores: Optional[Dict[int, float]] = None,
    ) -> None:
        """Record per-round selection decisions for fairness-aware tracking."""
        available_set = set(available_clients) if available_clients is not None else set(self.get_available_clients())
        selected_set = set(selected_client_ids)

        for client_id in available_set:
            metadata = self._clients[client_id]

            if client_id in selected_set:
                metadata.selection_count += 1
                metadata.selection_history.append(round_num)
                metadata.last_selected_round = round_num
                metadata.skipped_rounds = 0
            else:
                metadata.skipped_rounds += 1

            if scores is not None and client_id in scores:
                metadata.last_score = scores[client_id]

    def update_client_score(self, client_id: int, score: float) -> None:
        """Update cached score for one client."""
        if client_id not in self._clients:
            raise ValueError(f"Client {client_id} not registered")
        self._clients[client_id].last_score = score

    def record_failure(self, client_id: int) -> None:
        """
        Record a client failure.

        Args:
            client_id: Client that failed

        Raises:
            ValueError: If client not registered
        """
        if client_id not in self._clients:
            raise ValueError(f"Client {client_id} not registered")

        self._clients[client_id].failure_count += 1

    def set_availability(self, client_id: int, is_available: bool) -> None:
        """
        Set client availability status.

        Args:
            client_id: Client to update
            is_available: New availability status

        Raises:
            ValueError: If client not registered
        """
        if client_id not in self._clients:
            raise ValueError(f"Client {client_id} not registered")

        self._clients[client_id].is_available = is_available

    def get_client_metadata(self, client_id: int) -> ClientMetadata:
        """
        Get metadata for specific client.

        Args:
            client_id: Client identifier

        Returns:
            ClientMetadata object

        Raises:
            ValueError: If client not registered
        """
        if client_id not in self._clients:
            raise ValueError(f"Client {client_id} not registered")

        return self._clients[client_id]

    def get_all_clients(self) -> List[int]:
        """
        Get all registered client IDs.

        Returns:
            List of all client IDs
        """
        return list(self._clients.keys())

    def get_available_clients(self) -> List[int]:
        """
        Get IDs of available clients.

        Returns:
            List of available client IDs
        """
        return [
            client_id
            for client_id, metadata in self._clients.items()
            if metadata.is_available
        ]

    def get_num_clients(self) -> int:
        """
        Get total number of registered clients.

        Returns:
            Number of clients
        """
        return len(self._clients)

    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics across all clients.

        Returns:
            Dictionary with stats
        """
        if not self._clients:
            return {
                'total_clients': 0,
                'available_clients': 0,
                'total_dataset_size': 0,
            }

        total_dataset_size = sum(c.dataset_size for c in self._clients.values())
        available_count = len(self.get_available_clients())
        total_participations = sum(
            c.participation_count for c in self._clients.values()
        )
        total_failures = sum(c.failure_count for c in self._clients.values())

        return {
            'total_clients': len(self._clients),
            'available_clients': available_count,
            'total_dataset_size': total_dataset_size,
            'avg_dataset_size': total_dataset_size / len(self._clients),
            'total_participations': total_participations,
            'total_failures': total_failures,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ClientManager(total_clients={len(self._clients)}, "
            f"available={len(self.get_available_clients())})"
        )
