"""
Client selection strategies for federated learning.

Defines abstract interface and concrete implementations for selecting
clients to participate in each FL round.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from ..server.client_manager import ClientMetadata


@dataclass
class SelectionResult:
    """Structured selection output for logging and analysis."""

    selected_client_ids: List[int]
    client_scores: Dict[int, float] = field(default_factory=dict)
    selection_reasoning: Dict[int, str] = field(default_factory=dict)
    policy_name: str = "unknown"


class BaseSelectionStrategy(ABC):
    """Base interface for all client selection strategies."""

    @abstractmethod
    def select(
        self,
        available_clients: List[int],
        num_clients: int,
        round_num: int,
        client_metadata: Optional[Dict[int, "ClientMetadata"]] = None,
    ) -> List[int]:
        """Return selected client IDs for the round."""

    def select_with_details(
        self,
        available_clients: List[int],
        num_clients: int,
        round_num: int,
        client_metadata: Optional[Dict[int, "ClientMetadata"]] = None,
    ) -> SelectionResult:
        """Default detailed output wrapper around select()."""
        selected = self.select(
            available_clients=available_clients,
            num_clients=num_clients,
            round_num=round_num,
            client_metadata=client_metadata,
        )
        return SelectionResult(
            selected_client_ids=selected,
            client_scores={},
            selection_reasoning={cid: "Selected by strategy" for cid in selected},
            policy_name=self.__class__.__name__,
        )


class SelectionStrategy(BaseSelectionStrategy):
    """
    Abstract base class for client selection strategies.

    Selection strategies determine which clients participate in each round.
    This abstraction enables researchers to implement and compare different
    selection algorithms.

    Common strategies:
    - Random: Uniform random sampling
    - Resource-aware: Select fastest clients
    - Data-aware: Select clients with most data
    - Fairness-based: Ensure equal participation
    - Adaptive: Based on historical performance
    """

    @abstractmethod
    def select(
        self,
        available_clients: List[int],
        num_clients: int,
        round_num: int,
        client_metadata: Optional[Dict[int, "ClientMetadata"]] = None,
    ) -> List[int]:
        """
        Select clients for a round.

        Args:
            available_clients: List of available client IDs
            num_clients: Number of clients to select
            round_num: Current round number
            client_metadata: Optional metadata for decision-making

        Returns:
            List of selected client IDs

        Raises:
            ValueError: If num_clients > len(available_clients)
        """
        raise NotImplementedError


class RandomSelection(SelectionStrategy):
    """
    Random client selection strategy.

    Selects clients uniformly at random from available clients.
    This is the baseline selection strategy used in standard FL.

    Example:
        >>> strategy = RandomSelection(seed=42)
        >>> selected = strategy.select(
        ...     available_clients=[0, 1, 2, 3, 4],
        ...     num_clients=2,
        ...     round_num=0
        ... )
        >>> print(selected)  # [1, 3] (deterministic with seed)
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random selection strategy.

        Args:
            seed: Random seed for reproducibility (None = non-deterministic)
        """
        self.seed = seed
        self._rng = random.Random(seed)

    def select(
        self,
        available_clients: List[int],
        num_clients: int,
        round_num: int,
        client_metadata: Optional[Dict[int, "ClientMetadata"]] = None,
    ) -> List[int]:
        """
        Select clients uniformly at random.

        Args:
            available_clients: List of available client IDs
            num_clients: Number of clients to select
            round_num: Current round number (unused)
            client_metadata: Client metadata (unused)

        Returns:
            List of randomly selected client IDs

        Raises:
            ValueError: If num_clients > len(available_clients)
        """
        if num_clients > len(available_clients):
            raise ValueError(
                f"Cannot select {num_clients} clients from "
                f"{len(available_clients)} available clients"
            )

        if num_clients == len(available_clients):
            return available_clients.copy()

        return self._rng.sample(available_clients, num_clients)

    def __repr__(self) -> str:
        """String representation."""
        return f"RandomSelection(seed={self.seed})"


class DataAwareSelection(SelectionStrategy):
    """
    Data-aware client selection strategy.

    Selects clients with the most training data first. This can be beneficial
    when client datasets have high variance in size.

    Example:
        >>> strategy = DataAwareSelection()
        >>> selected = strategy.select(
        ...     available_clients=[0, 1, 2],
        ...     num_clients=2,
        ...     round_num=0,
        ...     client_metadata={
        ...         0: ClientMetadata(client_id=0, dataset_size=100),
        ...         1: ClientMetadata(client_id=1, dataset_size=500),
        ...         2: ClientMetadata(client_id=2, dataset_size=300),
        ...     }
        ... )
        >>> print(selected)  # [1, 2] (largest datasets)
    """

    def select(
        self,
        available_clients: List[int],
        num_clients: int,
        round_num: int,
        client_metadata: Optional[Dict[int, "ClientMetadata"]] = None,
    ) -> List[int]:
        """
        Select clients with largest datasets.

        Args:
            available_clients: List of available client IDs
            num_clients: Number of clients to select
            round_num: Current round number (unused)
            client_metadata: Client metadata with dataset_size

        Returns:
            List of client IDs sorted by dataset size (descending)

        Raises:
            ValueError: If num_clients > len(available_clients) or metadata missing
        """
        if num_clients > len(available_clients):
            raise ValueError(
                f"Cannot select {num_clients} clients from "
                f"{len(available_clients)} available clients"
            )

        if client_metadata is None:
            raise ValueError("client_metadata required for DataAwareSelection")

        # Sort clients by dataset size (descending)
        sorted_clients = sorted(
            available_clients,
            key=lambda c: client_metadata[c].dataset_size,
            reverse=True,
        )

        return sorted_clients[:num_clients]

    def __repr__(self) -> str:
        """String representation."""
        return "DataAwareSelection()"


class FairnessSelection(SelectionStrategy):
    """
    Fairness-based client selection strategy.

    Prioritizes clients that have participated less frequently to ensure
    fair participation across all clients.

    Example:
        >>> strategy = FairnessSelection(seed=42)
        >>> selected = strategy.select(
        ...     available_clients=[0, 1, 2],
        ...     num_clients=2,
        ...     round_num=5,
        ...     client_metadata={
        ...         0: ClientMetadata(client_id=0, dataset_size=100, participation_count=5),
        ...         1: ClientMetadata(client_id=1, dataset_size=100, participation_count=2),
        ...         2: ClientMetadata(client_id=2, dataset_size=100, participation_count=1),
        ...     }
        ... )
        >>> print(selected)  # [2, 1] (least participated first)
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize fairness selection strategy.

        Args:
            seed: Random seed for tie-breaking
        """
        self.seed = seed
        self._rng = random.Random(seed)

    def select(
        self,
        available_clients: List[int],
        num_clients: int,
        round_num: int,
        client_metadata: Optional[Dict[int, "ClientMetadata"]] = None,
    ) -> List[int]:
        """
        Select clients with least participation history.

        Args:
            available_clients: List of available client IDs
            num_clients: Number of clients to select
            round_num: Current round number (unused)
            client_metadata: Client metadata with participation_count

        Returns:
            List of client IDs sorted by participation count (ascending)

        Raises:
            ValueError: If num_clients > len(available_clients) or metadata missing
        """
        if num_clients > len(available_clients):
            raise ValueError(
                f"Cannot select {num_clients} clients from "
                f"{len(available_clients)} available clients"
            )

        if client_metadata is None:
            raise ValueError("client_metadata required for FairnessSelection")

        # Sort clients by participation count (ascending), shuffle for tie-breaking
        clients_shuffled = available_clients.copy()
        self._rng.shuffle(clients_shuffled)

        sorted_clients = sorted(
            clients_shuffled,
            key=lambda c: client_metadata[c].participation_count,
        )

        return sorted_clients[:num_clients]

    def __repr__(self) -> str:
        """String representation."""
        return f"FairnessSelection(seed={self.seed})"
