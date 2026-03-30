"""
Round manager for federated learning server.

Manages round state, tracks participation, and maintains history.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RoundState:
    """
    State and metrics for a single FL round.

    Attributes:
        round_num: Round number (0-indexed)
        selected_clients: Clients selected for this round
        participating_clients: Clients that successfully completed training
        dropped_clients: Clients that failed during training
        metrics: Aggregated metrics for this round
        start_time: Round start timestamp
        end_time: Round end timestamp (None if not finished)
        duration: Round duration in seconds (None if not finished)
    """

    round_num: int
    selected_clients: List[int] = field(default_factory=list)
    participating_clients: List[int] = field(default_factory=list)
    dropped_clients: List[int] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None

    def finalize(self) -> None:
        """
        Finalize round by recording end time and duration.
        """
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

    @property
    def participation_rate(self) -> float:
        """
        Calculate participation rate.

        Returns:
            Fraction of selected clients that participated
        """
        if not self.selected_clients:
            return 0.0
        return len(self.participating_clients) / len(self.selected_clients)

    @property
    def failure_rate(self) -> float:
        """
        Calculate failure rate.

        Returns:
            Fraction of selected clients that dropped
        """
        if not self.selected_clients:
            return 0.0
        return len(self.dropped_clients) / len(self.selected_clients)

    def to_dict(self) -> Dict:
        """
        Convert round state to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'round_num': self.round_num,
            'num_selected': len(self.selected_clients),
            'num_participating': len(self.participating_clients),
            'num_dropped': len(self.dropped_clients),
            'participation_rate': self.participation_rate,
            'failure_rate': self.failure_rate,
            'duration': self.duration,
            'metrics': self.metrics,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RoundState(round={self.round_num}, "
            f"selected={len(self.selected_clients)}, "
            f"participating={len(self.participating_clients)}, "
            f"dropped={len(self.dropped_clients)})"
        )


class RoundManager:
    """
    Manages round state and history for federated server.

    Responsibilities:
    - Create new round state
    - Track selected/participating/dropped clients
    - Record round metrics
    - Maintain history of all rounds
    - Provide round statistics

    Example:
        >>> manager = RoundManager()
        >>> round_state = manager.start_round(
        ...     round_num=0,
        ...     selected_clients=[0, 1, 2]
        ... )
        >>> manager.record_participation(client_id=0)
        >>> manager.record_participation(client_id=1)
        >>> manager.record_drop(client_id=2, reason="timeout")
        >>> metrics = {'avg_loss': 0.5, 'avg_accuracy': 0.85}
        >>> manager.end_round(metrics=metrics)
    """

    def __init__(self):
        """Initialize round manager."""
        self._history: List[RoundState] = []
        self._current_round: Optional[RoundState] = None

    def start_round(self, round_num: int, selected_clients: List[int]) -> RoundState:
        """
        Start a new round.

        Args:
            round_num: Round number (0-indexed)
            selected_clients: List of client IDs selected for this round

        Returns:
            RoundState object for this round

        Raises:
            RuntimeError: If a round is already in progress
        """
        if self._current_round is not None:
            raise RuntimeError(
                f"Round {self._current_round.round_num} still in progress"
            )

        self._current_round = RoundState(
            round_num=round_num, selected_clients=selected_clients.copy()
        )
        return self._current_round

    def record_participation(self, client_id: int) -> None:
        """
        Record successful client participation.

        Args:
            client_id: Client that completed training

        Raises:
            RuntimeError: If no round is in progress
        """
        if self._current_round is None:
            raise RuntimeError("No round in progress")

        if client_id not in self._current_round.participating_clients:
            self._current_round.participating_clients.append(client_id)

    def record_drop(self, client_id: int, reason: str = "unknown") -> None:
        """
        Record client dropout/failure.

        Args:
            client_id: Client that failed
            reason: Reason for failure (for logging)

        Raises:
            RuntimeError: If no round is in progress
        """
        if self._current_round is None:
            raise RuntimeError("No round in progress")

        if client_id not in self._current_round.dropped_clients:
            self._current_round.dropped_clients.append(client_id)

    def end_round(self, metrics: Dict) -> RoundState:
        """
        End current round and record metrics.

        Args:
            metrics: Aggregated metrics for this round

        Returns:
            Completed RoundState

        Raises:
            RuntimeError: If no round is in progress
        """
        if self._current_round is None:
            raise RuntimeError("No round in progress")

        self._current_round.metrics = metrics
        self._current_round.finalize()

        # Save to history
        self._history.append(self._current_round)
        completed_round = self._current_round
        self._current_round = None

        return completed_round

    def get_current_round(self) -> Optional[RoundState]:
        """
        Get current round state.

        Returns:
            Current RoundState or None if no round in progress
        """
        return self._current_round

    def get_round_history(self) -> List[RoundState]:
        """
        Get history of all completed rounds.

        Returns:
            List of RoundState objects
        """
        return self._history.copy()

    def get_round(self, round_num: int) -> Optional[RoundState]:
        """
        Get specific round from history.

        Args:
            round_num: Round number to retrieve

        Returns:
            RoundState if found, None otherwise
        """
        for round_state in self._history:
            if round_state.round_num == round_num:
                return round_state
        return None

    def get_num_rounds(self) -> int:
        """
        Get number of completed rounds.

        Returns:
            Number of rounds in history
        """
        return len(self._history)

    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics across all rounds.

        Returns:
            Dictionary with stats
        """
        if not self._history:
            return {
                'total_rounds': 0,
                'avg_participation_rate': 0.0,
                'avg_failure_rate': 0.0,
                'avg_duration': 0.0,
            }

        total_rounds = len(self._history)
        avg_participation = sum(r.participation_rate for r in self._history) / total_rounds
        avg_failure = sum(r.failure_rate for r in self._history) / total_rounds
        avg_duration = sum(r.duration for r in self._history if r.duration) / total_rounds

        return {
            'total_rounds': total_rounds,
            'avg_participation_rate': avg_participation,
            'avg_failure_rate': avg_failure,
            'avg_duration': avg_duration,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"RoundManager(completed_rounds={len(self._history)})"
