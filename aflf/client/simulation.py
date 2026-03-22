"""
Client simulation features for realistic federated learning experiments.

Provides utilities to simulate:
- Random client failures (dropout)
- Variable training speed (heterogeneous compute)
- Dataset imbalance (non-uniform data distribution)

These features help researchers test FL algorithms under realistic conditions.
"""

import random
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch.nn as nn
from torch.utils.data import DataLoader

from .client import FederatedClient, TrainingResult


@dataclass
class ClientSimulationConfig:
    """
    Configuration for client simulation.

    Attributes:
        failure_rate: Probability of client failure (0.0 = never fails, 1.0 = always fails)
        training_speed: Relative training speed (1.0 = normal, 0.5 = half speed, 2.0 = double)
        stragglers_delay_mean: Mean additional delay for stragglers (seconds)
        stragglers_delay_std: Std dev of straggler delay (seconds)
        max_retries: Maximum number of retry attempts on failure
        availability_window: (start_hour, end_hour) for client availability (24-hour format)

    Example:
        >>> config = ClientSimulationConfig(
        >>>     failure_rate=0.1,  # 10% chance of failure
        >>>     training_speed=0.5,  # Half-speed client
        >>> )
    """

    failure_rate: float = 0.0
    training_speed: float = 1.0
    stragglers_delay_mean: float = 0.0
    stragglers_delay_std: float = 0.0
    max_retries: int = 0
    availability_window: Optional[tuple] = None

    def __post_init__(self):
        """Validate configuration."""
        assert 0.0 <= self.failure_rate <= 1.0, "failure_rate must be in [0, 1]"
        assert self.training_speed > 0, "training_speed must be positive"
        assert self.stragglers_delay_mean >= 0, "stragglers_delay_mean must be non-negative"
        assert self.stragglers_delay_std >= 0, "stragglers_delay_std must be non-negative"
        assert self.max_retries >= 0, "max_retries must be non-negative"

        if self.availability_window is not None:
            start, end = self.availability_window
            assert 0 <= start < 24, "start_hour must be in [0, 24)"
            assert 0 <= end < 24, "end_hour must be in [0, 24)"


class ClientFailureException(Exception):
    """Exception raised when client simulation fails."""

    pass


class ClientUnavailableException(Exception):
    """Exception raised when client is outside availability window."""

    pass


class SimulatedFederatedClient(FederatedClient):
    """
    Federated client with realistic simulation features.

    Extends FederatedClient with:
    - Random failures (client dropout)
    - Variable training speed (heterogeneous compute)
    - Availability windows (diurnal patterns)
    - Straggler simulation (slow clients)

    Example:
        >>> sim_config = ClientSimulationConfig(
        >>>     failure_rate=0.1,
        >>>     training_speed=0.5,  # Slow client
        >>>     stragglers_delay_mean=5.0,  # 5s extra delay
        >>> )
        >>> client = SimulatedFederatedClient(
        >>>     client_id=0,
        >>>     train_loader=loader,
        >>>     simulation_config=sim_config,
        >>> )
        >>> try:
        >>>     result = client.train(model)
        >>> except ClientFailureException:
        >>>     print("Client failed during training")
    """

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[str] = None,
        verbose: bool = False,
        simulation_config: Optional[ClientSimulationConfig] = None,
    ):
        """
        Initialize simulated federated client.

        Args:
            client_id: Unique client identifier
            train_loader: Training data loader
            val_loader: Optional validation data loader
            device: Training device
            verbose: If True, print progress
            simulation_config: Simulation parameters (None for real client behavior)
        """
        super().__init__(
            client_id=client_id,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            verbose=verbose,
        )

        self.simulation_config = simulation_config or ClientSimulationConfig()

        # Track simulation statistics
        self.total_attempts = 0
        self.failed_attempts = 0
        self.total_training_time = 0.0

    def train(
        self,
        global_model: nn.Module,
        config: Optional[Dict] = None,
    ) -> TrainingResult:
        """
        Execute local training with simulation.

        Applies simulation features:
        - Checks availability window
        - Simulates random failures
        - Simulates variable training speed
        - Adds straggler delays

        Args:
            global_model: Global model from server
            config: Training configuration

        Returns:
            TrainingResult with updated weights and metrics

        Raises:
            ClientUnavailableException: If outside availability window
            ClientFailureException: If client fails during training

        Example:
            >>> result = client.train(model, config={'epochs': 5})
        """
        self.total_attempts += 1

        # Check availability window
        if not self._is_available():
            raise ClientUnavailableException(
                f"Client {self.client_id} is outside availability window"
            )

        # Simulate random failure
        if self._should_fail():
            self.failed_attempts += 1
            raise ClientFailureException(
                f"Client {self.client_id} failed (failure_rate={self.simulation_config.failure_rate})"
            )

        # Execute training with speed simulation
        start_time = time.time()

        if self.verbose and self.simulation_config.training_speed != 1.0:
            print(
                f"[Client {self.client_id}] Training speed: {self.simulation_config.training_speed:.2f}x"
            )

        # Call parent train method
        result = super().train(global_model=global_model, config=config)

        # Adjust training time for simulation
        real_time = time.time() - start_time
        simulated_time = real_time / self.simulation_config.training_speed

        # Add straggler delay if configured
        straggler_delay = self._get_straggler_delay()
        if straggler_delay > 0:
            if self.verbose:
                print(f"[Client {self.client_id}] Straggler delay: {straggler_delay:.2f}s")
            time.sleep(straggler_delay)
            simulated_time += straggler_delay

        # Update result with simulated time
        result.training_time = simulated_time
        self.total_training_time += simulated_time

        return result

    def _is_available(self) -> bool:
        """
        Check if client is available based on availability window.

        Returns:
            True if client is available, False otherwise
        """
        if self.simulation_config.availability_window is None:
            return True

        # Get current hour (for simulation, we could inject this)
        current_hour = time.localtime().tm_hour
        start_hour, end_hour = self.simulation_config.availability_window

        if start_hour <= end_hour:
            return start_hour <= current_hour < end_hour
        else:
            # Wraps around midnight (e.g., 22:00 to 06:00)
            return current_hour >= start_hour or current_hour < end_hour

    def _should_fail(self) -> bool:
        """
        Determine if client should fail this attempt.

        Returns:
            True if client should fail, False otherwise
        """
        return random.random() < self.simulation_config.failure_rate

    def _get_straggler_delay(self) -> float:
        """
        Get straggler delay (additional time for slow clients).

        Returns:
            Additional delay in seconds (0 if no delay)
        """
        if self.simulation_config.stragglers_delay_mean == 0:
            return 0.0

        # Sample from normal distribution, clipped to non-negative
        delay = random.gauss(
            self.simulation_config.stragglers_delay_mean,
            self.simulation_config.stragglers_delay_std,
        )
        return max(0.0, delay)

    def get_simulation_stats(self) -> Dict[str, float]:
        """
        Get simulation statistics for this client.

        Returns:
            Dictionary with simulation metrics:
                - total_attempts: Total training attempts
                - failed_attempts: Number of failures
                - success_rate: Fraction of successful attempts
                - total_training_time: Cumulative training time
                - avg_training_time: Average time per successful attempt

        Example:
            >>> stats = client.get_simulation_stats()
            >>> print(f"Success rate: {stats['success_rate']:.2%}")
        """
        successful_attempts = self.total_attempts - self.failed_attempts

        return {
            'total_attempts': self.total_attempts,
            'failed_attempts': self.failed_attempts,
            'success_rate': (
                successful_attempts / self.total_attempts
                if self.total_attempts > 0
                else 0.0
            ),
            'total_training_time': self.total_training_time,
            'avg_training_time': (
                self.total_training_time / successful_attempts
                if successful_attempts > 0
                else 0.0
            ),
        }

    def __repr__(self) -> str:
        """String representation."""
        base_repr = super().__repr__()
        if self.simulation_config.failure_rate > 0 or self.simulation_config.training_speed != 1.0:
            return (
                f"Simulated{base_repr[:-1]}, "
                f"failure_rate={self.simulation_config.failure_rate:.2f}, "
                f"speed={self.simulation_config.training_speed:.2f}x)"
            )
        return base_repr


def create_heterogeneous_clients(
    num_clients: int,
    train_loaders: list,
    val_loader: Optional[DataLoader] = None,
    device: Optional[str] = None,
    failure_rate_range: tuple = (0.0, 0.2),
    speed_range: tuple = (0.5, 2.0),
    straggler_probability: float = 0.1,
    straggler_delay_range: tuple = (5.0, 15.0),
) -> list:
    """
    Create a heterogeneous set of simulated clients.

    This function creates clients with varied characteristics to simulate
    realistic federated learning environments with diverse devices.

    Args:
        num_clients: Number of clients to create
        train_loaders: List of per-client training data loaders
        val_loader: Optional shared validation loader
        device: Training device
        failure_rate_range: (min, max) failure rate range
        speed_range: (min, max) training speed range
        straggler_probability: Probability of being a straggler
        straggler_delay_range: (min, max) straggler delay range (seconds)

    Returns:
        List of SimulatedFederatedClient instances

    Example:
        >>> # Create 10 heterogeneous clients
        >>> clients = create_heterogeneous_clients(
        >>>     num_clients=10,
        >>>     train_loaders=loaders,
        >>>     failure_rate_range=(0.0, 0.1),
        >>>     speed_range=(0.5, 1.5),
        >>> )
        >>> # Some clients will be fast, some slow, some unreliable
    """
    clients = []

    for client_id in range(num_clients):
        # Random failure rate
        failure_rate = random.uniform(*failure_rate_range)

        # Random training speed
        training_speed = random.uniform(*speed_range)

        # Straggler configuration
        is_straggler = random.random() < straggler_probability
        if is_straggler:
            straggler_delay_mean = random.uniform(*straggler_delay_range)
            straggler_delay_std = straggler_delay_mean * 0.2
        else:
            straggler_delay_mean = 0.0
            straggler_delay_std = 0.0

        sim_config = ClientSimulationConfig(
            failure_rate=failure_rate,
            training_speed=training_speed,
            stragglers_delay_mean=straggler_delay_mean,
            stragglers_delay_std=straggler_delay_std,
        )

        client = SimulatedFederatedClient(
            client_id=client_id,
            train_loader=train_loaders[client_id],
            val_loader=val_loader,
            device=device,
            simulation_config=sim_config,
        )

        clients.append(client)

    return clients


def compute_dataset_imbalance_metrics(
    clients: list,
) -> Dict[str, float]:
    """
    Compute dataset imbalance metrics across clients.

    Measures how imbalanced the data distribution is, which affects
    FL convergence and fairness.

    Args:
        clients: List of FederatedClient or SimulatedFederatedClient

    Returns:
        Dictionary with imbalance metrics:
            - mean_samples: Average samples per client
            - std_samples: Standard deviation of samples
            - min_samples: Minimum samples (smallest client)
            - max_samples: Maximum samples (largest client)
            - imbalance_ratio: max_samples / min_samples
            - gini_coefficient: Gini coefficient (0=perfect balance, 1=max imbalance)

    Example:
        >>> metrics = compute_dataset_imbalance_metrics(clients)
        >>> print(f"Imbalance ratio: {metrics['imbalance_ratio']:.2f}")
        >>> print(f"Gini coefficient: {metrics['gini_coefficient']:.3f}")
    """
    sample_counts = [client.num_samples for client in clients]

    if not sample_counts:
        return {}

    mean_samples = sum(sample_counts) / len(sample_counts)
    min_samples = min(sample_counts)
    max_samples = max(sample_counts)

    # Compute standard deviation
    variance = sum((x - mean_samples) ** 2 for x in sample_counts) / len(sample_counts)
    std_samples = variance ** 0.5

    # Compute Gini coefficient
    # https://en.wikipedia.org/wiki/Gini_coefficient
    sorted_counts = sorted(sample_counts)
    n = len(sorted_counts)
    cumsum = 0
    for i, count in enumerate(sorted_counts):
        cumsum += (2 * (i + 1) - n - 1) * count
    gini = cumsum / (n * sum(sample_counts))

    return {
        'mean_samples': mean_samples,
        'std_samples': std_samples,
        'min_samples': min_samples,
        'max_samples': max_samples,
        'imbalance_ratio': max_samples / min_samples if min_samples > 0 else float('inf'),
        'gini_coefficient': gini,
    }
