"""
Tests for client simulation features.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from aflf.client.simulation import (
    ClientFailureException,
    ClientSimulationConfig,
    ClientUnavailableException,
    SimulatedFederatedClient,
    compute_dataset_imbalance_metrics,
    create_heterogeneous_clients,
)
from aflf.models import SimpleCNN


@pytest.fixture
def dummy_data():
    """Create dummy dataset for testing."""
    X = torch.randn(100, 1, 28, 28)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(X, y)
    return dataset


@pytest.fixture
def train_loader(dummy_data):
    """Create training data loader."""
    return DataLoader(dummy_data, batch_size=10, shuffle=True)


@pytest.fixture
def model():
    """Create test model."""
    return SimpleCNN(num_classes=10)


class TestClientSimulationConfig:
    """Test ClientSimulationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = ClientSimulationConfig()
        assert config.failure_rate == 0.0
        assert config.training_speed == 1.0
        assert config.stragglers_delay_mean == 0.0
        assert config.max_retries == 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = ClientSimulationConfig(
            failure_rate=0.1,
            training_speed=0.5,
            stragglers_delay_mean=5.0,
        )
        assert config.failure_rate == 0.1
        assert config.training_speed == 0.5
        assert config.stragglers_delay_mean == 5.0

    def test_invalid_failure_rate(self):
        """Test that invalid failure rate raises error."""
        with pytest.raises(AssertionError):
            ClientSimulationConfig(failure_rate=-0.1)

        with pytest.raises(AssertionError):
            ClientSimulationConfig(failure_rate=1.5)

    def test_invalid_training_speed(self):
        """Test that invalid training speed raises error."""
        with pytest.raises(AssertionError):
            ClientSimulationConfig(training_speed=0.0)

        with pytest.raises(AssertionError):
            ClientSimulationConfig(training_speed=-1.0)

    def test_availability_window_validation(self):
        """Test availability window validation."""
        # Valid window
        config = ClientSimulationConfig(availability_window=(9, 17))
        assert config.availability_window == (9, 17)

        # Invalid window (hour >= 24)
        with pytest.raises(AssertionError):
            ClientSimulationConfig(availability_window=(9, 25))

        # Invalid window (hour < 0)
        with pytest.raises(AssertionError):
            ClientSimulationConfig(availability_window=(-1, 17))


class TestSimulatedFederatedClient:
    """Test SimulatedFederatedClient class."""

    def test_init_without_simulation(self, train_loader):
        """Test initialization without simulation config."""
        client = SimulatedFederatedClient(
            client_id=0,
            train_loader=train_loader,
        )
        assert client.client_id == 0
        assert client.simulation_config is not None
        assert client.simulation_config.failure_rate == 0.0

    def test_init_with_simulation(self, train_loader):
        """Test initialization with simulation config."""
        sim_config = ClientSimulationConfig(
            failure_rate=0.1,
            training_speed=0.5,
        )
        client = SimulatedFederatedClient(
            client_id=0,
            train_loader=train_loader,
            simulation_config=sim_config,
        )
        assert client.simulation_config.failure_rate == 0.1
        assert client.simulation_config.training_speed == 0.5

    def test_train_without_simulation(self, model, train_loader):
        """Test training without simulation (normal behavior)."""
        client = SimulatedFederatedClient(
            client_id=0,
            train_loader=train_loader,
            device='cpu',
        )

        result = client.train(
            global_model=model,
            config={'epochs': 1, 'lr': 0.01},
        )

        assert result.client_id == 0
        assert result.train_loss >= 0

    def test_train_with_guaranteed_failure(self, model, train_loader):
        """Test training with guaranteed failure."""
        sim_config = ClientSimulationConfig(failure_rate=1.0)
        client = SimulatedFederatedClient(
            client_id=0,
            train_loader=train_loader,
            simulation_config=sim_config,
            device='cpu',
        )

        with pytest.raises(ClientFailureException):
            client.train(global_model=model)

    def test_train_with_no_failure(self, model, train_loader):
        """Test training with no failure."""
        sim_config = ClientSimulationConfig(failure_rate=0.0)
        client = SimulatedFederatedClient(
            client_id=0,
            train_loader=train_loader,
            simulation_config=sim_config,
            device='cpu',
        )

        # Should not raise
        result = client.train(
            global_model=model,
            config={'epochs': 1},
        )
        assert result is not None

    def test_train_with_variable_speed(self, model, train_loader):
        """Test training with different speeds."""
        # Slow client
        slow_config = ClientSimulationConfig(training_speed=0.5)
        slow_client = SimulatedFederatedClient(
            client_id=0,
            train_loader=train_loader,
            simulation_config=slow_config,
            device='cpu',
        )

        # Fast client
        fast_config = ClientSimulationConfig(training_speed=2.0)
        fast_client = SimulatedFederatedClient(
            client_id=1,
            train_loader=train_loader,
            simulation_config=fast_config,
            device='cpu',
        )

        slow_result = slow_client.train(model, config={'epochs': 1})
        fast_result = fast_client.train(model, config={'epochs': 1})

        # Slow client should take longer (simulated)
        assert slow_result.training_time > fast_result.training_time

    def test_simulation_stats_tracking(self, model, train_loader):
        """Test that simulation statistics are tracked."""
        sim_config = ClientSimulationConfig(failure_rate=0.5)
        client = SimulatedFederatedClient(
            client_id=0,
            train_loader=train_loader,
            simulation_config=sim_config,
            device='cpu',
        )

        # Try training multiple times
        successes = 0
        failures = 0
        for _ in range(10):
            try:
                client.train(model, config={'epochs': 1})
                successes += 1
            except ClientFailureException:
                failures += 1

        stats = client.get_simulation_stats()

        assert stats['total_attempts'] == 10
        assert stats['failed_attempts'] == failures
        assert stats['success_rate'] == successes / 10

    def test_get_simulation_stats_empty(self, train_loader):
        """Test getting stats before any training."""
        client = SimulatedFederatedClient(
            client_id=0,
            train_loader=train_loader,
        )

        stats = client.get_simulation_stats()

        assert stats['total_attempts'] == 0
        assert stats['failed_attempts'] == 0
        assert stats['success_rate'] == 0.0

    def test_repr_with_simulation(self, train_loader):
        """Test string representation with simulation."""
        sim_config = ClientSimulationConfig(
            failure_rate=0.1,
            training_speed=0.5,
        )
        client = SimulatedFederatedClient(
            client_id=0,
            train_loader=train_loader,
            simulation_config=sim_config,
        )

        repr_str = repr(client)
        assert 'Simulated' in repr_str
        assert '0.1' in repr_str or '0.10' in repr_str
        assert '0.5' in repr_str or '0.50' in repr_str


class TestHeterogeneousClients:
    """Test heterogeneous client creation."""

    def test_create_heterogeneous_clients(self, dummy_data):
        """Test creating heterogeneous clients."""
        num_clients = 5

        # Create loaders
        loaders = [
            DataLoader(dummy_data, batch_size=10, shuffle=True)
            for _ in range(num_clients)
        ]

        clients = create_heterogeneous_clients(
            num_clients=num_clients,
            train_loaders=loaders,
            device='cpu',
        )

        assert len(clients) == num_clients
        assert all(isinstance(c, SimulatedFederatedClient) for c in clients)

    def test_heterogeneous_clients_have_varied_configs(self, dummy_data):
        """Test that clients have different configurations."""
        num_clients = 10

        loaders = [
            DataLoader(dummy_data, batch_size=10, shuffle=True)
            for _ in range(num_clients)
        ]

        clients = create_heterogeneous_clients(
            num_clients=num_clients,
            train_loaders=loaders,
            failure_rate_range=(0.0, 0.2),
            speed_range=(0.5, 2.0),
        )

        # Check that not all clients have the same config
        speeds = [c.simulation_config.training_speed for c in clients]
        failure_rates = [c.simulation_config.failure_rate for c in clients]

        # With 10 clients, very unlikely all have same speed
        assert len(set(speeds)) > 1
        # Check speeds are in range
        assert all(0.5 <= s <= 2.0 for s in speeds)
        assert all(0.0 <= f <= 0.2 for f in failure_rates)


class TestDatasetImbalanceMetrics:
    """Test dataset imbalance computation."""

    def test_balanced_dataset(self):
        """Test metrics with perfectly balanced dataset."""
        # Create clients with same number of samples
        clients = []
        for i in range(5):
            data = TensorDataset(
                torch.randn(100, 1, 28, 28),
                torch.randint(0, 10, (100,)),
            )
            loader = DataLoader(data, batch_size=10)
            client = SimulatedFederatedClient(
                client_id=i,
                train_loader=loader,
            )
            clients.append(client)

        metrics = compute_dataset_imbalance_metrics(clients)

        assert metrics['mean_samples'] == 100
        assert metrics['min_samples'] == 100
        assert metrics['max_samples'] == 100
        assert metrics['imbalance_ratio'] == 1.0
        assert metrics['gini_coefficient'] < 0.01  # Very low for balanced

    def test_imbalanced_dataset(self):
        """Test metrics with imbalanced dataset."""
        # Create clients with different sample counts
        sample_counts = [10, 50, 100, 200, 500]
        clients = []
        for i, count in enumerate(sample_counts):
            data = TensorDataset(
                torch.randn(count, 1, 28, 28),
                torch.randint(0, 10, (count,)),
            )
            loader = DataLoader(data, batch_size=10)
            client = SimulatedFederatedClient(
                client_id=i,
                train_loader=loader,
            )
            clients.append(client)

        metrics = compute_dataset_imbalance_metrics(clients)

        assert metrics['min_samples'] == 10
        assert metrics['max_samples'] == 500
        assert metrics['imbalance_ratio'] == 50.0  # 500 / 10
        assert metrics['gini_coefficient'] > 0.3  # Higher for imbalanced

    def test_empty_clients_list(self):
        """Test with empty clients list."""
        metrics = compute_dataset_imbalance_metrics([])
        assert metrics == {}

    def test_single_client(self):
        """Test with single client."""
        data = TensorDataset(
            torch.randn(100, 1, 28, 28),
            torch.randint(0, 10, (100,)),
        )
        loader = DataLoader(data, batch_size=10)
        client = SimulatedFederatedClient(client_id=0, train_loader=loader)

        metrics = compute_dataset_imbalance_metrics([client])

        assert metrics['mean_samples'] == 100
        assert metrics['imbalance_ratio'] == 1.0
        assert metrics['gini_coefficient'] == 0.0


class TestStragglersSimulation:
    """Test straggler simulation."""

    def test_straggler_delay(self, model, train_loader):
        """Test that straggler delay adds time."""
        # Client without straggler delay
        normal_config = ClientSimulationConfig(
            stragglers_delay_mean=0.0,
        )
        normal_client = SimulatedFederatedClient(
            client_id=0,
            train_loader=train_loader,
            simulation_config=normal_config,
            device='cpu',
        )

        # Client with straggler delay
        straggler_config = ClientSimulationConfig(
            stragglers_delay_mean=1.0,  # 1 second delay
            stragglers_delay_std=0.1,
        )
        straggler_client = SimulatedFederatedClient(
            client_id=1,
            train_loader=train_loader,
            simulation_config=straggler_config,
            device='cpu',
        )

        normal_result = normal_client.train(model, config={'epochs': 1})
        straggler_result = straggler_client.train(model, config={'epochs': 1})

        # Straggler should take longer (real delay is added)
        assert straggler_result.training_time > normal_result.training_time
