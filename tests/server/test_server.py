"""
Integration tests for FederatedServer.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from aflf.client.client import FederatedClient
from aflf.selection import RandomSelection
from aflf.server import FederatedServer


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def simple_model():
    """Create simple model for testing."""
    return SimpleModel()


@pytest.fixture
def dummy_clients():
    """Create dummy clients for testing."""
    clients = {}

    for i in range(5):
        # Create dummy data
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32)

        client = FederatedClient(
            client_id=i,
            train_loader=loader,
            val_loader=None,
            epochs=1,
            device="cpu",
        )
        clients[i] = client

    return clients


class TestFederatedServer:
    """Integration tests for FederatedServer."""

    def test_initialization(self, simple_model):
        """Test server initialization."""
        server = FederatedServer(
            model=simple_model,
            selection_strategy=RandomSelection(seed=42),
            num_clients_per_round=3,
        )

        assert server.client_manager.get_num_clients() == 0
        assert server.round_manager.get_num_rounds() == 0
        assert server.num_clients_per_round == 3

    def test_register_clients(self, simple_model):
        """Test client registration."""
        server = FederatedServer(model=simple_model)

        server.register_client(client_id=0, dataset_size=600)
        server.register_client(client_id=1, dataset_size=550)
        server.register_client(client_id=2, dataset_size=500)

        assert server.client_manager.get_num_clients() == 3
        assert server.client_manager.get_all_clients() == [0, 1, 2]

    def test_execute_round(self, simple_model, dummy_clients):
        """Test executing a round."""
        server = FederatedServer(
            model=simple_model,
            selection_strategy=RandomSelection(seed=42),
            num_clients_per_round=3,
        )

        # Register clients
        for client_id in dummy_clients.keys():
            server.register_client(client_id=client_id, dataset_size=100)

        # Execute round
        result = server.execute_round(round_num=0, clients=dummy_clients)

        # Check result structure
        assert result['round_num'] == 0
        assert result['num_selected'] == 3  # num_clients_per_round
        assert result['num_participating'] <= 3
        assert result['num_failed'] + result['num_participating'] == 3
        assert 0.0 <= result['participation_rate'] <= 1.0
        assert 'metrics' in result
        assert 'duration' in result

    def test_execute_multiple_rounds(self, simple_model, dummy_clients):
        """Test executing multiple rounds."""
        server = FederatedServer(
            model=simple_model,
            selection_strategy=RandomSelection(seed=42),
            num_clients_per_round=2,
        )

        # Register clients
        for client_id in dummy_clients.keys():
            server.register_client(client_id=client_id, dataset_size=100)

        # Execute 3 rounds
        for round_num in range(3):
            result = server.execute_round(round_num=round_num, clients=dummy_clients)
            assert result['round_num'] == round_num

        # Check history
        assert server.round_manager.get_num_rounds() == 3
        history = server.get_round_history()
        assert len(history) == 3

    def test_execute_round_no_available_clients_raises_error(self, simple_model):
        """Test that executing round with no clients raises error."""
        server = FederatedServer(model=simple_model)

        with pytest.raises(ValueError, match="No available clients"):
            server.execute_round(round_num=0, clients={})

    def test_get_global_model(self, simple_model):
        """Test getting global model."""
        server = FederatedServer(model=simple_model)

        global_model = server.get_global_model()
        assert isinstance(global_model, nn.Module)

    def test_get_global_parameters(self, simple_model):
        """Test getting global parameters."""
        server = FederatedServer(model=simple_model)

        params = server.get_global_parameters()
        assert 'fc.weight' in params
        assert 'fc.bias' in params

    def test_set_global_parameters(self, simple_model):
        """Test setting global parameters."""
        server = FederatedServer(model=simple_model)

        # Get initial parameters
        initial_params = server.get_global_parameters()

        # Modify parameters
        modified_params = initial_params.copy()
        for key in modified_params:
            modified_params[key] = torch.ones_like(modified_params[key])

        # Set new parameters
        server.set_global_parameters(modified_params)

        # Verify they were set
        new_params = server.get_global_parameters()
        for key in new_params:
            assert torch.allclose(new_params[key], torch.ones_like(new_params[key]))

    def test_get_client_summary(self, simple_model):
        """Test getting client summary."""
        server = FederatedServer(model=simple_model)

        server.register_client(client_id=0, dataset_size=600)
        server.register_client(client_id=1, dataset_size=500, is_available=False)

        summary = server.get_client_summary()
        assert summary['total_clients'] == 2
        assert summary['available_clients'] == 1
        assert summary['total_dataset_size'] == 1100

    def test_get_round_summary(self, simple_model, dummy_clients):
        """Test getting round summary."""
        server = FederatedServer(
            model=simple_model,
            selection_strategy=RandomSelection(seed=42),
            num_clients_per_round=2,
        )

        # Register clients
        for client_id in dummy_clients.keys():
            server.register_client(client_id=client_id, dataset_size=100)

        # Execute rounds
        server.execute_round(round_num=0, clients=dummy_clients)
        server.execute_round(round_num=1, clients=dummy_clients)

        summary = server.get_round_summary()
        assert summary['total_rounds'] == 2
        assert 'avg_participation_rate' in summary
        assert 'avg_duration' in summary

    def test_repr(self, simple_model):
        """Test string representation."""
        server = FederatedServer(model=simple_model)
        server.register_client(client_id=0, dataset_size=600)

        repr_str = repr(server)
        assert 'FederatedServer' in repr_str
        assert 'clients=1' in repr_str
        assert 'rounds=0' in repr_str
