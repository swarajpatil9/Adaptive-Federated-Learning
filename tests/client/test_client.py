"""
Tests for FederatedClient class.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from aflf.client import FederatedClient, TrainingResult
from aflf.client.client_utils import get_model_weights
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
def val_loader(dummy_data):
    """Create validation data loader."""
    return DataLoader(dummy_data, batch_size=10, shuffle=False)


@pytest.fixture
def model():
    """Create test model."""
    return SimpleCNN(num_classes=10)


class TestTrainingResult:
    """Test TrainingResult dataclass."""

    def test_training_result_creation(self):
        """Test creating TrainingResult."""
        weights = {'conv1.weight': torch.randn(10, 5)}

        result = TrainingResult(
            client_id=0,
            weights=weights,
            num_samples=100,
            train_loss=0.5,
            train_accuracy=0.85,
            val_loss=0.6,
            val_accuracy=0.80,
            training_time=10.5,
        )

        assert result.client_id == 0
        assert result.num_samples == 100
        assert result.train_loss == 0.5
        assert result.train_accuracy == 0.85
        assert result.val_loss == 0.6
        assert result.val_accuracy == 0.80
        assert result.training_time == 10.5

    def test_training_result_to_dict(self):
        """Test converting result to dictionary."""
        weights = {'conv1.weight': torch.randn(10, 5)}

        result = TrainingResult(
            client_id=0,
            weights=weights,
            num_samples=100,
            train_loss=0.5,
            train_accuracy=0.85,
            val_loss=None,
            val_accuracy=None,
            training_time=10.5,
        )

        result_dict = result.to_dict()

        assert 'client_id' in result_dict
        assert 'num_samples' in result_dict
        assert 'train_loss' in result_dict
        assert 'training_time' in result_dict
        assert 'weights' not in result_dict  # Weights excluded for brevity

    def test_training_result_repr(self):
        """Test string representation."""
        weights = {'conv1.weight': torch.randn(10, 5)}

        result = TrainingResult(
            client_id=5,
            weights=weights,
            num_samples=100,
            train_loss=0.5,
            train_accuracy=0.85,
            val_loss=None,
            val_accuracy=None,
            training_time=10.5,
        )

        repr_str = repr(result)
        assert 'TrainingResult' in repr_str
        assert '5' in repr_str  # client_id


class TestFederatedClientInit:
    """Test FederatedClient initialization."""

    def test_init_basic(self, train_loader):
        """Test basic initialization."""
        client = FederatedClient(
            client_id=0,
            train_loader=train_loader,
        )

        assert client.client_id == 0
        assert client.train_loader is train_loader
        assert client.val_loader is None
        assert client.num_train_samples == len(train_loader.dataset)

    def test_init_with_validation(self, train_loader, val_loader):
        """Test initialization with validation."""
        client = FederatedClient(
            client_id=0,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        assert client.val_loader is val_loader
        assert client.num_val_samples == len(val_loader.dataset)

    def test_init_with_device(self, train_loader):
        """Test initialization with specific device."""
        client = FederatedClient(
            client_id=0,
            train_loader=train_loader,
            device='cpu',
        )

        assert client.trainer.device.type == 'cpu'

    def test_init_with_verbose(self, train_loader):
        """Test initialization with verbose mode."""
        client = FederatedClient(
            client_id=0,
            train_loader=train_loader,
            verbose=True,
        )

        assert client.verbose is True
        assert client.trainer.verbose is True

    def test_repr(self, train_loader, val_loader):
        """Test string representation."""
        client = FederatedClient(
            client_id=5,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        repr_str = repr(client)
        assert 'FederatedClient' in repr_str
        assert '5' in repr_str  # client_id

    def test_num_samples_property(self, train_loader):
        """Test num_samples property."""
        client = FederatedClient(
            client_id=0,
            train_loader=train_loader,
        )

        assert client.num_samples == len(train_loader.dataset)
        assert client.num_samples == client.num_train_samples


class TestFederatedClientTrain:
    """Test client training functionality."""

    def test_train_basic(self, model, train_loader):
        """Test basic training."""
        client = FederatedClient(
            client_id=0,
            train_loader=train_loader,
            device='cpu',
            verbose=False,
        )

        result = client.train(global_model=model)

        assert isinstance(result, TrainingResult)
        assert result.client_id == 0
        assert result.num_samples == len(train_loader.dataset)
        assert result.train_loss >= 0
        assert 0 <= result.train_accuracy <= 1
        assert result.val_loss is None
        assert result.val_accuracy is None

    def test_train_with_config(self, model, train_loader):
        """Test training with configuration."""
        client = FederatedClient(
            client_id=0,
            train_loader=train_loader,
            device='cpu',
        )

        config = {
            'epochs': 3,
            'lr': 0.01,
            'optimizer': 'sgd',
            'momentum': 0.9,
        }

        result = client.train(global_model=model, config=config)

        assert result.train_loss >= 0

    def test_train_with_validation(self, model, train_loader, val_loader):
        """Test training with validation."""
        client = FederatedClient(
            client_id=0,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu',
        )

        result = client.train(global_model=model)

        assert result.val_loss is not None
        assert result.val_accuracy is not None
        assert result.val_loss >= 0
        assert 0 <= result.val_accuracy <= 1

    def test_train_returns_correct_weights(self, model, train_loader):
        """Test that training returns updated weights."""
        client = FederatedClient(
            client_id=0,
            train_loader=train_loader,
            device='cpu',
        )

        result = client.train(
            global_model=model,
            config={'epochs': 2, 'lr': 0.01},
        )

        # Check weights format
        assert isinstance(result.weights, dict)
        assert len(result.weights) > 0
        assert all(isinstance(v, torch.Tensor) for v in result.weights.values())

        # Check some expected keys
        assert 'conv1.weight' in result.weights
        assert 'conv1.bias' in result.weights

    def test_train_does_not_modify_global_model(self, model, train_loader):
        """Test that training doesn't modify the global model."""
        client = FederatedClient(
            client_id=0,
            train_loader=train_loader,
            device='cpu',
        )

        # Get initial weights
        initial_weights = get_model_weights(model)

        # Train
        client.train(
            global_model=model,
            config={'epochs': 2, 'lr': 0.01},
        )

        # Check global model unchanged
        final_weights = get_model_weights(model)

        for key in initial_weights.keys():
            assert torch.allclose(initial_weights[key], final_weights[key])

    def test_train_tracks_time(self, model, train_loader):
        """Test that training time is tracked."""
        client = FederatedClient(
            client_id=0,
            train_loader=train_loader,
            device='cpu',
        )

        result = client.train(global_model=model)

        assert result.training_time > 0

    def test_train_multiple_times(self, model, train_loader):
        """Test that same client can train multiple times."""
        client = FederatedClient(
            client_id=0,
            train_loader=train_loader,
            device='cpu',
        )

        # First training round
        result1 = client.train(
            global_model=model,
            config={'epochs': 1},
        )

        # Second training round (simulating new FL round)
        result2 = client.train(
            global_model=model,
            config={'epochs': 1},
        )

        assert result1.train_loss >= 0
        assert result2.train_loss >= 0

    def test_train_with_different_models(self, train_loader):
        """Test training with different model architectures."""
        client = FederatedClient(
            client_id=0,
            train_loader=train_loader,
            device='cpu',
        )

        # Train with first model
        model1 = SimpleCNN(num_classes=10)
        result1 = client.train(global_model=model1)

        # Train with second model
        model2 = SimpleCNN(num_classes=10, dropout_rate=0.3)
        result2 = client.train(global_model=model2)

        assert result1.train_loss >= 0
        assert result2.train_loss >= 0


class TestFederatedClientStateless:
    """Test that client training is stateless."""

    def test_client_stateless_between_rounds(self, model, train_loader):
        """Test that client doesn't maintain state between training rounds."""
        client = FederatedClient(
            client_id=0,
            train_loader=train_loader,
            device='cpu',
        )

        # Train first round
        result1 = client.train(
            global_model=model,
            config={'epochs': 1, 'lr': 0.01},
        )

        # Train second round with different config
        result2 = client.train(
            global_model=model,
            config={'epochs': 2, 'lr': 0.001},
        )

        # Both should succeed independently
        assert result1.train_loss >= 0
        assert result2.train_loss >= 0

    def test_multiple_clients_independent(self, model, dummy_data):
        """Test that multiple clients are independent."""
        # Create two clients with same data
        loader1 = DataLoader(dummy_data, batch_size=10, shuffle=True)
        loader2 = DataLoader(dummy_data, batch_size=10, shuffle=True)

        client1 = FederatedClient(
            client_id=0,
            train_loader=loader1,
            device='cpu',
        )

        client2 = FederatedClient(
            client_id=1,
            train_loader=loader2,
            device='cpu',
        )

        # Train both
        result1 = client1.train(global_model=model)
        result2 = client2.train(global_model=model)

        # Both should succeed
        assert result1.client_id == 0
        assert result2.client_id == 1
        assert result1.train_loss >= 0
        assert result2.train_loss >= 0


class TestFederatedClientEdgeCases:
    """Test edge cases and error handling."""

    def test_train_with_empty_config(self, model, train_loader):
        """Test training with empty config (should use defaults)."""
        client = FederatedClient(
            client_id=0,
            train_loader=train_loader,
            device='cpu',
        )

        result = client.train(global_model=model, config={})

        assert result.train_loss >= 0

    def test_train_with_none_config(self, model, train_loader):
        """Test training with None config (should use defaults)."""
        client = FederatedClient(
            client_id=0,
            train_loader=train_loader,
            device='cpu',
        )

        result = client.train(global_model=model, config=None)

        assert result.train_loss >= 0
