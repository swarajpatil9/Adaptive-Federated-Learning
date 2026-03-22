"""
Tests for LocalTrainer class.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from aflf.client.trainer import LocalTrainer
from aflf.models import SimpleCNN


@pytest.fixture
def dummy_data():
    """Create dummy dataset for testing."""
    # Create random data (MNIST-like: 28x28 grayscale)
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
    # Use same data for simplicity in tests
    return DataLoader(dummy_data, batch_size=10, shuffle=False)


@pytest.fixture
def model():
    """Create test model."""
    return SimpleCNN(num_classes=10)


class TestLocalTrainerInit:
    """Test LocalTrainer initialization."""

    def test_init_default(self):
        """Test default initialization."""
        trainer = LocalTrainer()
        assert trainer.device is not None
        assert trainer.verbose is False

    def test_init_with_device(self):
        """Test initialization with specific device."""
        trainer = LocalTrainer(device='cpu')
        assert trainer.device.type == 'cpu'

    def test_init_with_verbose(self):
        """Test initialization with verbose mode."""
        trainer = LocalTrainer(verbose=True)
        assert trainer.verbose is True

    def test_repr(self):
        """Test string representation."""
        trainer = LocalTrainer(device='cpu')
        repr_str = repr(trainer)
        assert 'LocalTrainer' in repr_str
        assert 'cpu' in repr_str


class TestLocalTrainerTrain:
    """Test training functionality."""

    def test_train_single_epoch(self, model, train_loader):
        """Test training for one epoch."""
        trainer = LocalTrainer(device='cpu', verbose=False)

        results = trainer.train(
            model=model,
            train_loader=train_loader,
            epochs=1,
            lr=0.01,
        )

        assert 'train_loss' in results
        assert 'train_accuracy' in results
        assert 'num_samples' in results
        assert results['num_samples'] == len(train_loader.dataset)

    def test_train_multiple_epochs(self, model, train_loader):
        """Test training for multiple epochs."""
        trainer = LocalTrainer(device='cpu', verbose=False)

        results = trainer.train(
            model=model,
            train_loader=train_loader,
            epochs=3,
            lr=0.01,
        )

        assert results['train_loss'] >= 0
        assert 0 <= results['train_accuracy'] <= 1

    def test_train_with_validation(self, model, train_loader, val_loader):
        """Test training with validation."""
        trainer = LocalTrainer(device='cpu', verbose=False)

        results = trainer.train(
            model=model,
            train_loader=train_loader,
            epochs=2,
            lr=0.01,
            val_loader=val_loader,
        )

        assert results['val_loss'] is not None
        assert results['val_accuracy'] is not None
        assert results['val_loss'] >= 0
        assert 0 <= results['val_accuracy'] <= 1

    def test_train_without_validation(self, model, train_loader):
        """Test training without validation."""
        trainer = LocalTrainer(device='cpu', verbose=False)

        results = trainer.train(
            model=model,
            train_loader=train_loader,
            epochs=1,
            lr=0.01,
        )

        assert results['val_loss'] is None
        assert results['val_accuracy'] is None

    def test_train_with_different_optimizers(self, model, train_loader):
        """Test training with different optimizers."""
        trainer = LocalTrainer(device='cpu')

        # Test SGD
        results_sgd = trainer.train(
            model=model,
            train_loader=train_loader,
            epochs=1,
            optimizer_name='sgd',
        )
        assert results_sgd['train_loss'] >= 0

        # Test Adam
        model2 = SimpleCNN(num_classes=10)
        results_adam = trainer.train(
            model=model2,
            train_loader=train_loader,
            epochs=1,
            optimizer_name='adam',
        )
        assert results_adam['train_loss'] >= 0

    def test_train_with_momentum(self, model, train_loader):
        """Test training with momentum."""
        trainer = LocalTrainer(device='cpu')

        results = trainer.train(
            model=model,
            train_loader=train_loader,
            epochs=1,
            optimizer_name='sgd',
            momentum=0.9,
        )

        assert results['train_loss'] >= 0

    def test_train_with_weight_decay(self, model, train_loader):
        """Test training with weight decay."""
        trainer = LocalTrainer(device='cpu')

        results = trainer.train(
            model=model,
            train_loader=train_loader,
            epochs=1,
            weight_decay=0.0001,
        )

        assert results['train_loss'] >= 0

    def test_train_modifies_model_weights(self, model, train_loader):
        """Test that training actually modifies model weights."""
        trainer = LocalTrainer(device='cpu')

        # Get initial weights
        initial_weights = {
            name: param.clone()
            for name, param in model.named_parameters()
        }

        # Train
        trainer.train(
            model=model,
            train_loader=train_loader,
            epochs=2,
            lr=0.01,
        )

        # Check that weights changed
        weights_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_weights[name]):
                weights_changed = True
                break

        assert weights_changed, "Model weights should change during training"


class TestLocalTrainerValidate:
    """Test validation functionality."""

    def test_validate(self, model, val_loader):
        """Test validation."""
        trainer = LocalTrainer(device='cpu')

        metrics = trainer.validate(
            model=model,
            val_loader=val_loader,
        )

        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'num_samples' in metrics
        assert metrics['num_samples'] == len(val_loader.dataset)

    def test_validate_with_criterion(self, model, val_loader):
        """Test validation with specific criterion."""
        trainer = LocalTrainer(device='cpu')
        criterion = torch.nn.CrossEntropyLoss()

        metrics = trainer.validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
        )

        assert metrics['loss'] >= 0

    def test_validate_does_not_modify_weights(self, model, val_loader):
        """Test that validation doesn't modify model weights."""
        trainer = LocalTrainer(device='cpu')

        # Get initial weights
        initial_weights = {
            name: param.clone()
            for name, param in model.named_parameters()
        }

        # Validate
        trainer.validate(model=model, val_loader=val_loader)

        # Check weights unchanged
        for name, param in model.named_parameters():
            assert torch.allclose(param, initial_weights[name])

    def test_validate_sets_eval_mode(self, model, val_loader):
        """Test that validation sets model to eval mode."""
        trainer = LocalTrainer(device='cpu')

        model.train()  # Ensure training mode
        assert model.training

        trainer.validate(model=model, val_loader=val_loader)

        # Model should be in eval mode after validation
        assert model.training is False


class TestLocalTrainerStateless:
    """Test that trainer is stateless."""

    def test_trainer_is_reusable(self, train_loader):
        """Test that same trainer can be used multiple times."""
        trainer = LocalTrainer(device='cpu')

        # Train first model
        model1 = SimpleCNN(num_classes=10)
        results1 = trainer.train(
            model=model1,
            train_loader=train_loader,
            epochs=1,
        )

        # Train second model
        model2 = SimpleCNN(num_classes=10)
        results2 = trainer.train(
            model=model2,
            train_loader=train_loader,
            epochs=1,
        )

        # Both should succeed
        assert results1['train_loss'] >= 0
        assert results2['train_loss'] >= 0

    def test_trainer_with_different_configs(self, model, train_loader):
        """Test trainer with different configs."""
        trainer = LocalTrainer(device='cpu')

        # Config 1
        results1 = trainer.train(
            model=model,
            train_loader=train_loader,
            epochs=1,
            lr=0.01,
        )

        # Config 2 (same model)
        results2 = trainer.train(
            model=model,
            train_loader=train_loader,
            epochs=2,
            lr=0.001,
        )

        assert results1['train_loss'] >= 0
        assert results2['train_loss'] >= 0
