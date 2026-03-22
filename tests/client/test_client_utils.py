"""
Tests for client utilities.
"""

import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from aflf.client.client_utils import (
    count_model_parameters,
    count_trainable_parameters,
    get_criterion,
    get_device,
    get_model_weights,
    get_optimizer,
    set_model_weights,
    set_reproducibility,
)
from aflf.models import SimpleCNN


class TestWeightManagement:
    """Test weight extraction and loading."""

    def test_get_model_weights(self):
        """Test extracting weights from model."""
        model = SimpleCNN(num_classes=10)
        weights = get_model_weights(model)

        assert isinstance(weights, dict)
        assert len(weights) > 0
        assert all(isinstance(v, torch.Tensor) for v in weights.values())

        # Check some expected keys
        assert 'conv1.weight' in weights
        assert 'conv1.bias' in weights

    def test_get_model_weights_are_cpu_tensors(self):
        """Test that extracted weights are CPU tensors."""
        model = SimpleCNN(num_classes=10)

        # Test with CPU model
        weights = get_model_weights(model)
        assert all(w.device.type == 'cpu' for w in weights.values())

        # Test with GPU model if available
        if torch.cuda.is_available():
            model = model.cuda()
            weights = get_model_weights(model)
            assert all(w.device.type == 'cpu' for w in weights.values())

    def test_set_model_weights(self):
        """Test loading weights into model."""
        model1 = SimpleCNN(num_classes=10)
        model2 = SimpleCNN(num_classes=10)

        # Get weights from model1
        weights = get_model_weights(model1)

        # Modify model2's weights randomly
        for param in model2.parameters():
            param.data.fill_(99.0)

        # Load model1's weights into model2
        set_model_weights(model2, weights)

        # Check that weights match
        weights2 = get_model_weights(model2)
        for key in weights.keys():
            assert torch.allclose(weights[key], weights2[key])

    def test_weight_round_trip(self):
        """Test that weights survive extraction and loading."""
        model = SimpleCNN(num_classes=10)

        # Set some specific values
        with torch.no_grad():
            model.conv1.weight.fill_(1.234)
            model.conv1.bias.fill_(5.678)

        # Extract and reload
        weights = get_model_weights(model)
        set_model_weights(model, weights)

        # Check values preserved
        assert torch.allclose(
            model.conv1.weight,
            torch.ones_like(model.conv1.weight) * 1.234
        )
        assert torch.allclose(
            model.conv1.bias,
            torch.ones_like(model.conv1.bias) * 5.678
        )


class TestDeviceSelection:
    """Test device selection utilities."""

    def test_get_device_auto(self):
        """Test auto device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']

    def test_get_device_cpu(self):
        """Test forcing CPU device."""
        device = get_device('cpu')
        assert device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_device_cuda(self):
        """Test CUDA device."""
        device = get_device('cuda')
        assert device.type == 'cuda'

    def test_get_device_with_index(self):
        """Test device with index."""
        if torch.cuda.is_available():
            device = get_device('cuda:0')
            assert device.type == 'cuda'
            assert device.index == 0


class TestReproducibility:
    """Test reproducibility utilities."""

    def test_set_reproducibility_python(self):
        """Test that Python random is seeded."""
        set_reproducibility(42)
        val1 = random.random()

        set_reproducibility(42)
        val2 = random.random()

        assert val1 == val2

    def test_set_reproducibility_numpy(self):
        """Test that NumPy random is seeded."""
        set_reproducibility(42)
        val1 = np.random.rand()

        set_reproducibility(42)
        val2 = np.random.rand()

        assert val1 == val2

    def test_set_reproducibility_torch(self):
        """Test that PyTorch random is seeded."""
        set_reproducibility(42)
        val1 = torch.rand(1).item()

        set_reproducibility(42)
        val2 = torch.rand(1).item()

        assert val1 == val2


class TestParameterCounting:
    """Test parameter counting utilities."""

    def test_count_model_parameters(self):
        """Test counting total parameters."""
        model = SimpleCNN(num_classes=10)
        count = count_model_parameters(model)

        # SimpleCNN should have around 62K parameters
        assert count > 60000
        assert count < 65000

    def test_count_trainable_parameters(self):
        """Test counting trainable parameters."""
        model = SimpleCNN(num_classes=10)

        # Initially all parameters are trainable
        total = count_model_parameters(model)
        trainable = count_trainable_parameters(model)
        assert total == trainable

        # Freeze conv1
        for param in model.conv1.parameters():
            param.requires_grad = False

        # Now trainable should be less
        trainable_after = count_trainable_parameters(model)
        assert trainable_after < trainable

    def test_count_parameters_with_frozen_layers(self):
        """Test parameter counting with frozen layers."""
        model = SimpleCNN(num_classes=10)

        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False

        trainable = count_trainable_parameters(model)
        assert trainable == 0


class TestOptimizer:
    """Test optimizer creation."""

    def test_get_optimizer_sgd(self):
        """Test creating SGD optimizer."""
        model = SimpleCNN(num_classes=10)
        optimizer = get_optimizer(model, 'sgd', lr=0.01, momentum=0.9)

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.defaults['lr'] == 0.01
        assert optimizer.defaults['momentum'] == 0.9

    def test_get_optimizer_adam(self):
        """Test creating Adam optimizer."""
        model = SimpleCNN(num_classes=10)
        optimizer = get_optimizer(model, 'adam', lr=0.001)

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults['lr'] == 0.001

    def test_get_optimizer_adamw(self):
        """Test creating AdamW optimizer."""
        model = SimpleCNN(num_classes=10)
        optimizer = get_optimizer(model, 'adamw', lr=0.001, weight_decay=0.01)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults['lr'] == 0.001
        assert optimizer.defaults['weight_decay'] == 0.01

    def test_get_optimizer_unknown(self):
        """Test that unknown optimizer raises error."""
        model = SimpleCNN(num_classes=10)

        with pytest.raises(ValueError, match="Unknown optimizer"):
            get_optimizer(model, 'unknown_optimizer')

    def test_get_optimizer_case_insensitive(self):
        """Test that optimizer name is case-insensitive."""
        model = SimpleCNN(num_classes=10)

        opt1 = get_optimizer(model, 'SGD', lr=0.01)
        opt2 = get_optimizer(model, 'sgd', lr=0.01)
        opt3 = get_optimizer(model, 'Sgd', lr=0.01)

        assert type(opt1) == type(opt2) == type(opt3)


class TestCriterion:
    """Test criterion creation."""

    def test_get_criterion_cross_entropy(self):
        """Test creating cross entropy loss."""
        criterion = get_criterion('cross_entropy')
        assert isinstance(criterion, nn.CrossEntropyLoss)

    def test_get_criterion_mse(self):
        """Test creating MSE loss."""
        criterion = get_criterion('mse')
        assert isinstance(criterion, nn.MSELoss)

    def test_get_criterion_bce(self):
        """Test creating BCE loss."""
        criterion = get_criterion('bce')
        assert isinstance(criterion, nn.BCEWithLogitsLoss)

    def test_get_criterion_unknown(self):
        """Test that unknown criterion raises error."""
        with pytest.raises(ValueError, match="Unknown criterion"):
            get_criterion('unknown_criterion')

    def test_get_criterion_case_insensitive(self):
        """Test that criterion name is case-insensitive."""
        crit1 = get_criterion('CROSS_ENTROPY')
        crit2 = get_criterion('cross_entropy')
        crit3 = get_criterion('Cross_Entropy')

        assert type(crit1) == type(crit2) == type(crit3)

    def test_criterion_forward_pass(self):
        """Test that criterion works for forward pass."""
        criterion = get_criterion('cross_entropy')

        # Create dummy data
        output = torch.randn(10, 5)  # 10 samples, 5 classes
        target = torch.randint(0, 5, (10,))  # 10 labels

        # Compute loss
        loss = criterion(output, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative
