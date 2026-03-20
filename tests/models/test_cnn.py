"""
Unit tests for CNN models.

Tests:
- SimpleCNN forward pass
- CNN forward pass
- CNNLarge forward pass
- Different input/output shapes
- Batch normalization variants
"""

import pytest
import torch

from aflf.models import CNN, CNNLarge, SimpleCNN


class TestSimpleCNN:
    """Test SimpleCNN model."""

    def test_forward_mnist(self):
        """Test forward pass with MNIST input."""
        model = SimpleCNN(num_classes=10)
        x = torch.randn(16, 1, 28, 28)
        logits = model(x)

        assert logits.shape == (16, 10)

    def test_different_num_classes(self):
        """Test with different number of classes."""
        model = SimpleCNN(num_classes=5)
        x = torch.randn(8, 1, 28, 28)
        logits = model(x)

        assert logits.shape == (8, 5)

    def test_single_sample(self):
        """Test with single sample (batch_size=1)."""
        model = SimpleCNN()
        x = torch.randn(1, 1, 28, 28)
        logits = model(x)

        assert logits.shape == (1, 10)

    def test_dropout_rate(self):
        """Test with different dropout rates."""
        model = SimpleCNN(dropout_rate=0.3)
        x = torch.randn(16, 1, 28, 28)
        logits = model(x)

        assert logits.shape == (16, 10)

    def test_eval_mode(self):
        """Test eval mode (dropout disabled)."""
        model = SimpleCNN()
        model.eval()

        x = torch.randn(16, 1, 28, 28)

        # Multiple passes should give same result in eval mode
        with torch.no_grad():
            logits1 = model(x)
            logits2 = model(x)

        assert torch.allclose(logits1, logits2)


class TestCNN:
    """Test CNN model."""

    def test_forward_cifar10(self):
        """Test forward pass with CIFAR-10 input."""
        model = CNN(num_classes=10)
        x = torch.randn(16, 3, 32, 32)
        logits = model(x)

        assert logits.shape == (16, 10)

    def test_with_batch_norm(self):
        """Test CNN with batch normalization."""
        model = CNN(num_classes=10, use_batch_norm=True)
        x = torch.randn(16, 3, 32, 32)
        logits = model(x)

        assert logits.shape == (16, 10)

        # Check that batch norm layers exist
        has_bn = any('bn' in name for name, _ in model.named_modules())
        assert has_bn

    def test_without_batch_norm(self):
        """Test CNN without batch normalization."""
        model = CNN(num_classes=10, use_batch_norm=False)
        x = torch.randn(16, 3, 32, 32)
        logits = model(x)

        assert logits.shape == (16, 10)

    def test_cifar100(self):
        """Test with CIFAR-100 (100 classes)."""
        model = CNN(num_classes=100)
        x = torch.randn(16, 3, 32, 32)
        logits = model(x)

        assert logits.shape == (16, 100)


class TestCNNLarge:
    """Test CNNLarge model."""

    def test_forward_cifar10(self):
        """Test forward pass with CIFAR-10."""
        model = CNNLarge(num_classes=10)
        x = torch.randn(16, 3, 32, 32)
        logits = model(x)

        assert logits.shape == (16, 10)

    def test_forward_cifar100(self):
        """Test forward pass with CIFAR-100."""
        model = CNNLarge(num_classes=100)
        x = torch.randn(16, 3, 32, 32)
        logits = model(x)

        assert logits.shape == (16, 100)

    def test_with_batch_norm(self):
        """Test CNNLarge with batch norm (default)."""
        model = CNNLarge(num_classes=10, use_batch_norm=True)
        x = torch.randn(16, 3, 32, 32)
        logits = model(x)

        assert logits.shape == (16, 10)

    def test_parameter_count(self):
        """Test that CNNLarge has more parameters than CNN."""
        model_cnn = CNN(num_classes=10)
        model_large = CNNLarge(num_classes=10)

        params_cnn = model_cnn.get_num_parameters()
        params_large = model_large.get_num_parameters()

        # CNNLarge should have significantly more parameters
        assert params_large > params_cnn * 5


class TestModelComparison:
    """Test comparative properties across models."""

    def test_parameter_counts(self):
        """Test that parameter counts match expected ranges."""
        models = {
            'SimpleCNN': SimpleCNN(num_classes=10),
            'CNN': CNN(num_classes=10),
            'CNNLarge': CNNLarge(num_classes=10),
        }

        expected_ranges = {
            'SimpleCNN': (50000, 70000),
            'CNN': (100000, 150000),
            'CNNLarge': (1000000, 1500000),
        }

        for name, model in models.items():
            num_params = model.get_num_parameters()
            min_expected, max_expected = expected_ranges[name]
            assert min_expected < num_params < max_expected, \
                f"{name} has {num_params} params, expected {min_expected}-{max_expected}"

    def test_model_sizes(self):
        """Test model sizes increase as expected."""
        simple = SimpleCNN(num_classes=10)
        cnn = CNN(num_classes=10)
        large = CNNLarge(num_classes=10)

        size_simple = simple.get_model_size_mb()
        size_cnn = cnn.get_model_size_mb()
        size_large = large.get_model_size_mb()

        # Sizes should increase
        assert size_simple < size_cnn < size_large

        # SimpleCNN should be under 0.5 MB
        assert size_simple < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
