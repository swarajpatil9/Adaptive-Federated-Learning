"""
Unit tests for BaseModel class.

Tests:
- Parameter extraction/loading
- Model metadata (size, count)
- Device management
- Model summary
"""

import numpy as np
import pytest
import torch

from aflf.models import SimpleCNN


@pytest.fixture
def simple_model():
    """Create a SimpleCNN instance for testing."""
    return SimpleCNN(num_classes=10)


class TestParameterExtraction:
    """Test parameter extraction and loading."""

    def test_get_parameters(self, simple_model):
        """Test parameter extraction."""
        params = simple_model.get_parameters()

        # Should return list of numpy arrays
        assert isinstance(params, list)
        assert len(params) > 0
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_set_parameters(self, simple_model):
        """Test parameter loading."""
        # Get original parameters
        params = simple_model.get_parameters()

        # Modify parameters
        modified_params = [p * 0.9 for p in params]

        # Load modified parameters
        simple_model.set_parameters(modified_params)

        # Verify parameters changed
        new_params = simple_model.get_parameters()
        for p1, p2 in zip(modified_params, new_params):
            assert np.allclose(p1, p2, atol=1e-7)

    def test_set_parameters_wrong_count(self, simple_model):
        """Test that wrong number of parameters raises error."""
        params = simple_model.get_parameters()

        # Try to set with wrong number
        with pytest.raises(ValueError, match="Parameter count mismatch"):
            simple_model.set_parameters(params[:-1])  # Missing one parameter


class TestModelMetadata:
    """Test model metadata functions."""

    def test_get_num_parameters(self, simple_model):
        """Test parameter counting."""
        num_params = simple_model.get_num_parameters()

        # SimpleCNN should have ~62K parameters
        assert 60000 < num_params < 65000

        # Should match manual count
        manual_count = sum(p.numel() for p in simple_model.parameters())
        assert num_params == manual_count

    def test_get_num_parameters_trainable_only(self, simple_model):
        """Test counting only trainable parameters."""
        # Freeze one parameter
        first_param = next(simple_model.parameters())
        first_param.requires_grad = False

        total = simple_model.get_num_parameters(trainable_only=False)
        trainable = simple_model.get_num_parameters(trainable_only=True)

        assert trainable < total
        assert trainable == total - first_param.numel()

    def test_get_model_size(self, simple_model):
        """Test model size calculation."""
        size_bytes = simple_model.get_model_size()

        # Should be num_params * 4 bytes (float32)
        num_params = simple_model.get_num_parameters()
        expected_size = num_params * 4

        assert size_bytes == expected_size

    def test_get_model_size_mb(self, simple_model):
        """Test model size in MB."""
        size_mb = simple_model.get_model_size_mb()

        # SimpleCNN should be around 0.24 MB
        assert 0.20 < size_mb < 0.30

    def test_get_layer_shapes(self, simple_model):
        """Test layer shape extraction."""
        shapes = simple_model.get_layer_shapes()

        assert isinstance(shapes, list)
        assert len(shapes) > 0

        # Each entry should be (name, shape) tuple
        for name, shape in shapes:
            assert isinstance(name, str)
            assert isinstance(shape, tuple)


class TestDeviceManagement:
    """Test device management."""

    def test_to_device_cpu(self, simple_model):
        """Test moving to CPU."""
        simple_model = simple_model.to_device(torch.device('cpu'))
        assert simple_model.get_device().type == 'cpu'

    def test_to_device_cuda(self, simple_model):
        """Test moving to CUDA (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        simple_model = simple_model.to_device(torch.device('cuda'))
        assert simple_model.get_device().type == 'cuda'

        # Verify parameters are on CUDA
        for param in simple_model.parameters():
            assert param.device.type == 'cuda'

    def test_to_device_mps(self, simple_model):
        """Test moving to MPS (if available)."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        simple_model = simple_model.to_device(torch.device('mps'))
        assert simple_model.get_device().type == 'mps'


class TestModelSummary:
    """Test model summary generation."""

    def test_summary(self, simple_model):
        """Test summary string generation."""
        summary = simple_model.summary()

        assert isinstance(summary, str)
        assert 'SimpleCNN' in summary
        assert 'parameters' in summary.lower()

    def test_repr(self, simple_model):
        """Test __repr__ method."""
        repr_str = repr(simple_model)

        assert 'SimpleCNN' in repr_str
        assert 'parameters=' in repr_str
        assert 'size=' in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
