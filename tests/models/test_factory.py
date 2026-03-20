"""
Unit tests for model factory.

Tests:
- Model creation by name
- Invalid model names
- Model info retrieval
- Registry operations
"""

import pytest

from aflf.models import create_model, get_model_info, list_available_models
from aflf.models.cnn import CNN, CNNLarge, SimpleCNN


class TestCreateModel:
    """Test create_model factory function."""

    def test_create_simple_cnn(self):
        """Test creating SimpleCNN."""
        model = create_model('simple_cnn', num_classes=10)
        assert isinstance(model, SimpleCNN)
        assert model.num_classes == 10

    def test_create_cnn(self):
        """Test creating CNN."""
        model = create_model('cnn', num_classes=10)
        assert isinstance(model, CNN)
        assert model.num_classes == 10

    def test_create_cnn_large(self):
        """Test creating CNNLarge."""
        model = create_model('cnn_large', num_classes=100)
        assert isinstance(model, CNNLarge)
        assert model.num_classes == 100

    def test_invalid_model_name(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError, match="Unknown model"):
            create_model('nonexistent_model')

    def test_with_extra_kwargs(self):
        """Test creating model with extra kwargs."""
        model = create_model('cnn', num_classes=10, use_batch_norm=True)
        assert isinstance(model, CNN)
        assert model.use_batch_norm

    def test_default_num_classes(self):
        """Test that models use default num_classes if not specified."""
        model = create_model('simple_cnn')
        assert model.num_classes == 10  # Default for SimpleCNN


class TestListAvailableModels:
    """Test list_available_models function."""

    def test_returns_dict(self):
        """Test that function returns dictionary."""
        models = list_available_models()
        assert isinstance(models, dict)

    def test_contains_all_models(self):
        """Test that all models are registered."""
        models = list_available_models()

        expected_models = ['simple_cnn', 'cnn', 'cnn_large']
        for model_name in expected_models:
            assert model_name in models

    def test_values_are_classes(self):
        """Test that dictionary values are model classes."""
        models = list_available_models()

        for model_class in models.values():
            assert callable(model_class)


class TestGetModelInfo:
    """Test get_model_info function."""

    def test_get_simple_cnn_info(self):
        """Test getting SimpleCNN info."""
        info = get_model_info('simple_cnn')

        assert info['name'] == 'simple_cnn'
        assert 'class' in info
        assert 'description' in info
        assert 'default_params' in info
        assert 'default_size_mb' in info

        # Check parameter count is reasonable
        assert 50000 < info['default_params'] < 70000

    def test_get_cnn_info(self):
        """Test getting CNN info."""
        info = get_model_info('cnn')

        assert info['name'] == 'cnn'
        assert 100000 < info['default_params'] < 150000

    def test_get_cnn_large_info(self):
        """Test getting CNNLarge info."""
        info = get_model_info('cnn_large')

        assert info['name'] == 'cnn_large'
        assert info['default_params'] > 1000000

    def test_invalid_model_name(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_info('nonexistent_model')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
