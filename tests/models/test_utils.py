"""
Unit tests for model utilities.

Tests:
- Parameter counting
- Model size calculation
- Model initialization
- Weight saving/loading
- Model comparison
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from aflf.models import SimpleCNN
from aflf.models.utils import (
    compare_models,
    count_parameters,
    freeze_layers,
    get_model_size_mb,
    initialize_model,
    load_model_weights,
    save_model_weights,
)


class TestCountParameters:
    """Test parameter counting utility."""

    def test_count_all_parameters(self):
        """Test counting all parameters."""
        model = SimpleCNN()
        count = count_parameters(model)

        # Should match model's own count
        assert count == model.get_num_parameters()

    def test_count_trainable_only(self):
        """Test counting only trainable parameters."""
        model = SimpleCNN()

        # Freeze one parameter
        first_param = next(model.parameters())
        first_param.requires_grad = False

        total = count_parameters(model, trainable_only=False)
        trainable = count_parameters(model, trainable_only=True)

        assert trainable < total


class TestModelSize:
    """Test model size calculation."""

    def test_get_model_size_mb(self):
        """Test model size in MB."""
        model = SimpleCNN()
        size_mb = get_model_size_mb(model)

        # Should match model's own calculation
        assert size_mb == model.get_model_size_mb()

        # Should be reasonable size
        assert 0.1 < size_mb < 1.0


class TestInitializeModel:
    """Test model initialization."""

    def test_deterministic_initialization(self):
        """Test that same seed produces identical initialization."""
        model1 = SimpleCNN()
        model1 = initialize_model(model1, seed=42)

        model2 = SimpleCNN()
        model2 = initialize_model(model2, seed=42)

        # Parameters should be identical
        params1 = model1.get_parameters()
        params2 = model2.get_parameters()

        for p1, p2 in zip(params1, params2):
            assert torch.allclose(
                torch.from_numpy(p1),
                torch.from_numpy(p2),
                atol=1e-10
            )

    def test_different_seeds_different_init(self):
        """Test that different seeds produce different initialization."""
        model1 = SimpleCNN()
        model1 = initialize_model(model1, seed=42)

        model2 = SimpleCNN()
        model2 = initialize_model(model2, seed=999)

        # Parameters should be different
        params1 = model1.get_parameters()
        params2 = model2.get_parameters()

        # At least one parameter should be significantly different
        max_diff = max(abs(p1 - p2).max() for p1, p2 in zip(params1, params2))
        assert max_diff > 0.01

    def test_initialization_methods(self):
        """Test different initialization methods."""
        methods = ['kaiming', 'xavier', 'normal']

        for method in methods:
            model = SimpleCNN()
            model = initialize_model(model, seed=42, method=method)

            # Should not raise error
            assert model is not None

    def test_invalid_initialization_method(self):
        """Test that invalid method raises error."""
        model = SimpleCNN()

        with pytest.raises(ValueError, match="Unknown initialization method"):
            initialize_model(model, method='invalid')


class TestSaveLoadWeights:
    """Test weight saving and loading."""

    def test_save_weights(self):
        """Test saving model weights."""
        model = SimpleCNN()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"
            save_model_weights(model, save_path)

            # File should exist
            assert save_path.exists()

    def test_load_weights(self):
        """Test loading model weights."""
        model1 = SimpleCNN()
        model1 = initialize_model(model1, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"

            # Save from model1
            save_model_weights(model1, save_path)

            # Load into model2
            model2 = SimpleCNN()
            load_model_weights(model2, save_path)

            # Weights should match
            params1 = model1.get_parameters()
            params2 = model2.get_parameters()

            for p1, p2 in zip(params1, params2):
                assert torch.allclose(
                    torch.from_numpy(p1),
                    torch.from_numpy(p2),
                    atol=1e-7
                )

    def test_load_nonexistent_file(self):
        """Test that loading from nonexistent file raises error."""
        model = SimpleCNN()

        with pytest.raises(FileNotFoundError):
            load_model_weights(model, "nonexistent_file.pt")

    def test_save_with_metadata(self):
        """Test saving with metadata."""
        model = SimpleCNN()
        metadata = {'epoch': 10, 'accuracy': 0.95}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"
            save_model_weights(model, save_path, metadata=metadata)

            # Load and check metadata
            loaded_metadata = load_model_weights(model, save_path)
            assert loaded_metadata == metadata

    def test_save_with_optimizer(self):
        """Test saving with optimizer state."""
        model = SimpleCNN()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"
            save_model_weights(
                model,
                save_path,
                include_optimizer=True,
                optimizer=optimizer,
            )

            # Load with optimizer
            new_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            load_model_weights(model, save_path, optimizer=new_optimizer)

            # Should not raise error
            assert True


class TestFreezeeLayers:
    """Test layer freezing utility."""

    def test_freeze_all(self):
        """Test freezing all layers."""
        model = SimpleCNN()
        model = freeze_layers(model, freeze_all=True)

        # All parameters should be frozen
        for param in model.parameters():
            assert not param.requires_grad

    def test_freeze_specific_layers(self):
        """Test freezing specific layers."""
        model = SimpleCNN()
        model = freeze_layers(model, layer_names=['conv1'])

        # conv1 parameters should be frozen
        for name, param in model.named_parameters():
            if name.startswith('conv1'):
                assert not param.requires_grad
            else:
                assert param.requires_grad

    def test_freeze_multiple_layers(self):
        """Test freezing multiple layers."""
        model = SimpleCNN()
        model = freeze_layers(model, layer_names=['conv1', 'conv2'])

        # conv1 and conv2 should be frozen
        frozen_count = sum(
            1 for name, param in model.named_parameters()
            if not param.requires_grad
        )

        assert frozen_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
