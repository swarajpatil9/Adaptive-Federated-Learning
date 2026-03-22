"""
Tests for MetricsTracker class.
"""

import pytest

from aflf.client.metrics import MetricsTracker


class TestMetricsTracker:
    """Test cases for MetricsTracker."""

    def test_init(self):
        """Test tracker initialization."""
        tracker = MetricsTracker()
        assert tracker.num_samples == 0
        assert tracker.loss == 0.0
        assert tracker.accuracy == 0.0

    def test_update_with_loss_and_accuracy(self):
        """Test updating with loss and accuracy."""
        tracker = MetricsTracker()
        tracker.update(loss=0.5, accuracy=0.8, num_samples=10)

        assert tracker.num_samples == 10
        assert tracker.loss == 0.5
        assert tracker.accuracy == 0.8

    def test_update_multiple_batches(self):
        """Test updating with multiple batches."""
        tracker = MetricsTracker()

        # Batch 1: 10 samples, loss=0.5, acc=0.8
        tracker.update(loss=0.5, accuracy=0.8, num_samples=10)

        # Batch 2: 20 samples, loss=0.3, acc=0.9
        tracker.update(loss=0.3, accuracy=0.9, num_samples=20)

        # Average should be weighted by samples
        # Loss: (0.5*10 + 0.3*20) / 30 = (5 + 6) / 30 = 11/30 ≈ 0.3667
        # Acc: (0.8*10 + 0.9*20) / 30 = (8 + 18) / 30 = 26/30 ≈ 0.8667

        assert tracker.num_samples == 30
        assert abs(tracker.loss - 11/30) < 1e-6
        assert abs(tracker.accuracy - 26/30) < 1e-6

    def test_accuracy_percentage_format(self):
        """Test that accuracy handles both 0-1 and 0-100 formats."""
        tracker = MetricsTracker()

        # Test 0-1 format
        tracker.update(loss=0.5, accuracy=0.85, num_samples=10)
        assert abs(tracker.accuracy - 0.85) < 1e-6

        # Test 0-100 format (should be converted to 0-1)
        tracker.reset()
        tracker.update(loss=0.5, accuracy=85.0, num_samples=10)
        assert abs(tracker.accuracy - 0.85) < 1e-6

    def test_get_metrics(self):
        """Test getting metrics dictionary."""
        tracker = MetricsTracker()
        tracker.update(loss=0.4, accuracy=0.85, num_samples=32)
        tracker.update(loss=0.6, accuracy=0.75, num_samples=16)

        metrics = tracker.get_metrics()

        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'num_samples' in metrics
        assert metrics['num_samples'] == 48

        # Loss: (0.4*32 + 0.6*16) / 48 = (12.8 + 9.6) / 48 = 0.4667
        expected_loss = (0.4*32 + 0.6*16) / 48
        assert abs(metrics['loss'] - expected_loss) < 1e-6

    def test_custom_metrics(self):
        """Test tracking custom metrics."""
        tracker = MetricsTracker()
        tracker.update(loss=0.5, accuracy=0.8, num_samples=10, f1_score=0.75)
        tracker.update(loss=0.3, accuracy=0.9, num_samples=20, f1_score=0.85)

        metrics = tracker.get_metrics()

        assert 'f1_score' in metrics
        # F1: (0.75*10 + 0.85*20) / 30 = (7.5 + 17) / 30 = 0.8167
        expected_f1 = (0.75*10 + 0.85*20) / 30
        assert abs(metrics['f1_score'] - expected_f1) < 1e-6

    def test_reset(self):
        """Test resetting tracker."""
        tracker = MetricsTracker()
        tracker.update(loss=0.5, accuracy=0.8, num_samples=10)

        assert tracker.num_samples == 10

        tracker.reset()

        assert tracker.num_samples == 0
        assert tracker.loss == 0.0
        assert tracker.accuracy == 0.0

    def test_empty_tracker_metrics(self):
        """Test getting metrics from empty tracker."""
        tracker = MetricsTracker()
        metrics = tracker.get_metrics()

        assert metrics['loss'] == 0.0
        assert metrics['accuracy'] == 0.0
        assert metrics['num_samples'] == 0

    def test_properties(self):
        """Test tracker properties."""
        tracker = MetricsTracker()
        tracker.update(loss=0.5, accuracy=0.8, num_samples=10)

        assert tracker.num_samples == 10
        assert tracker.loss == 0.5
        assert tracker.accuracy == 0.8

    def test_repr(self):
        """Test string representation."""
        tracker = MetricsTracker()
        tracker.update(loss=0.5, accuracy=0.8, num_samples=10)

        repr_str = repr(tracker)
        assert 'MetricsTracker' in repr_str
        assert '0.5' in repr_str or '0.50' in repr_str
        assert '0.8' in repr_str or '0.80' in repr_str
