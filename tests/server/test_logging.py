"""
Tests for server logging system.
"""

import json
import tempfile
from pathlib import Path

import pytest

from aflf.server.logger import ConsoleProgressLogger, ServerLogger
from aflf.server.metrics_tracker import MetricsTracker, ProgressTracker
from aflf.server.round_manager import RoundState


class TestServerLogger:
    """Tests for ServerLogger."""

    def test_initialization(self):
        """Test logger initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ServerLogger(
                experiment_name="test_exp",
                log_dir=tmpdir,
                enable_file_logging=True,
            )
            assert logger.experiment_name == "test_exp"
            assert logger.enable_file_logging is True

    def test_log_server_init(self):
        """Test logging server initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ServerLogger(
                experiment_name="test_exp",
                log_dir=tmpdir,
            )
            # Should not raise
            logger.log_server_init(
                num_clients=10,
                model_name="TestModel",
                selection_strategy="RandomSelection",
                config={'param1': 'value1'},
            )

    def test_log_round_lifecycle(self):
        """Test logging complete round lifecycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ServerLogger(
                experiment_name="test_exp",
                log_dir=tmpdir,
            )

            # Start round
            logger.log_round_start(round_num=0, num_selected=5)

            # Client events
            logger.log_client_start(round_num=0, client_id=0)
            logger.log_client_success(
                round_num=0,
                client_id=0,
                train_loss=0.5,
                train_acc=0.85,
                duration=10.5,
            )
            logger.log_client_failure(round_num=0, client_id=1, reason="timeout")

            # End round
            round_state = RoundState(round_num=0, selected_clients=[0, 1])
            round_state.participating_clients = [0]
            round_state.dropped_clients = [1]
            round_state.finalize()

            metrics = {'avg_train_loss': 0.5, 'avg_train_accuracy': 0.85}
            logger.log_round_end(round_state, metrics)

    def test_log_training_complete(self):
        """Test logging training completion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ServerLogger(
                experiment_name="test_exp",
                log_dir=tmpdir,
            )
            logger.log_training_complete(
                total_rounds=10,
                total_time=100.0,
                final_metrics={'final_loss': 0.3, 'final_accuracy': 0.90},
            )

    def test_log_file_creation(self):
        """Test that log file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ServerLogger(
                experiment_name="test_exp",
                log_dir=tmpdir,
                enable_file_logging=True,
            )
            logger.log_info("Test message")

            # Check that log file exists
            log_files = list(Path(tmpdir).glob("*.log"))
            assert len(log_files) > 0


class TestConsoleProgressLogger:
    """Tests for ConsoleProgressLogger."""

    def test_initialization(self):
        """Test progress logger initialization."""
        progress = ConsoleProgressLogger(total_rounds=10)
        assert progress.total_rounds == 10
        assert progress.bar_width == 50

    def test_update_round(self):
        """Test updating progress."""
        progress = ConsoleProgressLogger(total_rounds=10)

        # Should not raise
        progress.update_round(
            round_num=0,
            metrics={'avg_train_loss': 0.5, 'avg_train_accuracy': 0.85},
            num_participating=8,
            num_selected=10,
        )

    def test_finish(self):
        """Test finishing progress bar."""
        progress = ConsoleProgressLogger(total_rounds=10)
        progress.finish()  # Should print newline


class TestMetricsTracker:
    """Tests for MetricsTracker."""

    def test_initialization(self):
        """Test tracker initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(
                experiment_name="test_exp",
                output_dir=tmpdir,
                enable_tensorboard=False,
            )
            assert tracker.experiment_name == "test_exp"
            assert len(tracker.round_metrics) == 0

    def test_record_round(self):
        """Test recording round metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(
                experiment_name="test_exp",
                output_dir=tmpdir,
                enable_tensorboard=False,
            )

            round_state = RoundState(round_num=0, selected_clients=[0, 1, 2])
            round_state.participating_clients = [0, 1]
            round_state.finalize()

            metrics = {'avg_train_loss': 0.5, 'avg_train_accuracy': 0.85}
            tracker.record_round(round_state, metrics)

            assert len(tracker.round_metrics) == 1
            assert tracker.round_metrics[0]['round'] == 0
            assert tracker.round_metrics[0]['avg_train_loss'] == 0.5

    def test_record_client_result(self):
        """Test recording client results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(
                experiment_name="test_exp",
                output_dir=tmpdir,
                enable_tensorboard=False,
            )

            result_dict = {
                'train_loss': 0.5,
                'train_accuracy': 0.85,
                'num_samples': 100,
            }
            tracker.record_client_result(round_num=0, client_id=0, result_dict=result_dict)

            assert 0 in tracker.client_metrics
            assert len(tracker.client_metrics[0]) == 1
            assert tracker.client_metrics[0][0]['train_loss'] == 0.5

    def test_compute_summary(self):
        """Test computing summary statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(
                experiment_name="test_exp",
                output_dir=tmpdir,
                enable_tensorboard=False,
            )

            # Record multiple rounds
            for round_num in range(3):
                round_state = RoundState(round_num=round_num, selected_clients=[0, 1])
                round_state.participating_clients = [0, 1]
                round_state.finalize()

                metrics = {
                    'avg_train_loss': 0.5 - round_num * 0.1,
                    'avg_train_accuracy': 0.7 + round_num * 0.1,
                }
                tracker.record_round(round_state, metrics)

            summary = tracker.compute_summary()

            assert summary['num_rounds'] == 3
            assert 'total_time_seconds' in summary
            assert 'avg_round_time' in summary
            assert 'final_train_loss' in summary
            assert 'final_train_accuracy' in summary

    def test_export_json(self):
        """Test JSON export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(
                experiment_name="test_exp",
                output_dir=tmpdir,
                enable_tensorboard=False,
            )

            # Record a round
            round_state = RoundState(round_num=0, selected_clients=[0])
            round_state.finalize()
            metrics = {'avg_train_loss': 0.5}
            tracker.record_round(round_state, metrics)

            # Export
            json_path = tracker.export_json()
            assert Path(json_path).exists()

            # Verify content
            with open(json_path, 'r') as f:
                data = json.load(f)
                assert 'experiment_name' in data
                assert 'round_metrics' in data
                assert len(data['round_metrics']) == 1

    def test_export_csv(self):
        """Test CSV export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(
                experiment_name="test_exp",
                output_dir=tmpdir,
                enable_tensorboard=False,
            )

            # Record rounds
            for round_num in range(2):
                round_state = RoundState(round_num=round_num, selected_clients=[0])
                round_state.finalize()
                metrics = {'avg_train_loss': 0.5}
                tracker.record_round(round_state, metrics)

            # Export
            csv_path = tracker.export_csv()
            assert Path(csv_path).exists()

            # Verify it's a valid CSV
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 3  # Header + 2 rows

    def test_export_client_csv(self):
        """Test client CSV export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(
                experiment_name="test_exp",
                output_dir=tmpdir,
                enable_tensorboard=False,
            )

            # Record client results
            for client_id in range(2):
                result_dict = {'train_loss': 0.5, 'num_samples': 100}
                tracker.record_client_result(
                    round_num=0, client_id=client_id, result_dict=result_dict
                )

            # Export
            csv_path = tracker.export_client_csv()
            assert Path(csv_path).exists()


class TestProgressTracker:
    """Tests for ProgressTracker."""

    def test_initialization(self):
        """Test progress tracker initialization."""
        tracker = ProgressTracker(convergence_window=5, convergence_threshold=0.01)
        assert tracker.convergence_window == 5
        assert tracker.convergence_threshold == 0.01

    def test_update(self):
        """Test updating metrics."""
        tracker = ProgressTracker()
        tracker.update(loss=0.5, accuracy=0.85)

        assert len(tracker.loss_history) == 1
        assert len(tracker.accuracy_history) == 1
        assert tracker.loss_history[0] == 0.5
        assert tracker.accuracy_history[0] == 0.85

    def test_has_converged_not_enough_data(self):
        """Test convergence detection with insufficient data."""
        tracker = ProgressTracker(convergence_window=5)

        # Add less than window size
        for i in range(3):
            tracker.update(loss=0.5, accuracy=0.85)

        assert not tracker.has_converged()

    def test_has_converged_true(self):
        """Test convergence detection when converged."""
        tracker = ProgressTracker(convergence_window=5, convergence_threshold=0.01)

        # Add stable values (low variance)
        for i in range(10):
            tracker.update(loss=0.5 + i * 0.001, accuracy=0.85)

        assert tracker.has_converged(metric='loss')

    def test_has_converged_false(self):
        """Test convergence detection when not converged."""
        tracker = ProgressTracker(convergence_window=5, convergence_threshold=0.01)

        # Add varying values (high variance)
        for i in range(10):
            tracker.update(loss=0.5 + i * 0.1, accuracy=0.85)

        assert not tracker.has_converged(metric='loss')

    def test_get_improvement(self):
        """Test improvement calculation."""
        tracker = ProgressTracker()

        # Add improving metrics
        for i in range(10):
            tracker.update(loss=1.0 - i * 0.1, accuracy=0.5 + i * 0.05)

        improvement = tracker.get_improvement(window=5)

        assert improvement['loss_improvement'] > 0  # Loss decreased
        assert improvement['accuracy_improvement'] > 0  # Accuracy increased

    def test_get_best(self):
        """Test getting best metrics."""
        tracker = ProgressTracker()

        tracker.update(loss=0.5, accuracy=0.80)
        tracker.update(loss=0.3, accuracy=0.85)
        tracker.update(loss=0.4, accuracy=0.82)

        best = tracker.get_best()

        assert best['best_loss'] == 0.3
        assert best['best_accuracy'] == 0.85

    def test_get_best_empty(self):
        """Test getting best metrics with no data."""
        tracker = ProgressTracker()
        best = tracker.get_best()

        assert best['best_loss'] == float('inf')
        assert best['best_accuracy'] == 0.0
