"""
Tests for ClientManager.
"""

import pytest

from aflf.client.client import TrainingResult
from aflf.server.client_manager import ClientManager, ClientMetadata


class TestClientManager:
    """Tests for ClientManager class."""

    def test_initialization(self):
        """Test ClientManager initialization."""
        manager = ClientManager()
        assert manager.get_num_clients() == 0
        assert manager.get_all_clients() == []
        assert manager.get_available_clients() == []

    def test_register_client(self):
        """Test client registration."""
        manager = ClientManager()

        manager.register_client(client_id=0, dataset_size=600)
        assert manager.get_num_clients() == 1
        assert manager.get_all_clients() == [0]
        assert manager.get_available_clients() == [0]

        metadata = manager.get_client_metadata(0)
        assert metadata.client_id == 0
        assert metadata.dataset_size == 600
        assert metadata.is_available is True
        assert metadata.participation_count == 0
        assert metadata.failure_count == 0

    def test_register_duplicate_client_raises_error(self):
        """Test that registering duplicate client raises error."""
        manager = ClientManager()
        manager.register_client(client_id=0, dataset_size=600)

        with pytest.raises(ValueError, match="already registered"):
            manager.register_client(client_id=0, dataset_size=500)

    def test_register_multiple_clients(self):
        """Test registering multiple clients."""
        manager = ClientManager()

        for i in range(5):
            manager.register_client(client_id=i, dataset_size=500 + i * 10)

        assert manager.get_num_clients() == 5
        assert manager.get_all_clients() == [0, 1, 2, 3, 4]
        assert manager.get_available_clients() == [0, 1, 2, 3, 4]

    def test_set_availability(self):
        """Test setting client availability."""
        manager = ClientManager()
        manager.register_client(client_id=0, dataset_size=600)
        manager.register_client(client_id=1, dataset_size=550)

        assert manager.get_available_clients() == [0, 1]

        manager.set_availability(client_id=1, is_available=False)
        assert manager.get_available_clients() == [0]

        manager.set_availability(client_id=0, is_available=False)
        assert manager.get_available_clients() == []

        manager.set_availability(client_id=0, is_available=True)
        assert manager.get_available_clients() == [0]

    def test_update_from_result(self):
        """Test updating client metadata from training result."""
        import torch
        from collections import OrderedDict

        manager = ClientManager()
        manager.register_client(client_id=0, dataset_size=600)

        result = TrainingResult(
            client_id=0,
            weights=OrderedDict(),
            num_samples=600,
            train_loss=0.5,
            train_accuracy=0.85,
            val_loss=None,
            val_accuracy=None,
            training_time=10.5,
        )

        manager.update_from_result(result, round_num=0)

        metadata = manager.get_client_metadata(0)
        assert metadata.last_accuracy == 0.85
        assert metadata.last_loss == 0.5
        assert metadata.participation_count == 1
        assert metadata.total_training_time == 10.5
        assert metadata.last_round_participated == 0

    def test_update_from_multiple_results(self):
        """Test updating client metadata from multiple results."""
        from collections import OrderedDict

        manager = ClientManager()
        manager.register_client(client_id=0, dataset_size=600)

        # Round 0
        result1 = TrainingResult(
            client_id=0,
            weights=OrderedDict(),
            num_samples=600,
            train_loss=0.5,
            train_accuracy=0.85,
            val_loss=None,
            val_accuracy=None,
            training_time=10.5,
        )
        manager.update_from_result(result1, round_num=0)

        # Round 2 (skipped round 1)
        result2 = TrainingResult(
            client_id=0,
            weights=OrderedDict(),
            num_samples=600,
            train_loss=0.3,
            train_accuracy=0.90,
            val_loss=None,
            val_accuracy=None,
            training_time=9.5,
        )
        manager.update_from_result(result2, round_num=2)

        metadata = manager.get_client_metadata(0)
        assert metadata.last_accuracy == 0.90  # Latest
        assert metadata.last_loss == 0.3  # Latest
        assert metadata.participation_count == 2
        assert metadata.total_training_time == 20.0
        assert metadata.last_round_participated == 2

    def test_record_failure(self):
        """Test recording client failures."""
        manager = ClientManager()
        manager.register_client(client_id=0, dataset_size=600)

        assert manager.get_client_metadata(0).failure_count == 0

        manager.record_failure(client_id=0)
        assert manager.get_client_metadata(0).failure_count == 1

        manager.record_failure(client_id=0)
        assert manager.get_client_metadata(0).failure_count == 2

    def test_get_client_metadata_raises_error_for_unregistered(self):
        """Test that getting metadata for unregistered client raises error."""
        manager = ClientManager()

        with pytest.raises(ValueError, match="not registered"):
            manager.get_client_metadata(999)

    def test_summary_stats(self):
        """Test getting summary statistics."""
        manager = ClientManager()

        # Empty manager
        stats = manager.get_summary_stats()
        assert stats['total_clients'] == 0
        assert stats['available_clients'] == 0
        assert stats['total_dataset_size'] == 0

        # Add clients
        manager.register_client(client_id=0, dataset_size=600)
        manager.register_client(client_id=1, dataset_size=400)
        manager.register_client(client_id=2, dataset_size=500, is_available=False)

        stats = manager.get_summary_stats()
        assert stats['total_clients'] == 3
        assert stats['available_clients'] == 2
        assert stats['total_dataset_size'] == 1500
        assert stats['avg_dataset_size'] == 500
        assert stats['total_participations'] == 0
        assert stats['total_failures'] == 0

        # Add some failures and participations
        manager.record_failure(0)
        from collections import OrderedDict
        result = TrainingResult(
            client_id=1,
            weights=OrderedDict(),
            num_samples=400,
            train_loss=0.5,
            train_accuracy=0.85,
            val_loss=None,
            val_accuracy=None,
            training_time=10.0,
        )
        manager.update_from_result(result, round_num=0)

        stats = manager.get_summary_stats()
        assert stats['total_failures'] == 1
        assert stats['total_participations'] == 1

    def test_repr(self):
        """Test string representation."""
        manager = ClientManager()
        manager.register_client(client_id=0, dataset_size=600)
        manager.register_client(client_id=1, dataset_size=500, is_available=False)

        repr_str = repr(manager)
        assert 'ClientManager' in repr_str
        assert 'total_clients=2' in repr_str
        assert 'available=1' in repr_str

    def test_record_selection_updates_history_and_skips(self):
        """Test selection tracking for fairness-aware strategies."""
        manager = ClientManager()
        manager.register_client(client_id=0, dataset_size=600)
        manager.register_client(client_id=1, dataset_size=550)

        manager.record_selection(
            round_num=0,
            selected_client_ids=[0],
            available_clients=[0, 1],
            scores={0: 0.8, 1: 0.2},
        )

        m0 = manager.get_client_metadata(0)
        m1 = manager.get_client_metadata(1)

        assert m0.selection_count == 1
        assert m0.selection_history == [0]
        assert m0.skipped_rounds == 0
        assert m0.last_score == 0.8

        assert m1.selection_count == 0
        assert m1.selection_history == []
        assert m1.skipped_rounds == 1
        assert m1.last_score == 0.2

    def test_average_training_time_updates_from_results(self):
        """Test running average training time update."""
        from collections import OrderedDict

        manager = ClientManager()
        manager.register_client(client_id=0, dataset_size=600)

        result1 = TrainingResult(
            client_id=0,
            weights=OrderedDict(),
            num_samples=600,
            train_loss=0.5,
            train_accuracy=0.7,
            val_loss=None,
            val_accuracy=None,
            training_time=8.0,
        )
        result2 = TrainingResult(
            client_id=0,
            weights=OrderedDict(),
            num_samples=600,
            train_loss=0.4,
            train_accuracy=0.8,
            val_loss=None,
            val_accuracy=None,
            training_time=12.0,
        )

        manager.update_from_result(result1, round_num=0)
        manager.update_from_result(result2, round_num=1)

        metadata = manager.get_client_metadata(0)
        assert metadata.average_training_time == 10.0
        assert metadata.last_performance == 0.8
