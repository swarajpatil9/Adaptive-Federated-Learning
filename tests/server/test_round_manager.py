"""
Tests for RoundManager.
"""

import pytest

from aflf.server.round_manager import RoundManager, RoundState


class TestRoundState:
    """Tests for RoundState dataclass."""

    def test_initialization(self):
        """Test RoundState initialization."""
        state = RoundState(round_num=0, selected_clients=[0, 1, 2])

        assert state.round_num == 0
        assert state.selected_clients == [0, 1, 2]
        assert state.participating_clients == []
        assert state.dropped_clients == []
        assert state.metrics == {}
        assert state.start_time > 0
        assert state.end_time is None
        assert state.duration is None

    def test_participation_rate(self):
        """Test participation rate calculation."""
        state = RoundState(round_num=0, selected_clients=[0, 1, 2, 3, 4])

        # No participants
        assert state.participation_rate == 0.0

        # Some participants
        state.participating_clients = [0, 1, 2]
        assert state.participation_rate == 0.6  # 3/5

        # All participants
        state.participating_clients = [0, 1, 2, 3, 4]
        assert state.participation_rate == 1.0

    def test_failure_rate(self):
        """Test failure rate calculation."""
        state = RoundState(round_num=0, selected_clients=[0, 1, 2, 3, 4])

        # No failures
        assert state.failure_rate == 0.0

        # Some failures
        state.dropped_clients = [3, 4]
        assert state.failure_rate == 0.4  # 2/5

        # All failures
        state.dropped_clients = [0, 1, 2, 3, 4]
        assert state.failure_rate == 1.0

    def test_finalize(self):
        """Test round finalization."""
        state = RoundState(round_num=0, selected_clients=[0, 1])

        assert state.end_time is None
        assert state.duration is None

        state.finalize()

        assert state.end_time is not None
        assert state.duration is not None
        assert state.duration > 0

    def test_to_dict(self):
        """Test converting to dictionary."""
        state = RoundState(round_num=0, selected_clients=[0, 1, 2])
        state.participating_clients = [0, 1]
        state.dropped_clients = [2]
        state.metrics = {'avg_loss': 0.5}
        state.finalize()

        result = state.to_dict()

        assert result['round_num'] == 0
        assert result['num_selected'] == 3
        assert result['num_participating'] == 2
        assert result['num_dropped'] == 1
        assert result['participation_rate'] == pytest.approx(2 / 3)
        assert result['failure_rate'] == pytest.approx(1 / 3)
        assert result['duration'] > 0
        assert result['metrics'] == {'avg_loss': 0.5}

    def test_repr(self):
        """Test string representation."""
        state = RoundState(round_num=5, selected_clients=[0, 1, 2])
        state.participating_clients = [0, 1]
        state.dropped_clients = [2]

        repr_str = repr(state)
        assert 'RoundState' in repr_str
        assert 'round=5' in repr_str
        assert 'selected=3' in repr_str
        assert 'participating=2' in repr_str
        assert 'dropped=1' in repr_str


class TestRoundManager:
    """Tests for RoundManager class."""

    def test_initialization(self):
        """Test RoundManager initialization."""
        manager = RoundManager()

        assert manager.get_num_rounds() == 0
        assert manager.get_round_history() == []
        assert manager.get_current_round() is None

    def test_start_round(self):
        """Test starting a round."""
        manager = RoundManager()

        state = manager.start_round(round_num=0, selected_clients=[0, 1, 2])

        assert state.round_num == 0
        assert state.selected_clients == [0, 1, 2]
        assert manager.get_current_round() == state

    def test_start_round_while_in_progress_raises_error(self):
        """Test that starting a round while one is in progress raises error."""
        manager = RoundManager()
        manager.start_round(round_num=0, selected_clients=[0, 1])

        with pytest.raises(RuntimeError, match="still in progress"):
            manager.start_round(round_num=1, selected_clients=[2, 3])

    def test_record_participation(self):
        """Test recording client participation."""
        manager = RoundManager()
        state = manager.start_round(round_num=0, selected_clients=[0, 1, 2])

        manager.record_participation(client_id=0)
        manager.record_participation(client_id=1)

        assert state.participating_clients == [0, 1]

    def test_record_participation_no_round_raises_error(self):
        """Test that recording participation without active round raises error."""
        manager = RoundManager()

        with pytest.raises(RuntimeError, match="No round in progress"):
            manager.record_participation(client_id=0)

    def test_record_drop(self):
        """Test recording client dropout."""
        manager = RoundManager()
        state = manager.start_round(round_num=0, selected_clients=[0, 1, 2])

        manager.record_drop(client_id=2, reason="timeout")

        assert state.dropped_clients == [2]

    def test_record_drop_no_round_raises_error(self):
        """Test that recording drop without active round raises error."""
        manager = RoundManager()

        with pytest.raises(RuntimeError, match="No round in progress"):
            manager.record_drop(client_id=0, reason="timeout")

    def test_end_round(self):
        """Test ending a round."""
        manager = RoundManager()
        state = manager.start_round(round_num=0, selected_clients=[0, 1, 2])

        manager.record_participation(client_id=0)
        manager.record_participation(client_id=1)
        manager.record_drop(client_id=2, reason="failure")

        metrics = {'avg_loss': 0.5, 'avg_accuracy': 0.85}
        completed_state = manager.end_round(metrics=metrics)

        assert completed_state == state
        assert state.metrics == metrics
        assert state.end_time is not None
        assert state.duration is not None
        assert manager.get_current_round() is None
        assert manager.get_num_rounds() == 1
        assert manager.get_round_history() == [state]

    def test_end_round_no_round_raises_error(self):
        """Test that ending round without active round raises error."""
        manager = RoundManager()

        with pytest.raises(RuntimeError, match="No round in progress"):
            manager.end_round(metrics={})

    def test_multiple_rounds(self):
        """Test executing multiple rounds."""
        manager = RoundManager()

        # Round 0
        manager.start_round(round_num=0, selected_clients=[0, 1])
        manager.record_participation(client_id=0)
        manager.end_round(metrics={'loss': 0.5})

        # Round 1
        manager.start_round(round_num=1, selected_clients=[2, 3])
        manager.record_participation(client_id=2)
        manager.record_participation(client_id=3)
        manager.end_round(metrics={'loss': 0.4})

        # Round 2
        manager.start_round(round_num=2, selected_clients=[0, 1, 2])
        manager.record_participation(client_id=0)
        manager.end_round(metrics={'loss': 0.3})

        assert manager.get_num_rounds() == 3
        history = manager.get_round_history()
        assert len(history) == 3
        assert history[0].round_num == 0
        assert history[1].round_num == 1
        assert history[2].round_num == 2

    def test_get_round(self):
        """Test getting specific round from history."""
        manager = RoundManager()

        manager.start_round(round_num=0, selected_clients=[0, 1])
        manager.end_round(metrics={})

        manager.start_round(round_num=1, selected_clients=[2, 3])
        manager.end_round(metrics={})

        round_0 = manager.get_round(round_num=0)
        assert round_0 is not None
        assert round_0.round_num == 0

        round_1 = manager.get_round(round_num=1)
        assert round_1 is not None
        assert round_1.round_num == 1

        round_999 = manager.get_round(round_num=999)
        assert round_999 is None

    def test_summary_stats(self):
        """Test getting summary statistics."""
        manager = RoundManager()

        # Empty manager
        stats = manager.get_summary_stats()
        assert stats['total_rounds'] == 0
        assert stats['avg_participation_rate'] == 0.0
        assert stats['avg_failure_rate'] == 0.0
        assert stats['avg_duration'] == 0.0

        # Round 0: 2/3 participated
        manager.start_round(round_num=0, selected_clients=[0, 1, 2])
        manager.record_participation(0)
        manager.record_participation(1)
        manager.record_drop(2, "timeout")
        manager.end_round(metrics={})

        # Round 1: 3/3 participated
        manager.start_round(round_num=1, selected_clients=[0, 1, 2])
        manager.record_participation(0)
        manager.record_participation(1)
        manager.record_participation(2)
        manager.end_round(metrics={})

        stats = manager.get_summary_stats()
        assert stats['total_rounds'] == 2
        assert stats['avg_participation_rate'] == pytest.approx((2 / 3 + 1.0) / 2)
        assert stats['avg_failure_rate'] == pytest.approx((1 / 3 + 0.0) / 2)
        assert stats['avg_duration'] > 0

    def test_repr(self):
        """Test string representation."""
        manager = RoundManager()
        manager.start_round(round_num=0, selected_clients=[0, 1])
        manager.end_round(metrics={})
        manager.start_round(round_num=1, selected_clients=[2, 3])
        manager.end_round(metrics={})

        repr_str = repr(manager)
        assert 'RoundManager' in repr_str
        assert 'completed_rounds=2' in repr_str
