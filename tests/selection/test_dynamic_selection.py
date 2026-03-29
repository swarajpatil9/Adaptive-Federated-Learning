"""Tests for dynamic client selection strategy."""

from aflf.selection.dynamic_selection import DynamicSelectionStrategy
from aflf.selection.scoring import ClientScorer, ScoringWeights
from aflf.selection.selection_utils import SelectionPolicy, SelectionPolicyManager
from aflf.server.client_manager import ClientMetadata


def _metadata(
    client_id: int,
    dataset_size: int = 100,
    accuracy: float = 0.5,
    average_training_time: float = 1.0,
    participation_count: int = 1,
    selection_count: int = 1,
    failure_count: int = 0,
    skipped_rounds: int = 0,
    resource_score: float = 1.0,
):
    return ClientMetadata(
        client_id=client_id,
        dataset_size=dataset_size,
        last_accuracy=accuracy,
        average_training_time=average_training_time,
        participation_count=participation_count,
        selection_count=selection_count,
        failure_count=failure_count,
        skipped_rounds=skipped_rounds,
        resource_score=resource_score,
    )


class TestDynamicSelectionStrategy:
    """Validate scoring-based and fairness-aware selection."""

    def test_top_k_selects_highest_scored_clients(self):
        policy_manager = SelectionPolicyManager(
            default_policy=SelectionPolicy(name="top_k", mode="top_k", enforce_max_skip=False)
        )
        scorer = ClientScorer(
            weights=ScoringWeights(
                accuracy=1.0,
                data_size=0.0,
                reliability=0.0,
                latency=0.0,
                resource=0.0,
                participation_boost=0.0,
            )
        )
        strategy = DynamicSelectionStrategy(
            scorer=scorer,
            policy_manager=policy_manager,
            seed=42,
        )

        available = [0, 1, 2]
        metadata = {
            0: _metadata(0, accuracy=0.95),
            1: _metadata(1, accuracy=0.80),
            2: _metadata(2, accuracy=0.20),
        }

        selected = strategy.select(
            available_clients=available,
            num_clients=2,
            round_num=0,
            client_metadata=metadata,
        )

        assert selected == [0, 1]

    def test_fairness_override_selects_skipped_client(self):
        policy_manager = SelectionPolicyManager(
            default_policy=SelectionPolicy(
                name="top_k_fair",
                mode="top_k",
                max_skip_rounds=2,
                enforce_max_skip=True,
            )
        )
        scorer = ClientScorer(
            weights=ScoringWeights(
                accuracy=1.0,
                data_size=0.0,
                reliability=0.0,
                latency=0.0,
                resource=0.0,
                participation_boost=0.0,
            )
        )
        strategy = DynamicSelectionStrategy(
            scorer=scorer,
            policy_manager=policy_manager,
            seed=7,
        )

        available = [0, 1]
        metadata = {
            0: _metadata(0, accuracy=0.95, skipped_rounds=0),
            1: _metadata(1, accuracy=0.10, skipped_rounds=2),
        }

        decision = strategy.select_with_details(
            available_clients=available,
            num_clients=1,
            round_num=3,
            client_metadata=metadata,
        )

        assert decision.selected_client_ids == [1]
        assert "Fairness override" in decision.selection_reasoning[1]
