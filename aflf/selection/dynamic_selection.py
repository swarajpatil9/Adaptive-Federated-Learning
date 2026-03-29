"""Dynamic client selection strategy for federated learning."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Dict, List, Optional

from .ranking import ClientRanker
from .scoring import ClientScorer
from .selection_strategy import SelectionResult, SelectionStrategy
from .selection_utils import (
    SelectionPolicyManager,
    softmax_probabilities,
    weighted_sample_without_replacement,
)

if TYPE_CHECKING:
    from ..server.client_manager import ClientMetadata


class DynamicSelectionStrategy(SelectionStrategy):
    """
    Score-driven client selection with fairness and exploration.

    Supports:
    - top_k: deterministic best-ranked clients
    - probabilistic: weighted stochastic selection
    - hybrid: top-k exploitation + probabilistic exploration
    """

    def __init__(
        self,
        scorer: Optional[ClientScorer] = None,
        ranker: Optional[ClientRanker] = None,
        policy_manager: Optional[SelectionPolicyManager] = None,
        seed: Optional[int] = None,
    ):
        self.scorer = scorer or ClientScorer()
        self.ranker = ranker or ClientRanker()
        self.policy_manager = policy_manager or SelectionPolicyManager()
        self.seed = seed
        self._rng = random.Random(seed)

    @classmethod
    def from_config(cls, config: Optional[Dict], seed: Optional[int] = None) -> "DynamicSelectionStrategy":
        config = config or {}
        scorer = ClientScorer.from_config(config.get("weights"))
        policy_manager = SelectionPolicyManager.from_config(config.get("policy"))
        return cls(
            scorer=scorer,
            ranker=ClientRanker(),
            policy_manager=policy_manager,
            seed=seed,
        )

    def select(
        self,
        available_clients: List[int],
        num_clients: int,
        round_num: int,
        client_metadata: Optional[Dict[int, "ClientMetadata"]] = None,
    ) -> List[int]:
        decision = self.select_with_details(
            available_clients=available_clients,
            num_clients=num_clients,
            round_num=round_num,
            client_metadata=client_metadata,
        )
        return decision.selected_client_ids

    def select_with_details(
        self,
        available_clients: List[int],
        num_clients: int,
        round_num: int,
        client_metadata: Optional[Dict[int, "ClientMetadata"]] = None,
    ) -> SelectionResult:
        if num_clients > len(available_clients):
            raise ValueError(
                f"Cannot select {num_clients} clients from "
                f"{len(available_clients)} available clients"
            )
        if client_metadata is None:
            raise ValueError("client_metadata required for DynamicSelectionStrategy")

        if num_clients == 0:
            return SelectionResult(
                selected_client_ids=[],
                client_scores={},
                selection_reasoning={},
                policy_name=self.policy_manager.get_policy(round_num).name,
            )

        policy = self.policy_manager.get_policy(round_num)
        score_breakdowns = self.scorer.score_clients(available_clients, client_metadata)
        ranking = self.ranker.rank(
            available_clients=available_clients,
            score_breakdowns=score_breakdowns,
            client_metadata=client_metadata,
            policy=policy,
        )

        scores = {cid: breakdown.base_score for cid, breakdown in score_breakdowns.items()}
        selected: List[int] = []
        reasons: Dict[int, str] = {}

        # Fairness guarantee: clients skipped too long are selected first.
        for forced_cid in ranking.forced_clients:
            if len(selected) >= num_clients:
                break
            selected.append(forced_cid)
            reasons[forced_cid] = (
                f"Fairness override: skipped {client_metadata[forced_cid].skipped_rounds} rounds"
            )

        remaining_slots = num_clients - len(selected)
        if remaining_slots > 0:
            remaining_ranked = [
                rc for rc in ranking.ranked_clients if rc.client_id not in selected
            ]

            if policy.mode == "top_k":
                picked = [rc.client_id for rc in remaining_ranked[:remaining_slots]]
                selected.extend(picked)
                for cid in picked:
                    reasons[cid] = reasons.get(cid, "Top-K by dynamic score")

            elif policy.mode == "probabilistic":
                population = [rc.client_id for rc in remaining_ranked]
                probs = softmax_probabilities(
                    [rc.final_score for rc in remaining_ranked],
                    temperature=policy.probabilistic_temperature,
                )
                picked = weighted_sample_without_replacement(
                    rng=self._rng,
                    items=population,
                    weights=probs,
                    k=remaining_slots,
                )
                selected.extend(picked)
                for cid in picked:
                    reasons[cid] = reasons.get(cid, "Probabilistic weighted exploration")

            else:  # hybrid
                exploration_count = int(round(remaining_slots * policy.exploration_rate))
                exploration_count = min(exploration_count, remaining_slots)
                if remaining_slots > 0 and policy.min_exploration_clients > 0:
                    exploration_count = max(
                        exploration_count,
                        min(policy.min_exploration_clients, remaining_slots),
                    )
                exploitation_count = max(0, remaining_slots - exploration_count)

                exploit = [rc.client_id for rc in remaining_ranked[:exploitation_count]]
                selected.extend(exploit)
                for cid in exploit:
                    reasons[cid] = reasons.get(cid, "Hybrid exploitation (top score)")

                remaining_after_exploit = [
                    rc for rc in remaining_ranked if rc.client_id not in selected
                ]
                population = [rc.client_id for rc in remaining_after_exploit]
                probs = softmax_probabilities(
                    [rc.final_score for rc in remaining_after_exploit],
                    temperature=policy.probabilistic_temperature,
                )
                explore = weighted_sample_without_replacement(
                    rng=self._rng,
                    items=population,
                    weights=probs,
                    k=remaining_slots - len(exploit),
                )
                selected.extend(explore)
                for cid in explore:
                    reasons[cid] = reasons.get(cid, "Hybrid exploration (probabilistic)")

        return SelectionResult(
            selected_client_ids=selected[:num_clients],
            client_scores=scores,
            selection_reasoning={cid: reasons.get(cid, "Selected") for cid in selected[:num_clients]},
            policy_name=policy.name,
        )

    def __repr__(self) -> str:
        return f"DynamicSelectionStrategy(seed={self.seed})"
