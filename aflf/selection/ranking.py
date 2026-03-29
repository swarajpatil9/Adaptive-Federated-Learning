"""Client ranking module for dynamic selection strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

from .scoring import ClientScoreBreakdown
from .selection_utils import SelectionPolicy, normalize_dict

if TYPE_CHECKING:
    from ..server.client_manager import ClientMetadata


@dataclass
class RankedClient:
    """Ranked representation of one candidate client."""

    client_id: int
    base_score: float
    final_score: float
    forced_by_fairness: bool


@dataclass
class RankingResult:
    """Ordering and fairness decisions for a selection round."""

    ranked_clients: List[RankedClient]
    forced_clients: List[int]


class ClientRanker:
    """Ranks clients by score while enforcing participation fairness."""

    def rank(
        self,
        available_clients: List[int],
        score_breakdowns: Dict[int, ClientScoreBreakdown],
        client_metadata: Dict[int, "ClientMetadata"],
        policy: SelectionPolicy,
    ) -> RankingResult:
        if not available_clients:
            return RankingResult(ranked_clients=[], forced_clients=[])

        skipped_raw = {
            cid: float(client_metadata[cid].skipped_rounds)
            for cid in available_clients
        }
        skipped_normalized = normalize_dict(skipped_raw, default=0.0)

        forced_clients = []
        ranked_clients: List[RankedClient] = []

        for cid in available_clients:
            metadata = client_metadata[cid]
            base_score = score_breakdowns[cid].base_score
            fairness_boost = policy.fairness_boost_weight * skipped_normalized[cid]
            final_score = base_score + fairness_boost

            forced = bool(policy.enforce_max_skip and metadata.skipped_rounds >= policy.max_skip_rounds)
            if forced:
                forced_clients.append(cid)

            ranked_clients.append(
                RankedClient(
                    client_id=cid,
                    base_score=base_score,
                    final_score=final_score,
                    forced_by_fairness=forced,
                )
            )

        # Forced clients first, then by final score descending.
        ranked_clients.sort(
            key=lambda rc: (
                0 if rc.forced_by_fairness else 1,
                -rc.final_score,
            )
        )

        # Preserve ranking order in forced list too.
        forced_clients_sorted = [rc.client_id for rc in ranked_clients if rc.forced_by_fairness]

        return RankingResult(
            ranked_clients=ranked_clients,
            forced_clients=forced_clients_sorted,
        )
