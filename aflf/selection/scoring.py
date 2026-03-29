"""Client scoring primitives for dynamic federated client selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

from .selection_utils import clamp, normalize_dict

if TYPE_CHECKING:
    from ..server.client_manager import ClientMetadata


@dataclass
class ScoringWeights:
    """Configurable linear weights used by ClientScorer."""

    accuracy: float = 0.35
    data_size: float = 0.2
    reliability: float = 0.2
    latency: float = 0.15
    resource: float = 0.05
    participation_boost: float = 0.05


@dataclass
class ClientScoreBreakdown:
    """Per-client score decomposition for ranking and logging."""

    client_id: int
    accuracy_component: float
    data_size_component: float
    reliability_component: float
    latency_component: float
    resource_component: float
    participation_component: float
    base_score: float


class ClientScorer:
    """Computes weighted selection scores from client metadata."""

    def __init__(self, weights: Optional[ScoringWeights] = None):
        self.weights = weights or ScoringWeights()

    @classmethod
    def from_config(cls, config: Optional[Dict]) -> "ClientScorer":
        if not config:
            return cls()
        return cls(
            weights=ScoringWeights(
                accuracy=float(config.get("accuracy", 0.35)),
                data_size=float(config.get("data_size", 0.2)),
                reliability=float(config.get("reliability", 0.2)),
                latency=float(config.get("latency", 0.15)),
                resource=float(config.get("resource", 0.05)),
                participation_boost=float(config.get("participation_boost", 0.05)),
            )
        )

    def score_clients(
        self,
        available_clients: list[int],
        client_metadata: Dict[int, "ClientMetadata"],
    ) -> Dict[int, ClientScoreBreakdown]:
        """Compute score breakdowns for currently available clients."""
        if not available_clients:
            return {}

        accuracy_raw = {
            cid: self._safe_accuracy(client_metadata[cid].last_accuracy)
            for cid in available_clients
        }
        data_size_raw = {
            cid: float(client_metadata[cid].dataset_size)
            for cid in available_clients
        }
        reliability_raw = {
            cid: self._reliability_score(client_metadata[cid])
            for cid in available_clients
        }
        latency_raw = {
            cid: max(0.0, float(client_metadata[cid].average_training_time))
            for cid in available_clients
        }
        resource_raw = {
            cid: clamp(float(getattr(client_metadata[cid], "resource_score", 1.0)), 0.0, 1.0)
            for cid in available_clients
        }
        participation_raw = {
            cid: float(client_metadata[cid].selection_count)
            for cid in available_clients
        }

        accuracy = normalize_dict(accuracy_raw, default=0.5)
        data_size = normalize_dict(data_size_raw, default=0.5)
        reliability = normalize_dict(reliability_raw, default=0.5)
        latency = normalize_dict(latency_raw, default=0.5)
        resource = normalize_dict(resource_raw, default=0.5)
        participation = normalize_dict(participation_raw, default=0.0)

        scores: Dict[int, ClientScoreBreakdown] = {}
        w = self.weights

        for cid in available_clients:
            accuracy_component = w.accuracy * accuracy[cid]
            data_component = w.data_size * data_size[cid]
            reliability_component = w.reliability * reliability[cid]
            latency_component = w.latency * latency[cid]
            resource_component = w.resource * resource[cid]
            participation_component = w.participation_boost * (1.0 - participation[cid])

            base_score = (
                accuracy_component
                + data_component
                + reliability_component
                - latency_component
                + resource_component
                + participation_component
            )

            scores[cid] = ClientScoreBreakdown(
                client_id=cid,
                accuracy_component=accuracy_component,
                data_size_component=data_component,
                reliability_component=reliability_component,
                latency_component=latency_component,
                resource_component=resource_component,
                participation_component=participation_component,
                base_score=base_score,
            )

        return scores

    @staticmethod
    def _safe_accuracy(value: Optional[float]) -> float:
        if value is None:
            return 0.5
        return clamp(float(value), 0.0, 1.0)

    @staticmethod
    def _reliability_score(metadata: "ClientMetadata") -> float:
        successes = float(metadata.participation_count)
        failures = float(metadata.failure_count)
        total = successes + failures
        if total <= 0:
            return 0.5
        return successes / total
