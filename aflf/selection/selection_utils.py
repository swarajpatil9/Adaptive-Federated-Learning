"""Utilities for dynamic client selection policies and sampling."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass
class SelectionPolicy:
    """Configuration for dynamic client selection behavior."""

    name: str = "hybrid"
    mode: str = "hybrid"  # top_k | probabilistic | hybrid
    exploration_rate: float = 0.2
    min_exploration_clients: int = 1
    fairness_boost_weight: float = 0.15
    max_skip_rounds: int = 5
    enforce_max_skip: bool = True
    probabilistic_temperature: float = 1.0

    def validate(self) -> None:
        if self.mode not in {"top_k", "probabilistic", "hybrid"}:
            raise ValueError(
                f"Unsupported selection mode '{self.mode}'. "
                "Expected one of: top_k, probabilistic, hybrid"
            )
        if not 0.0 <= self.exploration_rate <= 1.0:
            raise ValueError("exploration_rate must be in [0.0, 1.0]")
        if self.min_exploration_clients < 0:
            raise ValueError("min_exploration_clients must be >= 0")
        if self.max_skip_rounds < 0:
            raise ValueError("max_skip_rounds must be >= 0")
        if self.probabilistic_temperature <= 0:
            raise ValueError("probabilistic_temperature must be > 0")


class SelectionPolicyManager:
    """Provides policy resolution for each training round."""

    def __init__(self, default_policy: Optional[SelectionPolicy] = None):
        self.default_policy = default_policy or SelectionPolicy()
        self.default_policy.validate()
        self._scheduled_policies: Dict[int, SelectionPolicy] = {}

    def add_policy_for_round(self, round_num: int, policy: SelectionPolicy) -> None:
        policy.validate()
        self._scheduled_policies[round_num] = policy

    def get_policy(self, round_num: int) -> SelectionPolicy:
        return self._scheduled_policies.get(round_num, self.default_policy)

    @classmethod
    def from_config(cls, config: Optional[Dict]) -> "SelectionPolicyManager":
        if not config:
            return cls(default_policy=SelectionPolicy())

        policy = SelectionPolicy(
            name=str(config.get("name", config.get("mode", "hybrid"))),
            mode=str(config.get("mode", "hybrid")),
            exploration_rate=float(config.get("exploration_rate", 0.2)),
            min_exploration_clients=int(config.get("min_exploration_clients", 1)),
            fairness_boost_weight=float(config.get("fairness_boost_weight", 0.15)),
            max_skip_rounds=int(config.get("max_skip_rounds", 5)),
            enforce_max_skip=bool(config.get("enforce_max_skip", True)),
            probabilistic_temperature=float(config.get("probabilistic_temperature", 1.0)),
        )
        policy.validate()
        return cls(default_policy=policy)


def normalize_dict(values: Dict[int, float], default: float = 0.0) -> Dict[int, float]:
    """Min-max normalize dictionary values into [0, 1]."""
    if not values:
        return {}

    min_value = min(values.values())
    max_value = max(values.values())
    if math.isclose(max_value, min_value):
        return {key: default for key in values}

    denom = max_value - min_value
    return {key: (value - min_value) / denom for key, value in values.items()}


def weighted_sample_without_replacement(
    rng: random.Random,
    items: Sequence[int],
    weights: Sequence[float],
    k: int,
) -> List[int]:
    """Sample k unique items with replacement-free weighted probability."""
    if k <= 0 or not items:
        return []

    k = min(k, len(items))
    remaining_items = list(items)
    remaining_weights = [max(w, 0.0) for w in weights]
    selected: List[int] = []

    for _ in range(k):
        total_weight = sum(remaining_weights)
        if total_weight <= 0:
            choice_index = rng.randrange(len(remaining_items))
        else:
            threshold = rng.random() * total_weight
            cumulative = 0.0
            choice_index = 0
            for idx, weight in enumerate(remaining_weights):
                cumulative += weight
                if cumulative >= threshold:
                    choice_index = idx
                    break

        selected.append(remaining_items.pop(choice_index))
        remaining_weights.pop(choice_index)

    return selected


def softmax_probabilities(
    scores: Iterable[float],
    temperature: float,
) -> List[float]:
    """Convert scores into numerically stable softmax probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    score_list = list(scores)
    if not score_list:
        return []

    max_score = max(score_list)
    exp_scores = [math.exp((s - max_score) / temperature) for s in score_list]
    total = sum(exp_scores)
    if total <= 0:
        return [1.0 / len(score_list)] * len(score_list)
    return [s / total for s in exp_scores]


def clamp(value: float, low: float, high: float) -> float:
    """Clamp value into [low, high]."""
    return max(low, min(value, high))
