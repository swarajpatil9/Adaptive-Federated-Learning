"""
Round execution pipeline for federated training.

This module wires server orchestration, aggregation, and global evaluation
for one complete federated round.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict

from ..aggregation.aggregation_base import AggregationStrategy
from ..client.client import FederatedClient
from ..evaluation.evaluator import GlobalEvaluator
from ..server.server import FederatedServer


@dataclass
class RoundExecutionOutput:
    """Output bundle for one federated round."""

    round_num: int
    server_round: Dict[str, Any]
    evaluation: Dict[str, float]
    total_duration: float


class RoundExecutor:
    """
    Execute one federated round end-to-end.

    Responsibilities:
    - invoke server orchestration for client selection and local training
    - aggregate client updates with configured aggregation strategy
    - update global model on server
    - evaluate global model on centralized test loader
    """

    def __init__(
        self,
        server: FederatedServer,
        aggregation_strategy: AggregationStrategy,
        evaluator: GlobalEvaluator,
    ):
        self.server = server
        self.aggregation_strategy = aggregation_strategy
        self.evaluator = evaluator

    def execute(
        self,
        round_num: int,
        clients: Dict[int, FederatedClient],
        client_train_config: Dict[str, Any],
        run_evaluation: bool,
    ) -> RoundExecutionOutput:
        """Run one complete round with optional evaluation."""
        start_time = time.time()

        server_round = self.server.execute_round(
            round_num=round_num,
            clients=clients,
            client_train_config=client_train_config,
        )

        results = server_round['results']

        if results:
            current_global_weights = self.server.get_global_parameters()
            new_global_weights = self.aggregation_strategy.aggregate(
                results=results,
                global_weights=current_global_weights,
            )
            self.server.set_global_parameters(new_global_weights)

        evaluation_metrics: Dict[str, float] = {}
        if run_evaluation:
            evaluation_metrics = self.evaluator.evaluate(self.server.get_global_model())

        total_duration = time.time() - start_time

        return RoundExecutionOutput(
            round_num=round_num,
            server_round=server_round,
            evaluation=evaluation_metrics,
            total_duration=total_duration,
        )
