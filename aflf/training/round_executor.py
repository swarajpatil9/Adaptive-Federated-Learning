"""
Round execution pipeline for federated training.

This module wires server orchestration, aggregation, and global evaluation
for one complete federated round.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

from ..aggregation.aggregation_base import AggregationStrategy
from ..communication import CommunicationConfig, ModelCompressor
from ..client.client import FederatedClient
from ..evaluation.evaluator import EvaluationManager
from ..server.server import FederatedServer

logger = logging.getLogger("aflf.training.round_executor")


@dataclass
class RoundExecutionOutput:
    """Output bundle for one federated round."""

    round_num: int
    server_round: Dict[str, Any]
    evaluation: Dict[str, Any]
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
        evaluator: EvaluationManager,
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
        optimization_metrics: Dict[str, Any] = None,
    ) -> RoundExecutionOutput:
        """Run one complete round with optional evaluation."""
        start_time = time.time()

        try:
            server_round = self.server.execute_round(
                round_num=round_num,
                clients=clients,
                client_train_config=client_train_config,
            )
        except Exception as exc:
            logger.exception("Server round execution failed at round %s: %s", round_num, exc)
            raise RuntimeError(f"Round {round_num} server execution failed") from exc

        results = server_round['results']

        communication_config = CommunicationConfig.from_dict(
            (client_train_config or {}).get('communication', {})
        )
        compressor = ModelCompressor(communication_config)

        try:
            for result in results:
                communication_metadata = getattr(result, 'communication_metadata', {}) or {}
                result.weights = compressor.decompress_model_update(
                    result.weights,
                    communication_metadata,
                )
        except Exception as exc:
            logger.exception("Model update decompression failed in round %s: %s", round_num, exc)
            results = []

        if results:
            try:
                current_global_weights = self.server.get_global_parameters()
                new_global_weights = self.aggregation_strategy.aggregate(
                    results=results,
                    global_weights=current_global_weights,
                )
                self.server.set_global_parameters(new_global_weights)
            except Exception as exc:
                logger.exception("Aggregation failed in round %s: %s", round_num, exc)

        evaluation_metrics: Dict[str, Any] = {}
        if run_evaluation:
            try:
                round_metrics = self.evaluator.evaluate_round(
                    round_num=round_num,
                    model=self.server.get_global_model(),
                    server_round=server_round,
                    round_time=time.time() - start_time,
                    criterion_name=(client_train_config or {}).get(
                        'criterion',
                        'cross_entropy',
                    ),
                    optimization_metrics=optimization_metrics,
                )
                evaluation_metrics = self.evaluator.round_metrics_dict(round_metrics)
            except Exception as exc:
                logger.exception("Evaluation failed in round %s: %s", round_num, exc)

        total_duration = time.time() - start_time

        return RoundExecutionOutput(
            round_num=round_num,
            server_round=server_round,
            evaluation=evaluation_metrics,
            total_duration=total_duration,
        )
