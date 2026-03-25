"""
Federated training controller.

Implements the baseline end-to-end FL loop:
initialize global model -> register clients -> select clients -> local train
-> aggregate -> update global model -> evaluate -> repeat.
"""

from typing import Any, Dict, Optional

from ..aggregation.fedavg import FedAvg
from ..client import FederatedClient
from ..client.client_utils import set_reproducibility
from ..data.federated_data import FederatedDataModule
from ..evaluation.evaluator import GlobalEvaluator
from ..models.factory import create_model
from ..selection.selection_strategy import RandomSelection, SelectionStrategy
from ..server.server import FederatedServer
from .round_executor import RoundExecutor
from .training_utils import (
    FederatedTrainingConfig,
    RoundTrainingRecord,
    TrainingProgressTracker,
    format_round_log,
)


class FederatedTrainer:
    """
    Main controller for baseline federated training.

    The trainer coordinates modules instead of embedding their logic:
    - server/orchestrator for selection and client execution
    - aggregation strategy for model aggregation
    - evaluator for global model evaluation
    """

    def __init__(
        self,
        model,
        data_module: FederatedDataModule,
        config: FederatedTrainingConfig,
        selection_strategy: Optional[SelectionStrategy] = None,
    ):
        self.model = model
        self.data_module = data_module
        self.config = config

        set_reproducibility(self.config.seed)

        self.selection_strategy = selection_strategy or RandomSelection(seed=self.config.seed)
        self.aggregation_strategy = FedAvg()

        self.server = FederatedServer(
            model=self.model,
            selection_strategy=self.selection_strategy,
            aggregation_strategy=self.aggregation_strategy,
            num_clients_per_round=self.config.clients_per_round,
            device=self.config.device,
        )

        self.clients = self._build_clients()

        self.evaluator = GlobalEvaluator(
            test_loader=self.data_module.get_test_loader(),
            device=self.config.device,
        )

        self.round_executor = RoundExecutor(
            server=self.server,
            aggregation_strategy=self.aggregation_strategy,
            evaluator=self.evaluator,
        )

        self.progress = TrainingProgressTracker()

    def _build_clients(self) -> Dict[int, FederatedClient]:
        """Create client objects and register them on the server."""
        clients: Dict[int, FederatedClient] = {}

        for client_id in range(self.data_module.num_clients):
            train_loader = self.data_module.get_client_loader(
                client_id=client_id,
                batch_size=self.config.batch_size,
                shuffle=True,
            )

            client = FederatedClient(
                client_id=client_id,
                train_loader=train_loader,
                val_loader=None,
                device=self.config.device,
                verbose=self.config.verbose,
            )
            clients[client_id] = client

            self.server.register_client(
                client_id=client_id,
                dataset_size=client.num_samples,
                is_available=True,
            )

        return clients

    def fit(self) -> Dict[str, Any]:
        """Run full federated training process for configured rounds."""
        client_train_config = self.config.to_client_train_config()

        for round_num in range(self.config.num_rounds):
            should_evaluate = (
                (round_num % self.config.evaluation_frequency == 0)
                or (round_num == self.config.num_rounds - 1)
            )

            output = self.round_executor.execute(
                round_num=round_num,
                clients=self.clients,
                client_train_config=client_train_config,
                run_evaluation=should_evaluate,
            )

            server_metrics = output.server_round.get('metrics', {})
            eval_metrics = output.evaluation

            record = RoundTrainingRecord(
                round_num=round_num,
                global_loss=float(eval_metrics.get('global_loss', 0.0)),
                global_accuracy=float(eval_metrics.get('global_accuracy', 0.0)),
                avg_train_loss=float(server_metrics.get('avg_train_loss', 0.0)),
                avg_train_accuracy=float(server_metrics.get('avg_train_accuracy', 0.0)),
                selected_clients=int(output.server_round.get('num_selected', 0)),
                participating_clients=int(output.server_round.get('num_participating', 0)),
                failed_clients=int(output.server_round.get('num_failed', 0)),
                participation_rate=float(output.server_round.get('participation_rate', 0.0)),
                round_duration=float(output.total_duration),
            )

            self.progress.add_record(record)
            print(format_round_log(record))

        return {
            'config': self.config.to_dict(),
            'history': self.progress.records,
            'summary': self.progress.summary(),
        }

    @classmethod
    def from_components(
        cls,
        data_config: Dict[str, Any],
        model_config: Dict[str, Any],
        federated_config: FederatedTrainingConfig,
    ) -> 'FederatedTrainer':
        """Build trainer from config dictionaries."""
        model = create_model(
            model_name=model_config.get('name', 'simple_cnn'),
            **{k: v for k, v in model_config.items() if k != 'name'},
        )

        data_module = FederatedDataModule(
            dataset_name=data_config.get('dataset_name', 'mnist'),
            data_root=data_config.get('data_root', './data'),
            num_clients=int(data_config.get('num_clients', 100)),
            partition_strategy=data_config.get('partition_strategy', 'iid'),
            batch_size=int(federated_config.batch_size),
            test_batch_size=int(data_config.get('test_batch_size', 128)),
            num_workers=int(data_config.get('num_workers', 0)),
            seed=int(federated_config.seed),
            download=bool(data_config.get('download', True)),
            alpha=data_config.get('alpha', 0.5),
            shards_per_client=data_config.get('shards_per_client', 2),
            min_samples_per_client=data_config.get('min_samples_per_client', 10),
        )

        return cls(
            model=model,
            data_module=data_module,
            config=federated_config,
        )
