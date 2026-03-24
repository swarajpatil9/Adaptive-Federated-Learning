"""
Example: Using the Federated Server (Phase 5 - Orchestration Only)

This example demonstrates:
1. Server initialization
2. Client registration
3. Round execution with client selection
4. Metrics tracking
5. Failure handling

NOTE: Aggregation is not implemented yet (Phase 6).
This example shows orchestration capabilities only.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from aflf.client import FederatedClient, SimulatedFederatedClient, ClientSimulationConfig
from aflf.selection import RandomSelection, DataAwareSelection, FairnessSelection
from aflf.server import FederatedServer


# ----- 1. Define Model -----


class SimpleModel(nn.Module):
    """Simple model for demonstration."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# ----- 2. Create Dummy Clients -----


def create_dummy_clients(num_clients=10, samples_per_client=100):
    """Create dummy clients with synthetic data."""
    clients = {}

    for i in range(num_clients):
        # Create synthetic data
        X = torch.randn(samples_per_client, 10)
        y = torch.randint(0, 2, (samples_per_client,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Create client (with optional simulation features)
        if i % 3 == 0:  # Every 3rd client is simulated with 10% failure rate
            sim_config = ClientSimulationConfig(
                failure_rate=0.1, training_speed=0.8  # Slower client
            )
            client = SimulatedFederatedClient(
                client_id=i,
                train_loader=loader,
                val_loader=None,
                epochs=1,
                device="cpu",
                simulation_config=sim_config,
            )
        else:
            client = FederatedClient(
                client_id=i,
                train_loader=loader,
                val_loader=None,
                epochs=1,
                device="cpu",
            )

        clients[i] = client

    return clients


# ----- 3. Initialize Server -----


def main():
    print("=" * 70)
    print("PHASE 5: Server Orchestration Demonstration")
    print("=" * 70)

    # Create model
    model = SimpleModel()
    print(f"\n[1] Created model: {model.__class__.__name__}")

    # Create clients
    num_clients = 10
    clients = create_dummy_clients(num_clients=num_clients, samples_per_client=100)
    print(f"[2] Created {num_clients} clients (mix of regular and simulated)")

    # Initialize server with RandomSelection
    print("\n[3] Initializing server with RandomSelection strategy...")
    server = FederatedServer(
        model=model,
        selection_strategy=RandomSelection(seed=42),
        num_clients_per_round=5,  # Select 5 clients per round
    )

    # Register clients
    print("[4] Registering clients with server...")
    for client_id, client in clients.items():
        dataset_size = len(client.train_loader.dataset)
        server.register_client(client_id=client_id, dataset_size=dataset_size)

    print(f"    Registered {server.client_manager.get_num_clients()} clients")

    # ----- 4. Execute Rounds -----

    print("\n" + "=" * 70)
    print("EXECUTING FEDERATED ROUNDS (Orchestration Only - No Aggregation)")
    print("=" * 70)

    num_rounds = 3
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num} ---")

        # Execute round
        result = server.execute_round(round_num=round_num, clients=clients)

        # Display results
        print(f"  Selected:      {result['num_selected']} clients")
        print(f"  Participated:  {result['num_participating']} clients")
        print(f"  Failed:        {result['num_failed']} clients")
        print(
            f"  Success Rate:  {result['participation_rate']:.1%}"
        )
        print(f"  Duration:      {result['duration']:.2f}s")

        # Display metrics
        if result['metrics']:
            metrics = result['metrics']
            print(f"\n  Metrics:")
            print(f"    Avg Train Loss:     {metrics.get('avg_train_loss', 0):.4f}")
            print(
                f"    Avg Train Accuracy: {metrics.get('avg_train_accuracy', 0):.4f}"
            )
            print(
                f"    Total Samples:      {metrics.get('total_samples', 0)}"
            )

    # ----- 5. Display Summary -----

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Client summary
    client_summary = server.get_client_summary()
    print(f"\nClient Statistics:")
    print(f"  Total clients:       {client_summary['total_clients']}")
    print(f"  Available clients:   {client_summary['available_clients']}")
    print(f"  Total dataset size:  {client_summary['total_dataset_size']}")
    print(f"  Avg dataset size:    {client_summary['avg_dataset_size']:.1f}")
    print(f"  Total participations: {client_summary['total_participations']}")
    print(f"  Total failures:      {client_summary['total_failures']}")

    # Round summary
    round_summary = server.get_round_summary()
    print(f"\nRound Statistics:")
    print(f"  Total rounds:           {round_summary['total_rounds']}")
    print(
        f"  Avg participation rate: {round_summary['avg_participation_rate']:.1%}"
    )
    print(f"  Avg failure rate:       {round_summary['avg_failure_rate']:.1%}")
    print(f"  Avg duration:           {round_summary['avg_duration']:.2f}s")

    # ----- 6. Demonstrate Different Selection Strategies -----

    print("\n" + "=" * 70)
    print("TESTING DIFFERENT SELECTION STRATEGIES")
    print("=" * 70)

    # Create new server with DataAwareSelection
    print("\n[1] DataAwareSelection (selects clients with most data):")
    server_data_aware = FederatedServer(
        model=SimpleModel(),
        selection_strategy=DataAwareSelection(),
        num_clients_per_round=3,
    )

    # Register clients with varying dataset sizes
    for i in range(5):
        server_data_aware.register_client(
            client_id=i, dataset_size=100 + i * 50  # 100, 150, 200, 250, 300
        )

    result = server_data_aware.execute_round(round_num=0, clients=clients)
    print(f"    Selected {result['num_selected']} clients")
    print(f"    Participating: {result['num_participating']} clients")

    # FairnessSelection
    print("\n[2] FairnessSelection (ensures equal participation):")
    server_fairness = FederatedServer(
        model=SimpleModel(),
        selection_strategy=FairnessSelection(seed=42),
        num_clients_per_round=3,
    )

    for i in range(5):
        server_fairness.register_client(client_id=i, dataset_size=100)

    # Execute multiple rounds to see fairness
    for round_num in range(3):
        result = server_fairness.execute_round(round_num=round_num, clients=clients)
        print(
            f"    Round {round_num}: {result['num_participating']} clients participated"
        )

    print("\n" + "=" * 70)
    print("PHASE 5 COMPLETE: Server Orchestration Working")
    print("NEXT: Phase 6 will add aggregation (FedAvg, etc.)")
    print("=" * 70)


if __name__ == "__main__":
    main()
