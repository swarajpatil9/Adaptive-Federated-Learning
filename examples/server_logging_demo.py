"""
Example: Server Logging System for Federated Learning

Demonstrates comprehensive logging features:
1. Structured console/file logging
2. Progress bar tracking
3. Metrics export (JSON/CSV)
4. TensorBoard integration (optional)
5. Convergence tracking

This follows patterns from research repositories like Flower, FedML, and PySyft.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from aflf.client import FederatedClient
from aflf.selection import RandomSelection
from aflf.server import (
    FederatedServer,
    MetricsTracker,
    ProgressTracker,
    ServerLogger,
)


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


# ----- 2. Create Clients -----


def create_clients(num_clients=10, samples_per_client=100):
    """Create dummy clients with synthetic data."""
    clients = {}

    for i in range(num_clients):
        X = torch.randn(samples_per_client, 10)
        y = torch.randint(0, 2, (samples_per_client,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        client = FederatedClient(
            client_id=i,
            train_loader=loader,
            val_loader=None,
            epochs=1,
            device="cpu",
        )
        clients[i] = client

    return clients


# ----- 3. Example 1: Basic Logging -----


def example_basic_logging():
    """Example with basic structured logging to console and file."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Structured Logging")
    print("=" * 70)

    # Initialize logger
    logger = ServerLogger(
        experiment_name="basic_logging_demo",
        log_dir="logs",
        console_level=20,  # INFO
        enable_file_logging=True,
    )

    # Create server with logger
    server = FederatedServer(
        model=SimpleModel(),
        selection_strategy=RandomSelection(seed=42),
        num_clients_per_round=5,
        server_logger=logger,
    )

    # Create and register clients
    clients = create_clients(num_clients=10)
    for client_id, client in clients.items():
        server.register_client(client_id=client_id, dataset_size=100)

    # Run training
    print("\nRunning 3 rounds with structured logging...\n")
    for round_num in range(3):
        server.execute_round(round_num=round_num, clients=clients)

    print("\nCheck logs/ directory for detailed log file!")


# ----- 4. Example 2: Progress Bar -----


def example_progress_bar():
    """Example with console progress bar."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Progress Bar Tracking")
    print("=" * 70)

    # Create server (no logger for cleaner output)
    server = FederatedServer(
        model=SimpleModel(),
        selection_strategy=RandomSelection(seed=42),
        num_clients_per_round=5,
    )

    # Create and register clients
    clients = create_clients(num_clients=10)
    for client_id, client in clients.items():
        server.register_client(client_id=client_id, dataset_size=100)

    # Run with progress bar using run_training()
    print("\nRunning 10 rounds with progress bar...\n")
    summary = server.run_training(num_rounds=10, clients=clients, show_progress=True)

    print(f"\nCompleted in {summary['total_time']:.2f}s")


# ----- 5. Example 3: Metrics Export -----


def example_metrics_export():
    """Example with metrics tracking and export to JSON/CSV."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Metrics Tracking and Export")
    print("=" * 70)

    # Initialize metrics tracker
    tracker = MetricsTracker(
        experiment_name="metrics_demo",
        output_dir="results",
        enable_tensorboard=False,  # Set to True if you have tensorboard installed
    )

    # Create server with tracker
    server = FederatedServer(
        model=SimpleModel(),
        selection_strategy=RandomSelection(seed=42),
        num_clients_per_round=5,
        metrics_tracker=tracker,
    )

    # Create and register clients
    clients = create_clients(num_clients=10)
    for client_id, client in clients.items():
        server.register_client(client_id=client_id, dataset_size=100)

    # Run training
    print("\nRunning 5 rounds with metrics tracking...\n")
    for round_num in range(5):
        result = server.execute_round(round_num=round_num, clients=clients)
        print(
            f"Round {round_num}: "
            f"{result['num_participating']}/{result['num_selected']} clients, "
            f"Loss: {result['metrics'].get('avg_train_loss', 0):.4f}"
        )

    # Compute summary
    print("\n--- Summary Statistics ---")
    summary = tracker.compute_summary()
    print(f"Total rounds:       {summary['num_rounds']}")
    print(f"Total time:         {summary['total_time_seconds']:.2f}s")
    print(f"Avg round time:     {summary['avg_round_time']:.2f}s")
    print(f"Final train loss:   {summary['final_train_loss']:.4f}")
    print(f"Final train acc:    {summary['final_train_accuracy']:.4f}")

    # Export metrics
    print("\n--- Exporting Metrics ---")
    paths = server.export_metrics(export_json=True, export_csv=True)
    print(f"JSON:       {paths.get('json', 'N/A')}")
    print(f"CSV:        {paths.get('csv', 'N/A')}")
    print(f"Client CSV: {paths.get('client_csv', 'N/A')}")

    # Close tracker
    tracker.close()


# ----- 6. Example 4: Complete Logging System -----


def example_complete_logging():
    """Example with all logging features enabled."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Complete Logging System")
    print("=" * 70)

    # Initialize logger and tracker
    logger = ServerLogger(
        experiment_name="complete_demo",
        log_dir="logs",
        console_level=20,  # INFO
    )

    tracker = MetricsTracker(
        experiment_name="complete_demo",
        output_dir="results",
        enable_tensorboard=False,
    )

    # Create server with full logging
    server = FederatedServer(
        model=SimpleModel(),
        selection_strategy=RandomSelection(seed=42),
        num_clients_per_round=5,
        server_logger=logger,
        metrics_tracker=tracker,
    )

    # Create and register clients
    clients = create_clients(num_clients=10)
    for client_id, client in clients.items():
        server.register_client(client_id=client_id, dataset_size=100)

    # Run training with all features
    print("\nRunning 10 rounds with full logging...\n")
    summary = server.run_training(num_rounds=10, clients=clients, show_progress=True)

    # Export everything
    print("\n--- Exporting Results ---")
    paths = server.export_metrics()
    print(f"Exported to: {paths}")

    # Cleanup
    tracker.close()


# ----- 7. Example 5: Convergence Tracking -----


def example_convergence_tracking():
    """Example with convergence detection."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Convergence Tracking")
    print("=" * 70)

    # Initialize progress tracker
    progress_tracker = ProgressTracker(
        convergence_window=5, convergence_threshold=0.01
    )

    # Create server
    server = FederatedServer(
        model=SimpleModel(),
        selection_strategy=RandomSelection(seed=42),
        num_clients_per_round=5,
    )

    # Create and register clients
    clients = create_clients(num_clients=10)
    for client_id, client in clients.items():
        server.register_client(client_id=client_id, dataset_size=100)

    # Run until convergence
    print("\nRunning until convergence...\n")
    max_rounds = 50
    for round_num in range(max_rounds):
        result = server.execute_round(round_num=round_num, clients=clients)

        # Update progress tracker
        loss = result['metrics'].get('avg_train_loss', 0)
        accuracy = result['metrics'].get('avg_train_accuracy', 0)
        progress_tracker.update(loss=loss, accuracy=accuracy)

        print(
            f"Round {round_num}: Loss={loss:.4f}, Acc={accuracy:.4f}",
            end="",
        )

        # Check convergence
        if round_num >= 10 and progress_tracker.has_converged(metric='loss'):
            print(" -> CONVERGED!")
            break
        else:
            print()

    # Display best results
    best = progress_tracker.get_best()
    print(f"\nBest loss:     {best['best_loss']:.4f}")
    print(f"Best accuracy: {best['best_accuracy']:.4f}")

    # Display improvement
    improvement = progress_tracker.get_improvement(window=5)
    print(f"\nLoss improvement (last 5 rounds):     {improvement['loss_improvement']:.4f}")
    print(f"Accuracy improvement (last 5 rounds): {improvement['accuracy_improvement']:.4f}")


# ----- Run Examples -----


def main():
    print("\n" + "=" * 70)
    print("FEDERATED LEARNING SERVER LOGGING SYSTEM")
    print("=" * 70)
    print("\nDemonstrating logging features similar to research repositories:")
    print("  - Structured logging (console + file)")
    print("  - Progress bar tracking")
    print("  - Metrics export (JSON/CSV)")
    print("  - TensorBoard integration")
    print("  - Convergence detection")

    # Run examples
    example_basic_logging()
    example_progress_bar()
    example_metrics_export()
    example_convergence_tracking()
    example_complete_logging()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nCheck the following directories:")
    print("  - logs/       : Structured log files")
    print("  - results/    : Exported metrics (JSON/CSV)")


if __name__ == "__main__":
    main()
