#!/usr/bin/env python3
"""
Demo script for client simulation features.

Demonstrates:
1. Random client failures
2. Variable training speed (heterogeneous compute)
3. Dataset imbalance analysis
4. Straggler simulation
5. Creating heterogeneous client populations
"""

import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from aflf.client import (
    ClientSimulationConfig,
    SimulatedFederatedClient,
    compute_dataset_imbalance_metrics,
    create_heterogeneous_clients,
    set_reproducibility,
)
from aflf.data import FederatedDataModule
from aflf.models import create_model


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def demo_client_failures():
    """Demonstrate random client failures."""
    print_section("DEMO 1: Random Client Failures")

    set_reproducibility(42)

    # Create data
    print("\n[1] Setting up environment...")
    data_module = FederatedDataModule(
        dataset_name='mnist',
        data_root='./data',
        num_clients=5,
        batch_size=32,
    )
    print(f"    ✓ Created {data_module.num_clients} clients")

    # Create model
    model = create_model('simple_cnn', num_classes=10)
    print(f"    ✓ Model: SimpleCNN")

    # Create clients with different failure rates
    print("\n[2] Creating clients with different failure rates...")
    failure_rates = [0.0, 0.1, 0.3, 0.5, 0.8]

    clients = []
    for client_id, failure_rate in enumerate(failure_rates):
        train_loader = data_module.get_client_loader(client_id)
        sim_config = ClientSimulationConfig(failure_rate=failure_rate)

        client = SimulatedFederatedClient(
            client_id=client_id,
            train_loader=train_loader,
            simulation_config=sim_config,
            device='cpu',
            verbose=False,
        )
        clients.append(client)
        print(f"    ✓ Client {client_id}: failure_rate={failure_rate:.1%}")

    # Simulate multiple training rounds
    print("\n[3] Simulating 10 training rounds...")
    print("\n    Client | Failures | Success Rate")
    print("    " + "-" * 40)

    for client in clients:
        for round_num in range(10):
            try:
                client.train(model, config={'epochs': 1})
            except Exception:
                pass  # Expected failures

        stats = client.get_simulation_stats()
        print(
            f"    {client.client_id:6} | "
            f"{stats['failed_attempts']:8} | "
            f"{stats['success_rate']:11.1%}"
        )

    print("\n    ✓ Failure simulation complete!")
    print("    Note: Higher failure rate → lower success rate")


def demo_variable_speed():
    """Demonstrate variable training speed."""
    print_section("DEMO 2: Variable Training Speed (Heterogeneous Compute)")

    set_reproducibility(42)

    # Create data
    print("\n[1] Setting up environment...")
    data_module = FederatedDataModule(
        dataset_name='mnist',
        data_root='./data',
        num_clients=4,
        batch_size=32,
    )

    model = create_model('simple_cnn', num_classes=10)
    print("    ✓ Environment ready")

    # Create clients with different speeds
    print("\n[2] Creating clients with different compute capabilities...")
    speeds = {
        0: 0.25,  # Very slow device (4x slower)
        1: 0.5,   # Slow device (2x slower)
        2: 1.0,   # Normal device
        3: 2.0,   # Fast device (2x faster)
    }

    clients = []
    for client_id, speed in speeds.items():
        train_loader = data_module.get_client_loader(client_id)
        sim_config = ClientSimulationConfig(training_speed=speed)

        client = SimulatedFederatedClient(
            client_id=client_id,
            train_loader=train_loader,
            simulation_config=sim_config,
            device='cpu',
            verbose=False,
        )
        clients.append(client)

        speed_label = {0.25: "Very Slow", 0.5: "Slow", 1.0: "Normal", 2.0: "Fast"}[speed]
        print(f"    ✓ Client {client_id}: {speed_label:10} (speed={speed:.2f}x)")

    # Train all clients
    print("\n[3] Training all clients (1 epoch)...")
    print("\n    Client | Speed    | Training Time")
    print("    " + "-" * 40)

    results = []
    for client in clients:
        result = client.train(model, config={'epochs': 1})
        results.append(result)
        speed = client.simulation_config.training_speed
        print(
            f"    {client.client_id:6} | {speed:6.2f}x  | "
            f"{result.training_time:12.2f}s"
        )

    print("\n    ✓ Speed simulation complete!")
    print("    Note: Slower devices take more time (simulated)")


def demo_dataset_imbalance():
    """Demonstrate dataset imbalance analysis."""
    print_section("DEMO 3: Dataset Imbalance Analysis")

    set_reproducibility(42)

    print("\n[1] Creating balanced (IID) dataset...")
    data_iid = FederatedDataModule(
        dataset_name='mnist',
        data_root='./data',
        num_clients=5,
        partition_strategy='iid',
        batch_size=32,
        seed=42,
    )

    clients_iid = []
    for i in range(5):
        loader = data_iid.get_client_loader(i)
        client = SimulatedFederatedClient(
            client_id=i,
            train_loader=loader,
            device='cpu',
        )
        clients_iid.append(client)

    print("\n[2] Creating imbalanced (Dirichlet α=0.1) dataset...")
    data_noniid = FederatedDataModule(
        dataset_name='mnist',
        data_root='./data',
        num_clients=5,
        partition_strategy='dirichlet',
        alpha=0.1,  # Low alpha = high imbalance
        batch_size=32,
        seed=42,
    )

    clients_noniid = []
    for i in range(5):
        loader = data_noniid.get_client_loader(i)
        client = SimulatedFederatedClient(
            client_id=i,
            train_loader=loader,
            device='cpu',
        )
        clients_noniid.append(client)

    # Compute metrics
    print("\n[3] Computing imbalance metrics...")

    metrics_iid = compute_dataset_imbalance_metrics(clients_iid)
    metrics_noniid = compute_dataset_imbalance_metrics(clients_noniid)

    print("\n    Metric              | IID (Balanced) | Non-IID (Imbalanced)")
    print("    " + "-" * 65)
    print(
        f"    Mean samples        | "
        f"{metrics_iid['mean_samples']:14.1f} | "
        f"{metrics_noniid['mean_samples']:20.1f}"
    )
    print(
        f"    Std dev             | "
        f"{metrics_iid['std_samples']:14.1f} | "
        f"{metrics_noniid['std_samples']:20.1f}"
    )
    print(
        f"    Min samples         | "
        f"{metrics_iid['min_samples']:14.0f} | "
        f"{metrics_noniid['min_samples']:20.0f}"
    )
    print(
        f"    Max samples         | "
        f"{metrics_iid['max_samples']:14.0f} | "
        f"{metrics_noniid['max_samples']:20.0f}"
    )
    print(
        f"    Imbalance ratio     | "
        f"{metrics_iid['imbalance_ratio']:14.2f} | "
        f"{metrics_noniid['imbalance_ratio']:20.2f}"
    )
    print(
        f"    Gini coefficient    | "
        f"{metrics_iid['gini_coefficient']:14.3f} | "
        f"{metrics_noniid['gini_coefficient']:20.3f}"
    )

    print("\n    ✓ Imbalance analysis complete!")
    print("    Note: Lower Gini = more balanced, Higher Gini = more imbalanced")


def demo_stragglers():
    """Demonstrate straggler simulation."""
    print_section("DEMO 4: Straggler Simulation")

    set_reproducibility(42)

    print("\n[1] Setting up environment...")
    data_module = FederatedDataModule(
        dataset_name='mnist',
        data_root='./data',
        num_clients=5,
        batch_size=32,
    )

    model = create_model('simple_cnn', num_classes=10)
    print("    ✓ Environment ready")

    print("\n[2] Creating clients (some are stragglers)...")

    # Normal clients
    normal_clients = []
    for i in range(3):
        loader = data_module.get_client_loader(i)
        config = ClientSimulationConfig(stragglers_delay_mean=0.0)
        client = SimulatedFederatedClient(
            client_id=i,
            train_loader=loader,
            simulation_config=config,
            device='cpu',
            verbose=False,
        )
        normal_clients.append(client)
        print(f"    ✓ Client {i}: Normal (no delay)")

    # Straggler clients
    straggler_clients = []
    straggler_delays = [1.0, 3.0]  # Seconds
    for i, delay in enumerate(straggler_delays, start=3):
        loader = data_module.get_client_loader(i)
        config = ClientSimulationConfig(
            stragglers_delay_mean=delay,
            stragglers_delay_std=delay * 0.1,
        )
        client = SimulatedFederatedClient(
            client_id=i,
            train_loader=loader,
            simulation_config=config,
            device='cpu',
            verbose=False,
        )
        straggler_clients.append(client)
        print(f"    ✓ Client {i}: Straggler (~{delay:.1f}s delay)")

    # Train all clients
    print("\n[3] Training all clients...")
    print("\n    Client | Type      | Training Time")
    print("    " + "-" * 40)

    all_clients = normal_clients + straggler_clients
    for client in all_clients:
        result = client.train(model, config={'epochs': 1})
        client_type = "Normal" if client.simulation_config.stragglers_delay_mean == 0 else "Straggler"
        print(
            f"    {client.client_id:6} | {client_type:9} | "
            f"{result.training_time:12.2f}s"
        )

    print("\n    ✓ Straggler simulation complete!")
    print("    Note: Stragglers take longer due to delays (network, I/O, etc.)")


def demo_heterogeneous_population():
    """Demonstrate creating heterogeneous client population."""
    print_section("DEMO 5: Heterogeneous Client Population")

    set_reproducibility(42)

    print("\n[1] Creating data for 10 clients...")
    data_module = FederatedDataModule(
        dataset_name='mnist',
        data_root='./data',
        num_clients=10,
        partition_strategy='dirichlet',
        alpha=0.5,
        batch_size=32,
    )

    loaders = [data_module.get_client_loader(i) for i in range(10)]
    print("    ✓ Data ready")

    print("\n[2] Creating heterogeneous client population...")
    print("    (Random failure rates, speeds, and straggler status)")

    clients = create_heterogeneous_clients(
        num_clients=10,
        train_loaders=loaders,
        device='cpu',
        failure_rate_range=(0.0, 0.3),
        speed_range=(0.5, 2.0),
        straggler_probability=0.3,
        straggler_delay_range=(2.0, 5.0),
    )

    print("\n    Client | Samples | Failure Rate | Speed | Straggler")
    print("    " + "-" * 65)

    for client in clients:
        is_straggler = client.simulation_config.stragglers_delay_mean > 0
        straggler_str = f"Yes (~{client.simulation_config.stragglers_delay_mean:.1f}s)" if is_straggler else "No"

        print(
            f"    {client.client_id:6} | "
            f"{client.num_samples:7} | "
            f"{client.simulation_config.failure_rate:12.1%} | "
            f"{client.simulation_config.training_speed:5.2f}x | "
            f"{straggler_str}"
        )

    # Analyze imbalance
    print("\n[3] Dataset imbalance analysis...")
    metrics = compute_dataset_imbalance_metrics(clients)
    print(f"    • Imbalance ratio: {metrics['imbalance_ratio']:.2f}")
    print(f"    • Gini coefficient: {metrics['gini_coefficient']:.3f}")

    print("\n    ✓ Heterogeneous population created!")
    print("    This simulates a realistic FL environment with diverse devices.")


def main():
    """Run all demos."""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 22 + "CLIENT SIMULATION FEATURES" + " " * 30 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    try:
        # Run demos
        demo_client_failures()
        demo_variable_speed()
        demo_dataset_imbalance()
        demo_stragglers()
        demo_heterogeneous_population()

        # Success
        print_section("✓ ALL SIMULATION DEMOS COMPLETE")
        print("\nClient simulation features implemented:")
        print("  1. Random client failures (dropout simulation)")
        print("  2. Variable training speed (heterogeneous compute)")
        print("  3. Dataset imbalance analysis (Gini coefficient)")
        print("  4. Straggler simulation (slow clients)")
        print("  5. Heterogeneous population creation")
        print("\nThese features enable realistic FL experiments and algorithm testing.")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
