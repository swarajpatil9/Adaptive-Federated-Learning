#!/usr/bin/env python3
"""
Advanced Demo: Client Simulation Features

Demonstrates realistic FL scenarios with:
1. Client failures (dropout)
2. Variable training speeds (heterogeneous compute)
3. Dataset imbalance analysis
4. Straggler clients
5. Availability windows
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from aflf.client import (
    ClientFailureException,
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
    """Demonstrate client failure/dropout simulation."""
    print_section("DEMO 1: Client Failures (Dropout)")

    set_reproducibility(42)

    # Create data and model
    data_module = FederatedDataModule(
        dataset_name='mnist',
        data_root='./data',
        num_clients=5,
        batch_size=32,
    )
    model = create_model('simple_cnn', num_classes=10)

    print("\n[1] Creating clients with different failure rates...")

    # Reliable client (0% failure)
    reliable_loader = data_module.get_client_loader(0)
    reliable_client = SimulatedFederatedClient(
        client_id=0,
        train_loader=reliable_loader,
        simulation_config=ClientSimulationConfig(failure_rate=0.0),
        device='cpu',
    )

    # Unreliable client (50% failure)
    unreliable_loader = data_module.get_client_loader(1)
    unreliable_client = SimulatedFederatedClient(
        client_id=1,
        train_loader=unreliable_loader,
        simulation_config=ClientSimulationConfig(failure_rate=0.5),
        device='cpu',
    )

    # Very unreliable client (90% failure)
    very_unreliable_loader = data_module.get_client_loader(2)
    very_unreliable_client = SimulatedFederatedClient(
        client_id=2,
        train_loader=very_unreliable_loader,
        simulation_config=ClientSimulationConfig(failure_rate=0.9),
        device='cpu',
    )

    clients = [reliable_client, unreliable_client, very_unreliable_client]

    print("\n[2] Simulating 10 training rounds...")

    for client in clients:
        print(f"\n  Client {client.client_id} (failure_rate={client.simulation_config.failure_rate:.1%}):")

        successes = 0
        failures = 0

        for round_num in range(10):
            try:
                result = client.train(model, config={'epochs': 1, 'lr': 0.01})
                successes += 1
            except ClientFailureException:
                failures += 1

        stats = client.get_simulation_stats()
        print(f"    Successes: {successes}/10 ({stats['success_rate']:.1%})")
        print(f"    Failures: {failures}/10")
        print(f"    Avg training time: {stats['avg_training_time']:.2f}s")

    print("\n    ✓ Client failure simulation complete!")


def demo_variable_speed():
    """Demonstrate variable training speed (heterogeneous compute)."""
    print_section("DEMO 2: Variable Training Speed (Heterogeneous Compute)")

    set_reproducibility(42)

    # Create data and model
    data_module = FederatedDataModule(
        dataset_name='mnist',
        data_root='./data',
        num_clients=4,
        batch_size=32,
    )
    model = create_model('simple_cnn', num_classes=10)

    print("\n[1] Creating clients with different compute speeds...")

    # High-end device (2x faster)
    fast_loader = data_module.get_client_loader(0)
    fast_client = SimulatedFederatedClient(
        client_id=0,
        train_loader=fast_loader,
        simulation_config=ClientSimulationConfig(training_speed=2.0),
        device='cpu',
        verbose=False,
    )
    print("    • Client 0: Fast device (2.0x speed)")

    # Normal device (1x speed)
    normal_loader = data_module.get_client_loader(1)
    normal_client = SimulatedFederatedClient(
        client_id=1,
        train_loader=normal_loader,
        simulation_config=ClientSimulationConfig(training_speed=1.0),
        device='cpu',
        verbose=False,
    )
    print("    • Client 1: Normal device (1.0x speed)")

    # Low-end device (0.5x speed)
    slow_loader = data_module.get_client_loader(2)
    slow_client = SimulatedFederatedClient(
        client_id=2,
        train_loader=slow_loader,
        simulation_config=ClientSimulationConfig(training_speed=0.5),
        device='cpu',
        verbose=False,
    )
    print("    • Client 2: Slow device (0.5x speed)")

    # Very slow device (0.25x speed)
    very_slow_loader = data_module.get_client_loader(3)
    very_slow_client = SimulatedFederatedClient(
        client_id=3,
        train_loader=very_slow_loader,
        simulation_config=ClientSimulationConfig(training_speed=0.25),
        device='cpu',
        verbose=False,
    )
    print("    • Client 3: Very slow device (0.25x speed)")

    clients = [fast_client, normal_client, slow_client, very_slow_client]

    print("\n[2] Training all clients (2 epochs each)...")

    results = []
    for client in clients:
        result = client.train(model, config={'epochs': 2})
        results.append(result)
        print(f"    Client {client.client_id}: {result.training_time:.2f}s")

    # Show relative speeds
    print("\n[3] Relative training times:")
    base_time = results[1].training_time  # Normal client
    for i, result in enumerate(results):
        relative = result.training_time / base_time
        print(f"    Client {i}: {relative:.2f}x the normal time")

    print("\n    ✓ Speed variation simulation complete!")


def demo_dataset_imbalance():
    """Demonstrate dataset imbalance analysis."""
    print_section("DEMO 3: Dataset Imbalance Analysis")

    set_reproducibility(42)

    print("\n[1] Creating IID dataset (balanced)...")

    iid_data = FederatedDataModule(
        dataset_name='mnist',
        data_root='./data',
        num_clients=10,
        partition_strategy='iid',
        seed=42,
    )

    iid_clients = []
    for i in range(iid_data.num_clients):
        loader = iid_data.get_client_loader(i)
        client = SimulatedFederatedClient(
            client_id=i,
            train_loader=loader,
        )
        iid_clients.append(client)

    iid_metrics = compute_dataset_imbalance_metrics(iid_clients)

    print("\n  IID Dataset Metrics:")
    print(f"    Mean samples: {iid_metrics['mean_samples']:.0f}")
    print(f"    Std samples: {iid_metrics['std_samples']:.0f}")
    print(f"    Min samples: {iid_metrics['min_samples']}")
    print(f"    Max samples: {iid_metrics['max_samples']}")
    print(f"    Imbalance ratio: {iid_metrics['imbalance_ratio']:.2f}:1")
    print(f"    Gini coefficient: {iid_metrics['gini_coefficient']:.3f}")

    print("\n[2] Creating Non-IID dataset (imbalanced)...")

    noniid_data = FederatedDataModule(
        dataset_name='mnist',
        data_root='./data',
        num_clients=10,
        partition_strategy='dirichlet',
        alpha=0.1,  # Low alpha = high imbalance
        seed=42,
    )

    noniid_clients = []
    for i in range(noniid_data.num_clients):
        loader = noniid_data.get_client_loader(i)
        client = SimulatedFederatedClient(
            client_id=i,
            train_loader=loader,
        )
        noniid_clients.append(client)

    noniid_metrics = compute_dataset_imbalance_metrics(noniid_clients)

    print("\n  Non-IID Dataset Metrics (Dirichlet α=0.1):")
    print(f"    Mean samples: {noniid_metrics['mean_samples']:.0f}")
    print(f"    Std samples: {noniid_metrics['std_samples']:.0f}")
    print(f"    Min samples: {noniid_metrics['min_samples']}")
    print(f"    Max samples: {noniid_metrics['max_samples']}")
    print(f"    Imbalance ratio: {noniid_metrics['imbalance_ratio']:.2f}:1")
    print(f"    Gini coefficient: {noniid_metrics['gini_coefficient']:.3f}")

    print("\n[3] Interpretation:")
    if iid_metrics['gini_coefficient'] < 0.2:
        print(f"    IID: Well-balanced (Gini = {iid_metrics['gini_coefficient']:.3f})")
    if noniid_metrics['gini_coefficient'] > 0.3:
        print(f"    Non-IID: Highly imbalanced (Gini = {noniid_metrics['gini_coefficient']:.3f})")

    print("\n    ✓ Dataset imbalance analysis complete!")


def demo_heterogeneous_clients():
    """Demonstrate creating heterogeneous clients."""
    print_section("DEMO 4: Heterogeneous FL Environment")

    set_reproducibility(42)

    print("\n[1] Creating diverse federated environment...")

    # Create data
    data_module = FederatedDataModule(
        dataset_name='mnist',
        data_root='./data',
        num_clients=10,
        partition_strategy='dirichlet',
        alpha=0.5,
        batch_size=32,
        seed=42,
    )

    # Create heterogeneous clients
    loaders = [data_module.get_client_loader(i) for i in range(10)]

    clients = create_heterogeneous_clients(
        num_clients=10,
        train_loaders=loaders,
        device='cpu',
        failure_rate_range=(0.0, 0.3),
        speed_range=(0.3, 2.0),
        straggler_probability=0.2,
        straggler_delay_range=(2.0, 8.0),
    )

    print(f"    ✓ Created {len(clients)} heterogeneous clients")

    print("\n[2] Client characteristics:")
    for i, client in enumerate(clients):
        config = client.simulation_config
        is_straggler = config.stragglers_delay_mean > 0
        print(f"    Client {i}:")
        print(f"      Samples: {client.num_samples}")
        print(f"      Failure rate: {config.failure_rate:.1%}")
        print(f"      Speed: {config.training_speed:.2f}x")
        if is_straggler:
            print(f"      Straggler: Yes (avg delay: {config.stragglers_delay_mean:.1f}s)")

    print("\n[3] Simulating one FL round...")

    model = create_model('simple_cnn', num_classes=10)
    successful_clients = 0
    failed_clients = 0
    total_time = 0

    for client in clients:
        try:
            result = client.train(model, config={'epochs': 1, 'lr': 0.01})
            successful_clients += 1
            total_time += result.training_time
            print(f"    ✓ Client {client.client_id}: {result.training_time:.2f}s, loss={result.train_loss:.3f}")
        except ClientFailureException:
            failed_clients += 1
            print(f"    ✗ Client {client.client_id}: Failed")

    print(f"\n[4] Round summary:")
    print(f"    Successful: {successful_clients}/{len(clients)} ({successful_clients/len(clients):.1%})")
    print(f"    Failed: {failed_clients}/{len(clients)}")
    print(f"    Total time: {total_time:.2f}s")
    print(f"    Avg time per client: {total_time/successful_clients:.2f}s" if successful_clients > 0 else "    N/A")

    # Compute imbalance
    metrics = compute_dataset_imbalance_metrics(clients)
    print(f"    Dataset imbalance ratio: {metrics['imbalance_ratio']:.2f}:1")

    print("\n    ✓ Heterogeneous FL simulation complete!")


def demo_straggler_clients():
    """Demonstrate straggler client simulation."""
    print_section("DEMO 5: Straggler Clients")

    set_reproducibility(42)

    # Create data and model
    data_module = FederatedDataModule(
        dataset_name='mnist',
        data_root='./data',
        num_clients=3,
        batch_size=32,
    )
    model = create_model('simple_cnn', num_classes=10)

    print("\n[1] Creating normal and straggler clients...")

    # Normal client
    normal_loader = data_module.get_client_loader(0)
    normal_client = SimulatedFederatedClient(
        client_id=0,
        train_loader=normal_loader,
        simulation_config=ClientSimulationConfig(
            stragglers_delay_mean=0.0,  # No delay
        ),
        device='cpu',
    )
    print("    • Client 0: Normal (no delay)")

    # Mild straggler
    mild_loader = data_module.get_client_loader(1)
    mild_straggler = SimulatedFederatedClient(
        client_id=1,
        train_loader=mild_loader,
        simulation_config=ClientSimulationConfig(
            stragglers_delay_mean=2.0,  # 2s delay
            stragglers_delay_std=0.5,
        ),
        device='cpu',
    )
    print("    • Client 1: Mild straggler (2s ± 0.5s delay)")

    # Severe straggler
    severe_loader = data_module.get_client_loader(2)
    severe_straggler = SimulatedFederatedClient(
        client_id=2,
        train_loader=severe_loader,
        simulation_config=ClientSimulationConfig(
            stragglers_delay_mean=5.0,  # 5s delay
            stragglers_delay_std=1.0,
        ),
        device='cpu',
    )
    print("    • Client 2: Severe straggler (5s ± 1s delay)")

    clients = [normal_client, mild_straggler, severe_straggler]

    print("\n[2] Training all clients...")

    for client in clients:
        result = client.train(model, config={'epochs': 1})
        print(f"    Client {client.client_id}: {result.training_time:.2f}s")

    print("\n    ✓ Straggler simulation complete!")
    print("\n    Note: Stragglers can delay synchronous FL rounds significantly.")


def main():
    """Run all demos."""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 18 + "CLIENT SIMULATION FEATURES DEMO" + " " * 29 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    try:
        demo_client_failures()
        demo_variable_speed()
        demo_dataset_imbalance()
        demo_heterogeneous_clients()
        demo_straggler_clients()

        # Success
        print_section("✓ ALL DEMOS COMPLETE")
        print("\nClient simulation features demonstrated:")
        print("  1. Random client failures/dropout")
        print("  2. Variable training speeds (heterogeneous compute)")
        print("  3. Dataset imbalance analysis (Gini coefficient)")
        print("  4. Heterogeneous FL environments")
        print("  5. Straggler clients (delayed responses)")
        print("\nThese features enable realistic FL research experiments!")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
