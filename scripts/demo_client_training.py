#!/usr/bin/env python3
"""
Demo script for PHASE 4: Client Training Pipeline

Demonstrates:
1. Creating federated clients
2. Local training execution
3. Metrics tracking
4. Weight extraction
5. Simulating multiple clients

This shows the client training system in isolation,
without server communication or aggregation.
"""

import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from aflf.client import FederatedClient, set_reproducibility
from aflf.data import FederatedDataModule
from aflf.models import create_model


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def demo_single_client():
    """Demonstrate single client training."""
    print_section("DEMO 1: Single Client Training")

    # Set seed for reproducibility
    set_reproducibility(42)

    # Create data module
    print("\n[1] Creating federated data module...")
    data_module = FederatedDataModule(
        dataset_name='mnist',
        data_root='./data',
        num_clients=10,
        partition_strategy='iid',
        batch_size=32,
        seed=42,
        download=True,
    )
    print(f"    ✓ Created {data_module.num_clients} clients")
    print(f"    ✓ Training samples: {len(data_module.train_dataset):,}")

    # Create global model
    print("\n[2] Creating global model...")
    global_model = create_model('simple_cnn', num_classes=10)
    print(f"    ✓ Model: SimpleCNN")
    print(f"    ✓ Parameters: {global_model.get_num_parameters():,}")

    # Create client 0
    print("\n[3] Creating client...")
    train_loader = data_module.get_client_loader(client_id=0)
    val_loader = data_module.get_test_loader()

    client = FederatedClient(
        client_id=0,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cpu',
        verbose=True,
    )
    print(f"    ✓ {client}")

    # Train locally
    print("\n[4] Executing local training...")
    training_config = {
        'epochs': 3,
        'lr': 0.01,
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 0.0001,
    }

    result = client.train(global_model, config=training_config)

    # Display results
    print("\n[5] Training results:")
    print(f"    • Client ID: {result.client_id}")
    print(f"    • Samples: {result.num_samples}")
    print(f"    • Train Loss: {result.train_loss:.4f}")
    print(f"    • Train Accuracy: {result.train_accuracy:.4f}")
    print(f"    • Val Loss: {result.val_loss:.4f}")
    print(f"    • Val Accuracy: {result.val_accuracy:.4f}")
    print(f"    • Training Time: {result.training_time:.2f}s")
    print(f"    • Weights Shape: {len(result.weights)} tensors")

    print("\n    ✓ Client training complete!")


def demo_multiple_clients():
    """Demonstrate multiple clients training in parallel."""
    print_section("DEMO 2: Multiple Clients (Simulating FL Round)")

    # Set seed
    set_reproducibility(42)

    # Create data module
    print("\n[1] Setting up federated environment...")
    data_module = FederatedDataModule(
        dataset_name='mnist',
        data_root='./data',
        num_clients=5,
        partition_strategy='dirichlet',
        batch_size=32,
        seed=42,
        alpha=0.5,  # Dirichlet concentration
    )
    print(f"    ✓ Created {data_module.num_clients} clients (Non-IID via Dirichlet)")

    # Create global model
    global_model = create_model('simple_cnn', num_classes=10)
    print(f"    ✓ Global model: SimpleCNN")

    # Create clients
    print("\n[2] Creating clients...")
    clients = []
    for client_id in range(data_module.num_clients):
        train_loader = data_module.get_client_loader(client_id)
        client = FederatedClient(
            client_id=client_id,
            train_loader=train_loader,
            device='cpu',
            verbose=False,
        )
        clients.append(client)
        print(f"    ✓ Client {client_id}: {client.num_samples} samples")

    # Simulate FL round
    print("\n[3] Simulating FL round (all clients train)...")
    training_config = {
        'epochs': 2,
        'lr': 0.01,
        'optimizer': 'sgd',
        'momentum': 0.9,
    }

    results = []
    for client in clients:
        print(f"\n    Training Client {client.client_id}...")
        result = client.train(global_model, config=training_config)
        results.append(result)
        print(f"      Loss: {result.train_loss:.4f}, Acc: {result.train_accuracy:.4f}, Time: {result.training_time:.2f}s")

    # Summary statistics
    print("\n[4] Round summary:")
    avg_loss = sum(r.train_loss for r in results) / len(results)
    avg_acc = sum(r.train_accuracy for r in results) / len(results)
    total_samples = sum(r.num_samples for r in results)
    total_time = sum(r.training_time for r in results)

    print(f"    • Total clients: {len(clients)}")
    print(f"    • Total samples: {total_samples}")
    print(f"    • Average loss: {avg_loss:.4f}")
    print(f"    • Average accuracy: {avg_acc:.4f}")
    print(f"    • Total training time: {total_time:.2f}s")

    print("\n    ✓ FL round complete!")
    print("\n    Note: In Phase 5, these weights will be aggregated by the server.")


def demo_weight_extraction():
    """Demonstrate weight extraction and loading."""
    print_section("DEMO 3: Weight Management")

    set_reproducibility(42)

    # Create two models
    print("\n[1] Creating two models...")
    model_1 = create_model('simple_cnn', num_classes=10)
    model_2 = create_model('simple_cnn', num_classes=10)
    print("    ✓ Model 1 (will be trained)")
    print("    ✓ Model 2 (will receive weights)")

    # Create minimal data
    print("\n[2] Creating data...")
    data_module = FederatedDataModule(
        dataset_name='mnist',
        data_root='./data',
        num_clients=1,
        batch_size=32,
    )
    train_loader = data_module.get_client_loader(0)

    # Train model 1
    print("\n[3] Training Model 1...")
    client = FederatedClient(
        client_id=0,
        train_loader=train_loader,
        device='cpu',
        verbose=False,
    )

    result = client.train(
        model_1,
        config={'epochs': 1, 'lr': 0.01}
    )
    print(f"    ✓ Trained loss: {result.train_loss:.4f}")

    # Extract weights
    print("\n[4] Extracting weights...")
    from aflf.client import get_model_weights, set_model_weights

    weights = get_model_weights(model_1)
    print(f"    ✓ Extracted {len(weights)} weight tensors")
    print(f"    ✓ First layer: {list(weights.keys())[0]}")
    print(f"    ✓ Shape: {weights[list(weights.keys())[0]].shape}")

    # Load into model 2
    print("\n[5] Loading weights into Model 2...")
    set_model_weights(model_2, weights)
    print("    ✓ Weights loaded")

    # Verify they match
    print("\n[6] Verifying weight equality...")
    weights_2 = get_model_weights(model_2)
    all_match = all(
        torch.allclose(weights[key], weights_2[key])
        for key in weights.keys()
    )
    print(f"    ✓ All weights match: {all_match}")


def demo_metrics_tracking():
    """Demonstrate metrics tracking."""
    print_section("DEMO 4: Metrics Tracking")

    from aflf.client import MetricsTracker

    print("\n[1] Creating metrics tracker...")
    tracker = MetricsTracker()
    print(f"    ✓ {tracker}")

    print("\n[2] Simulating training batches...")
    # Simulate 5 batches
    batch_losses = [0.8, 0.6, 0.5, 0.45, 0.4]
    batch_accs = [0.75, 0.82, 0.85, 0.88, 0.90]

    for i, (loss, acc) in enumerate(zip(batch_losses, batch_accs)):
        tracker.update(loss=loss, accuracy=acc, num_samples=32)
        print(f"    Batch {i+1}: loss={loss:.2f}, acc={acc:.2f}")

    print("\n[3] Getting epoch metrics...")
    metrics = tracker.get_metrics()
    print(f"    • Average loss: {metrics['loss']:.4f}")
    print(f"    • Average accuracy: {metrics['accuracy']:.4f}")
    print(f"    • Total samples: {metrics['num_samples']}")

    print("\n[4] Resetting tracker...")
    tracker.reset()
    print(f"    ✓ {tracker}")


def main():
    """Run all demos."""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "PHASE 4: CLIENT TRAINING PIPELINE" + " " * 25 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    try:
        # Run demos
        demo_single_client()
        demo_multiple_clients()
        demo_weight_extraction()
        demo_metrics_tracking()

        # Success
        print_section("✓ ALL DEMOS COMPLETE")
        print("\nClient training pipeline is working correctly!")
        print("\nKey capabilities demonstrated:")
        print("  1. Single client training with metrics")
        print("  2. Multiple clients (simulated FL round)")
        print("  3. Weight extraction and loading")
        print("  4. Metrics tracking and aggregation")
        print("\nNext phase (Phase 5): Server aggregation and FL orchestration")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
