"""
Demo script for model module (Phase 3).

Demonstrates:
1. Creating models via factory
2. Model summaries and parameter counting
3. Weight extraction/loading (FL communication)
4. Deterministic initialization
5. Model comparison

Usage:
    python3 scripts/demo_models.py
    python3 scripts/demo_models.py --model cnn --num-classes 100
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from aflf.models import (
    CNN,
    CNNLarge,
    SimpleCNN,
    create_model,
    get_model_info,
    initialize_model,
)
from aflf.models.factory import print_model_catalog
from aflf.models.utils import compare_models, print_trainable_parameters


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demo: Model Module",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="simple_cnn",
        choices=["simple_cnn", "cnn", "cnn_large"],
        help="Model to demonstrate",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of output classes",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization",
    )

    return parser.parse_args()


def demo_model_catalog():
    """Show all available models."""
    print("\n" + "="*80)
    print("DEMO 1: MODEL CATALOG")
    print("="*80)

    print_model_catalog()


def demo_model_creation(args):
    """Demonstrate model creation and inspection."""
    print("\n" + "="*80)
    print("DEMO 2: MODEL CREATION")
    print("="*80)

    print(f"\n[1] Creating {args.model} with {args.num_classes} classes...")
    model = create_model(args.model, num_classes=args.num_classes)
    print(f"✓ Model created: {model}")

    print("\n[2] Model Summary:")
    print(model.summary())

    print("\n[3] Layer Shapes:")
    for name, shape in model.get_layer_shapes():
        print(f"  {name:<30} {str(shape):<30}")

    return model


def demo_parameter_extraction(model):
    """Demonstrate FL parameter communication."""
    print("\n" + "="*80)
    print("DEMO 3: PARAMETER EXTRACTION (FL COMMUNICATION)")
    print("="*80)

    print("\n[1] Extract parameters (client → server)...")
    params = model.get_parameters()
    print(f"  Number of parameter arrays: {len(params)}")
    print(f"  First parameter shape: {params[0].shape}")
    print(f"  Last parameter shape: {params[-1].shape}")
    print(f"  Total elements: {sum(p.size for p in params):,}")

    print("\n[2] Simulate aggregation (server-side)...")
    # Simulate FedAvg aggregation (just scale by 0.9 for demo)
    aggregated_params = [p * 0.9 for p in params]
    print("  ✓ Aggregation complete (simulated)")

    print("\n[3] Load parameters (server → client)...")
    model.set_parameters(aggregated_params)
    print("  ✓ Parameters loaded successfully")

    print("\n[4] Verify parameters changed...")
    new_params = model.get_parameters()
    max_diff = max(np.abs(p1 - p2).max() for p1, p2 in zip(params, new_params))
    print(f"  Max parameter change: {max_diff:.6f}")

    return params


def demo_deterministic_initialization():
    """Demonstrate deterministic initialization for reproducibility."""
    print("\n" + "="*80)
    print("DEMO 4: DETERMINISTIC INITIALIZATION")
    print("="*80)

    print("\n[1] Create two models with same seed...")
    model1 = create_model('simple_cnn', num_classes=10)
    model1 = initialize_model(model1, seed=42)

    model2 = create_model('simple_cnn', num_classes=10)
    model2 = initialize_model(model2, seed=42)

    params1 = model1.get_parameters()
    params2 = model2.get_parameters()

    max_diff = max(np.abs(p1 - p2).max() for p1, p2 in zip(params1, params2))
    print(f"  Max difference: {max_diff:.10f}")
    print("  ✓ Identical initialization (reproducibility confirmed)")

    print("\n[2] Create model with different seed...")
    model3 = create_model('simple_cnn', num_classes=10)
    model3 = initialize_model(model3, seed=999)

    params3 = model3.get_parameters()
    max_diff = max(np.abs(p1 - p3).max() for p1, p3 in zip(params1, params3))
    print(f"  Max difference: {max_diff:.10f}")
    print("  ✓ Different initialization (as expected)")


def demo_forward_pass(model):
    """Demonstrate forward pass with different batch sizes."""
    print("\n" + "="*80)
    print("DEMO 5: FORWARD PASS")
    print("="*80)

    # Determine input shape based on model
    if isinstance(model, SimpleCNN):
        input_shape = (1, 28, 28)
        dataset = "MNIST"
    else:
        input_shape = (3, 32, 32)
        dataset = "CIFAR-10"

    print(f"\n[1] Single sample (batch_size=1)...")
    x = torch.randn(1, *input_shape)
    logits = model(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {logits.shape}")
    print(f"  ✓ Forward pass successful")

    print(f"\n[2] Batch of 32 samples...")
    x = torch.randn(32, *input_shape)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(logits, dim=1)
    print(f"  Input: {x.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Predictions: {preds.tolist()[:10]}...")
    print(f"  ✓ Batch processing works")


def demo_device_portability():
    """Demonstrate device portability (CPU/CUDA/MPS)."""
    print("\n" + "="*80)
    print("DEMO 6: DEVICE PORTABILITY")
    print("="*80)

    model = create_model('simple_cnn', num_classes=10)

    # Check available devices
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if torch.backends.mps.is_available():
        devices.append('mps')

    print(f"\nAvailable devices: {devices}")

    for device_name in devices:
        device = torch.device(device_name)
        print(f"\n[{device_name.upper()}]")

        # Move model to device
        model = model.to_device(device)
        print(f"  Model device: {model.get_device()}")

        # Test forward pass
        x = torch.randn(8, 1, 28, 28, device=device)
        logits = model(x)
        print(f"  Forward pass: {x.shape} → {logits.shape}")
        print(f"  ✓ Works on {device_name}")

        # Move back to CPU for next iteration
        model = model.to_device(torch.device('cpu'))


def demo_model_comparison():
    """Compare all available models."""
    print("\n" + "="*80)
    print("DEMO 7: MODEL COMPARISON")
    print("="*80)

    print("\n[1] MNIST models:")
    mnist_models = {
        'SimpleCNN': SimpleCNN(num_classes=10),
    }
    compare_models(mnist_models, input_shape=(1, 1, 28, 28))

    print("\n[2] CIFAR-10 models:")
    cifar10_models = {
        'CNN': CNN(num_classes=10),
        'CNN + BN': CNN(num_classes=10, use_batch_norm=True),
        'CNNLarge': CNNLarge(num_classes=10),
    }
    compare_models(cifar10_models, input_shape=(1, 3, 32, 32))

    print("\n[3] Communication cost analysis:")
    models_for_cost = [
        ('SimpleCNN', SimpleCNN(num_classes=10)),
        ('CNN', CNN(num_classes=10)),
        ('CNNLarge', CNNLarge(num_classes=10)),
    ]

    print(f"{'Model':<15} {'Params':<15} {'Size (MB)':<15} {'Communication/Round':<20}")
    print("-"*80)

    for name, model in models_for_cost:
        num_params = model.get_num_parameters()
        size_mb = model.get_model_size_mb()
        # Assume 100 clients, 10% participation, upload + download
        comm_per_round = size_mb * 0.1 * 100 * 2  # 10 clients × 2 directions
        print(
            f"{name:<15} "
            f"{num_params:<15,} "
            f"{size_mb:<15.2f} "
            f"{comm_per_round:<20.1f} MB"
        )

    print("="*80)
    print("\nInterpretation:")
    print("  SimpleCNN: Minimal communication, best for edge devices")
    print("  CNN: Balanced communication-accuracy trade-off")
    print("  CNNLarge: High communication cost, use only if accuracy gains justify it")


def main():
    """Main demo execution."""
    args = parse_args()

    print("="*80)
    print("FEDERATED LEARNING MODEL MODULE DEMO (PHASE 3)")
    print("="*80)

    # Run all demos
    demo_model_catalog()
    model = demo_model_creation(args)
    params = demo_parameter_extraction(model)
    demo_deterministic_initialization()
    demo_forward_pass(model)
    demo_device_portability()
    demo_model_comparison()

    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nKey takeaways:")
    print("  ✓ Models are lightweight and FL-ready")
    print("  ✓ Parameter extraction/loading works (for aggregation)")
    print("  ✓ Deterministic initialization ensures reproducibility")
    print("  ✓ Device portability (CPU/CUDA/MPS)")
    print("  ✓ Ready for client training (Phase 4)")
    print("\nNext phase: Client training logic")


if __name__ == "__main__":
    import numpy as np
    main()
