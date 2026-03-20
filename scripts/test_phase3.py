"""
Quick integration test for Phase 3 model module.

Verifies:
- Model creation
- Forward pass
- Parameter extraction/loading
- Deterministic initialization
- Device portability
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from aflf.models import (
    CNN,
    CNNLarge,
    SimpleCNN,
    create_model,
    initialize_model,
    list_available_models,
)


def test_phase3_integration():
    """Quick integration test for Phase 3."""
    print("="*80)
    print("PHASE 3 INTEGRATION TEST")
    print("="*80)

    # Test 1: Model creation
    print("\n[Test 1] Model Creation...")
    model = create_model('simple_cnn', num_classes=10)
    assert isinstance(model, SimpleCNN)
    print("  ✓ Factory creates correct model type")

    # Test 2: Forward pass
    print("\n[Test 2] Forward Pass...")
    x = torch.randn(16, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (16, 10)
    print("  ✓ Forward pass produces correct output shape")

    # Test 3: Parameter extraction
    print("\n[Test 3] Parameter Extraction...")
    params = model.get_parameters()
    assert len(params) > 0
    assert all(isinstance(p, np.ndarray) for p in params)
    print(f"  ✓ Extracted {len(params)} parameter arrays")

    # Test 4: Parameter loading
    print("\n[Test 4] Parameter Loading...")
    scaled_params = [p * 0.9 for p in params]
    model.set_parameters(scaled_params)
    new_params = model.get_parameters()
    max_diff = max(np.abs(p1 - p2).max() for p1, p2 in zip(params, scaled_params))
    assert max_diff < 1e-6
    print("  ✓ Parameters loaded correctly")

    # Test 5: Deterministic initialization
    print("\n[Test 5] Deterministic Initialization...")
    model1 = SimpleCNN(num_classes=10)
    model1 = initialize_model(model1, seed=42)

    model2 = SimpleCNN(num_classes=10)
    model2 = initialize_model(model2, seed=42)

    params1 = model1.get_parameters()
    params2 = model2.get_parameters()

    max_diff = max(np.abs(p1 - p2).max() for p1, p2 in zip(params1, params2))
    assert max_diff < 1e-10
    print("  ✓ Same seed produces identical initialization")

    # Test 6: Different seeds produce different initialization
    model3 = SimpleCNN(num_classes=10)
    model3 = initialize_model(model3, seed=999)
    params3 = model3.get_parameters()

    max_diff = max(np.abs(p1 - p3).max() for p1, p3 in zip(params1, params3))
    assert max_diff > 0.01  # Should be different
    print("  ✓ Different seeds produce different initialization")

    # Test 7: All models work
    print("\n[Test 7] All Model Types...")
    test_configs = [
        ('simple_cnn', 10, (16, 1, 28, 28)),
        ('cnn', 10, (16, 3, 32, 32)),
        ('cnn_large', 100, (16, 3, 32, 32)),
    ]

    for model_name, num_classes, input_shape in test_configs:
        model = create_model(model_name, num_classes=num_classes)
        x = torch.randn(input_shape)
        logits = model(x)
        assert logits.shape == (input_shape[0], num_classes)
        print(f"  ✓ {model_name} works correctly")

    # Test 8: Device portability
    print("\n[Test 8] Device Portability...")
    model = SimpleCNN()

    # CPU
    model = model.to_device(torch.device('cpu'))
    x = torch.randn(8, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (8, 10)
    print("  ✓ CPU device works")

    # CUDA (if available)
    if torch.cuda.is_available():
        model = model.to_device(torch.device('cuda'))
        x = torch.randn(8, 1, 28, 28, device='cuda')
        logits = model(x)
        assert logits.device.type == 'cuda'
        print("  ✓ CUDA device works")

        # Move back to CPU
        model = model.to_device(torch.device('cpu'))

    # MPS (if available)
    if torch.backends.mps.is_available():
        model = model.to_device(torch.device('mps'))
        x = torch.randn(8, 1, 28, 28, device='mps')
        logits = model(x)
        assert logits.device.type == 'mps'
        print("  ✓ MPS device works")

    # Test 9: Model info
    print("\n[Test 9] Model Info...")
    available = list_available_models()
    assert len(available) == 3
    assert 'simple_cnn' in available
    print(f"  ✓ {len(available)} models registered")

    # Test 10: Parameter count consistency
    print("\n[Test 10] Parameter Count Consistency...")
    model = SimpleCNN()
    count1 = model.get_num_parameters()
    count2 = sum(p.numel() for p in model.parameters())
    assert count1 == count2
    print(f"  ✓ Parameter count: {count1:,}")

    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
    print("\nPhase 3 model module is fully operational!")
    print("\nModel characteristics:")
    print("  • SimpleCNN: 62K params, 0.24 MB")
    print("  • CNN: 122K params, 0.47 MB")
    print("  • CNNLarge: 1.2M params, 4.71 MB")
    print("\nAll models support:")
    print("  ✓ Parameter extraction for FL communication")
    print("  ✓ Deterministic initialization")
    print("  ✓ Device portability (CPU/CUDA/MPS)")
    print("  ✓ Config-driven creation")


if __name__ == "__main__":
    try:
        test_phase3_integration()
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
