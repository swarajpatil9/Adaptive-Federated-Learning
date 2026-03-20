# Quick Start: Running the Code

## 🚀 One-Time Setup

```bash
cd /Users/swarajpatil/Developer/adaptive-federated-learning
source venv/bin/activate
pip install -e .
```

## ✅ Verify Everything Works

```bash
# Test Phase 3 (Models)
python3 scripts/test_phase3.py

# Test Phase 2 (Data)
python3 scripts/test_phase2.py

# Run all tests with coverage
pytest tests/ -v --cov=aflf
```

## 🔬 Demo Scripts

```bash
# Show data pipeline in action
python3 scripts/demo_data_pipeline.py

# Show models in action
python3 scripts/demo_models.py

# Custom examples
python3 scripts/demo_data_pipeline.py --dataset cifar10 --strategy dirichlet --alpha 0.1 --num-clients 50

python3 scripts/demo_models.py --model cnn --num-classes 10
```

## 📊 Interactive Python

```bash
# Start Python with venv active
source venv/bin/activate
python3

# Then run this:
from aflf.data import FederatedDataModule
from aflf.models import create_model, SimpleCNN

# Create 100 MNIST clients with non-IID data
dm = FederatedDataModule(
    dataset_name='mnist',
    num_clients=100,
    partition_strategy='dirichlet',
    alpha=0.5,
    seed=42
)

# Create model
model = create_model('cnn', num_classes=10)

# Get client data
client_loader = dm.get_client_loader(client_id=0)
batch_x, batch_y = next(iter(client_loader))

# Extract parameters for FL
params = model.get_parameters()
print(f"Model params: {len(params)} arrays")
print(f"Batch shape: {batch_x.shape}")
```

## 📋 Command Summary

| Command | What it does |
|---------|-------------|
| `python3 scripts/test_phase3.py` | Run all model tests (10+ tests) |
| `python3 scripts/test_phase2.py` | Run all data tests (10+ tests) |
| `pytest tests/ -v` | Run full test suite (85+ tests) |
| `python3 scripts/demo_data_pipeline.py` | Interactive data demo |
| `python3 scripts/demo_models.py` | Interactive model demo |

## 🐛 Troubleshooting

**Error: `command not found: python3`**
```bash
# Use full path to venv Python
/Users/swarajpatil/Developer/adaptive-federated-learning/venv/bin/python3 scripts/test_phase3.py
```

**Error: `No module named 'torch'`**
```bash
source venv/bin/activate
pip install -e .
```

**Error: Import failures**
```bash
# Make sure you're in venv
source venv/bin/activate

# And in the correct directory
cd /Users/swarajpatil/Developer/adaptive-federated-learning
```

## ✨ Expected Output

When you run `python3 scripts/test_phase3.py`:
```
================================================================================
PHASE 3 INTEGRATION TEST
================================================================================

[Test 1] Model Creation...
  ✓ Factory creates correct model type

[Test 2] Forward Pass...
  ✓ Forward pass produces correct output shape

...

[Test 10] Parameter Count Consistency...
  ✓ Parameter count: 421,642

================================================================================
ALL TESTS PASSED ✓
================================================================================
```

## 📚 What's Available

- **2,738 lines** of production code (Phase 2-3)
- **85+ test methods** across 6 test files
- **3 datasets**: MNIST, CIFAR-10, CIFAR-100
- **3 models**: SimpleCNN (62K), CNN (122K), CNNLarge (1.2M params)
- **3 partitioning strategies**: IID, Dirichlet, Pathological
- **Full FL infrastructure**: ready for Phase 4 training

---

**TL;DR - Copy & Paste:**

```bash
cd /Users/swarajpatil/Developer/adaptive-federated-learning
source venv/bin/activate
python3 scripts/test_phase3.py
```
