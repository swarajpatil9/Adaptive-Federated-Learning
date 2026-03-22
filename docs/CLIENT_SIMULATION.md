# Client Simulation Features

## Overview

Advanced simulation capabilities for realistic federated learning experiments. These features help researchers test FL algorithms under real-world conditions with heterogeneous, unreliable clients.

---

## Features

### 1. **Random Client Failures** (Dropout Simulation)

Simulate clients that randomly fail during training due to network issues, battery depletion, or disconnection.

```python
from aflf.client import SimulatedFederatedClient, ClientSimulationConfig

# Create client with 20% failure rate
sim_config = ClientSimulationConfig(failure_rate=0.2)

client = SimulatedFederatedClient(
    client_id=0,
    train_loader=train_loader,
    simulation_config=sim_config,
)

try:
    result = client.train(model)
except ClientFailureException:
    print("Client failed during training")
```

**Use cases:**
- Test FL algorithms under client dropout
- Evaluate aggregation robustness
- Study convergence with partial participation

---

### 2. **Variable Training Speed** (Heterogeneous Compute)

Simulate clients with different computational capabilities (mobile phones, laptops, servers).

```python
# Slow client (half speed)
slow_config = ClientSimulationConfig(training_speed=0.5)
slow_client = SimulatedFederatedClient(..., simulation_config=slow_config)

# Fast client (double speed)
fast_config = ClientSimulationConfig(training_speed=2.0)
fast_client = SimulatedFederatedClient(..., simulation_config=fast_config)

# Slow client will take 2x longer (simulated)
```

**Speed multipliers:**
- `0.25`: Very slow device (4x slower than baseline)
- `0.5`: Slow device (2x slower)
- `1.0`: Normal device (baseline)
- `2.0`: Fast device (2x faster)
- `4.0`: Very fast device (4x faster)

---

### 3. **Dataset Imbalance Analysis**

Measure how imbalanced the data distribution is across clients.

```python
from aflf.client import compute_dataset_imbalance_metrics

metrics = compute_dataset_imbalance_metrics(clients)

print(f"Imbalance ratio: {metrics['imbalance_ratio']:.2f}")
print(f"Gini coefficient: {metrics['gini_coefficient']:.3f}")
```

**Metrics returned:**
- `mean_samples`: Average samples per client
- `std_samples`: Standard deviation
- `min_samples`: Smallest client
- `max_samples`: Largest client
- `imbalance_ratio`: max/min (1.0 = perfect balance)
- `gini_coefficient`: 0-1 scale (0 = perfect balance, 1 = max imbalance)

**Interpretation:**
- **IID data**: Gini ≈ 0.01-0.05, ratio ≈ 1.0-1.2
- **Dirichlet α=0.5**: Gini ≈ 0.2-0.4, ratio ≈ 2-5
- **Dirichlet α=0.1**: Gini ≈ 0.4-0.6, ratio ≈ 10-50
- **Pathological**: Gini ≈ 0.5-0.7, ratio ≈ 50-100

---

### 4. **Straggler Simulation**

Simulate clients with additional delays (network latency, I/O bottlenecks).

```python
# Straggler with ~5 second delay
straggler_config = ClientSimulationConfig(
    stragglers_delay_mean=5.0,
    stragglers_delay_std=1.0,  # ±1s variance
)

client = SimulatedFederatedClient(..., simulation_config=straggler_config)

# Training will include real delay (not just simulated time)
result = client.train(model)
```

**Delay is sampled from:** N(mean, std²) and clipped to non-negative

---

### 5. **Heterogeneous Population Creation**

Automatically create diverse client populations with mixed characteristics.

```python
from aflf.client import create_heterogeneous_clients

clients = create_heterogeneous_clients(
    num_clients=10,
    train_loaders=loaders,
    failure_rate_range=(0.0, 0.3),      # 0-30% failure
    speed_range=(0.5, 2.0),             # 0.5x-2x speed
    straggler_probability=0.2,          # 20% are stragglers
    straggler_delay_range=(2.0, 10.0),  # 2-10s delay
)
```

This creates realistic FL environments where:
- Some clients are fast, some are slow
- Some clients are reliable, some fail often
- Some clients have network issues (stragglers)

---

## Complete Example

```python
from aflf.client import (
    SimulatedFederatedClient,
    ClientSimulationConfig,
    ClientFailureException,
    create_heterogeneous_clients,
    compute_dataset_imbalance_metrics,
)
from aflf.data import FederatedDataModule
from aflf.models import create_model

# Setup
data_module = FederatedDataModule(
    dataset_name='mnist',
    num_clients=10,
    partition_strategy='dirichlet',
    alpha=0.5,
)

model = create_model('simple_cnn', num_classes=10)

# Create heterogeneous clients
loaders = [data_module.get_client_loader(i) for i in range(10)]
clients = create_heterogeneous_clients(
    num_clients=10,
    train_loaders=loaders,
    failure_rate_range=(0.0, 0.2),
    speed_range=(0.5, 2.0),
)

# Analyze data imbalance
metrics = compute_dataset_imbalance_metrics(clients)
print(f"Gini coefficient: {metrics['gini_coefficient']:.3f}")

# Simulate training round with failures
results = []
for client in clients:
    try:
        result = client.train(model, config={'epochs': 5})
        results.append(result)
        print(f"Client {client.client_id}: Success ({result.training_time:.1f}s)")
    except ClientFailureException:
        print(f"Client {client.client_id}: Failed")

# Check success rates
for client in clients:
    stats = client.get_simulation_stats()
    print(f"Client {client.client_id}: {stats['success_rate']:.1%} success rate")
```

---

## Research Applications

### 1. **Testing Aggregation Robustness**
Test how aggregation algorithms (FedAvg, FedProx, etc.) handle client dropout:
```python
failure_rates = [0.0, 0.1, 0.3, 0.5]
for failure_rate in failure_rates:
    # Run FL with this failure rate
    # Measure convergence
```

### 2. **Resource-Aware Client Selection**
Select fast, reliable clients preferentially:
```python
# Select clients with low failure rate and high speed
selected = [
    c for c in clients
    if c.simulation_config.failure_rate < 0.1
    and c.simulation_config.training_speed > 1.0
]
```

### 3. **Fairness Under Imbalance**
Study how data imbalance affects model fairness:
```python
metrics = compute_dataset_imbalance_metrics(clients)
if metrics['gini_coefficient'] > 0.5:
    print("High imbalance - may need fairness interventions")
```

### 4. **Straggler Mitigation**
Test synchronous vs asynchronous aggregation:
```python
# Synchronous: wait for all (including stragglers)
# Asynchronous: aggregate as results arrive

# Compare convergence speed
```

---

## Configuration Reference

### ClientSimulationConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `failure_rate` | float | 0.0 | Probability of failure (0-1) |
| `training_speed` | float | 1.0 | Speed multiplier (>0) |
| `stragglers_delay_mean` | float | 0.0 | Mean delay in seconds |
| `stragglers_delay_std` | float | 0.0 | Delay std deviation |
| `max_retries` | int | 0 | Max retry attempts |
| `availability_window` | tuple | None | (start_hour, end_hour) |

---

## Performance Notes

### Simulation Overhead

- **Failure simulation**: Negligible overhead (single random number)
- **Speed simulation**: Affects reported time only, not real compute
- **Straggler delay**: Adds real delays (sleep) to simulate network/I/O

### Memory Usage

Simulation features add minimal memory overhead:
- `ClientSimulationConfig`: ~100 bytes
- Simulation stats tracking: ~50 bytes per client

### Recommendations

1. **For quick experiments**: Use speed simulation (no real delays)
2. **For realistic timing**: Include straggler delays
3. **For large-scale (1000+ clients)**: Disable verbose mode
4. **For reproducibility**: Fix random seed with `set_reproducibility()`

---

## Common Patterns

### Pattern 1: Test Dropout Resilience
```python
for failure_rate in [0.0, 0.1, 0.2, 0.3]:
    clients = [
        SimulatedFederatedClient(
            ...,
            simulation_config=ClientSimulationConfig(failure_rate=failure_rate)
        )
        for i in range(num_clients)
    ]
    # Run FL, measure convergence
```

### Pattern 2: Stratified Client Selection
```python
# Categorize clients by speed
fast_clients = [c for c in clients if c.simulation_config.training_speed >= 1.5]
slow_clients = [c for c in clients if c.simulation_config.training_speed < 0.7]

# Select mix: 70% fast, 30% slow
selected = random.sample(fast_clients, 7) + random.sample(slow_clients, 3)
```

### Pattern 3: Fairness-Aware Training
```python
# Give more epochs to clients with less data
for client in clients:
    epochs = int(5 * (max_samples / client.num_samples))
    result = client.train(model, config={'epochs': epochs})
```

---

## Testing

Run simulation tests:
```bash
.venv/bin/python -m pytest tests/client/test_simulation.py -v
```

Run simulation demo:
```bash
.venv/bin/python scripts/demo_client_simulation.py
```

---

## Related Documentation

- **Client Training**: See `aflf/client/README.md`
- **Data Partitioning**: See `aflf/data/README.md`
- **FL Algorithms**: See `aflf/aggregation/README.md` (Phase 5)

---

## References

**Papers using client simulation:**
- McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg, 2017)
- Li et al. "Federated Optimization in Heterogeneous Networks" (FedProx, 2020)
- Chen et al. "Asynchronous Federated Optimization" (2020)
- Kairouz et al. "Advances and Open Problems in Federated Learning" (2021)
