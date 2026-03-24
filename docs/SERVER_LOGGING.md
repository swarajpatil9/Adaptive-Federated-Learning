# Server Logging System Documentation

## Overview

The federated learning server includes a comprehensive logging system similar to research repositories like **Flower**, **FedML**, and **PySyft**. The system tracks:

- **Round execution**: Start/end times, duration, selected clients
- **Client participation**: Success/failure, metrics per client
- **Global progress**: Loss/accuracy trends, convergence detection
- **System events**: Initialization, checkpoints, warnings

## Components

### 1. ServerLogger

Structured logging to console and file with multiple log levels.

**Features**:
- Console output with timestamps
- File logging with rotation
- Multiple log levels (DEBUG, INFO, WARNING, ERROR)
- Formatted output for readability

**Usage**:
```python
from aflf.server import ServerLogger, FederatedServer

# Initialize logger
logger = ServerLogger(
    experiment_name="mnist_fedavg",
    log_dir="logs",
    console_level=logging.INFO,
    file_level=logging.DEBUG
)

# Create server with logger
server = FederatedServer(
    model=model,
    server_logger=logger
)

# Logging happens automatically during training
server.run_training(num_rounds=50, clients=clients)
```

**Output Example**:
```
======================================================================
FEDERATED SERVER INITIALIZED
======================================================================
Experiment:          mnist_fedavg
Model:               SimpleCNN
Total clients:       100
Selection strategy:  RandomSelection(seed=42)
======================================================================

----------------------------------------------------------------------
ROUND 0 STARTED
----------------------------------------------------------------------
Selected 10 clients for training

----------------------------------------------------------------------
ROUND 0 COMPLETED
----------------------------------------------------------------------
Participation:  9/10 clients (90.0%)
Failed clients: 1
Duration:       15.23s

Metrics:
  Train Loss:     0.4523
  Train Accuracy: 0.8721
  Total Samples:  5400
----------------------------------------------------------------------
```

### 2. MetricsTracker

Tracks and exports metrics to JSON, CSV, and TensorBoard.

**Features**:
- Round-level metrics storage
- Client-level result tracking
- JSON export (structured data)
- CSV export (spreadsheet compatible)
- TensorBoard integration (optional)
- Summary statistics computation

**Usage**:
```python
from aflf.server import MetricsTracker

# Initialize tracker
tracker = MetricsTracker(
    experiment_name="mnist_fedavg",
    output_dir="results",
    enable_tensorboard=True,
    tensorboard_dir="runs/mnist"
)

# Create server with tracker
server = FederatedServer(
    model=model,
    metrics_tracker=tracker
)

# Run training
server.run_training(num_rounds=50, clients=clients)

# Export metrics
server.export_metrics(export_json=True, export_csv=True)

# Cleanup
tracker.close()
```

**JSON Export Format**:
```json
{
  "experiment_name": "mnist_fedavg",
  "summary": {
    "num_rounds": 50,
    "total_time_seconds": 756.2,
    "avg_round_time": 15.12,
    "final_train_loss": 0.1234,
    "final_train_accuracy": 0.9567
  },
  "round_metrics": [
    {
      "round": 0,
      "elapsed_time": 15.23,
      "num_participating": 9,
      "participation_rate": 0.9,
      "avg_train_loss": 0.4523,
      "avg_train_accuracy": 0.8721
    },
    ...
  ],
  "client_metrics": {
    "0": [
      {
        "round": 0,
        "train_loss": 0.45,
        "train_accuracy": 0.87,
        "num_samples": 600
      }
    ]
  }
}
```

**CSV Export** (rounds):
```csv
round,elapsed_time,num_participating,participation_rate,avg_train_loss,avg_train_accuracy
0,15.23,9,0.9,0.4523,0.8721
1,14.87,10,1.0,0.3821,0.8954
...
```

### 3. ConsoleProgressLogger

Real-time progress bar for training visualization.

**Features**:
- ASCII progress bar
- Live metrics display
- ETA calculation
- No external dependencies (pure Python)

**Usage**:
```python
# Automatic progress bar with run_training()
server.run_training(
    num_rounds=50,
    clients=clients,
    show_progress=True
)
```

**Output**:
```
Round 25/50 |█████████████████████████░░░░░░░░░░░░░░░░░░░░░░░| 50% [8/10 clients] Loss: 0.2345 Acc: 0.9123 ETA: 380s
```

### 4. ProgressTracker

Convergence detection and improvement tracking.

**Features**:
- Convergence detection (variance-based)
- Best metrics tracking
- Improvement calculation over windows
- Early stopping support

**Usage**:
```python
from aflf.server import ProgressTracker

# Initialize tracker
progress = ProgressTracker(
    convergence_window=5,
    convergence_threshold=0.001
)

# Run until convergence
for round_num in range(max_rounds):
    result = server.execute_round(round_num, clients)

    loss = result['metrics']['avg_train_loss']
    accuracy = result['metrics']['avg_train_accuracy']
    progress.update(loss=loss, accuracy=accuracy)

    # Check convergence
    if progress.has_converged(metric='loss'):
        print(f"Converged at round {round_num}!")
        break

# Get best results
best = progress.get_best()
print(f"Best loss: {best['best_loss']:.4f}")
print(f"Best accuracy: {best['best_accuracy']:.4f}")
```

## Complete Integration Example

```python
from aflf.server import (
    FederatedServer,
    ServerLogger,
    MetricsTracker,
    ProgressTracker
)
from aflf.selection import RandomSelection

# 1. Initialize logging components
logger = ServerLogger(
    experiment_name="mnist_fedavg_full",
    log_dir="logs",
    console_level=logging.INFO
)

tracker = MetricsTracker(
    experiment_name="mnist_fedavg_full",
    output_dir="results",
    enable_tensorboard=True
)

progress = ProgressTracker(
    convergence_window=5,
    convergence_threshold=0.001
)

# 2. Create server with logging
server = FederatedServer(
    model=model,
    selection_strategy=RandomSelection(seed=42),
    num_clients_per_round=10,
    server_logger=logger,
    metrics_tracker=tracker
)

# 3. Register clients
for client_id, client in enumerate(clients):
    server.register_client(client_id=client_id, dataset_size=600)

# 4. Run training with convergence detection
max_rounds = 100
for round_num in range(max_rounds):
    result = server.execute_round(round_num=round_num, clients=clients_dict)

    # Update progress tracker
    metrics = result['metrics']
    progress.update(
        loss=metrics['avg_train_loss'],
        accuracy=metrics['avg_train_accuracy']
    )

    # Check convergence
    if round_num >= 10 and progress.has_converged():
        logger.log_info(f"Training converged at round {round_num}")
        break

# 5. Export results
paths = server.export_metrics(export_json=True, export_csv=True)
print(f"Results exported to: {paths}")

# 6. Cleanup
tracker.close()
```

## TensorBoard Integration

When `enable_tensorboard=True`, metrics are automatically logged to TensorBoard:

```bash
# Start TensorBoard
tensorboard --logdir results/tensorboard

# View at http://localhost:6006
```

**Logged Metrics**:
- `Loss/train` - Training loss per round
- `Loss/val` - Validation loss per round
- `Accuracy/train` - Training accuracy per round
- `Accuracy/val` - Validation accuracy per round
- `Participation/rate` - Client participation rate
- `Participation/num_clients` - Number of participating clients
- `Failure/rate` - Client failure rate
- `Timing/round_duration` - Round execution time

## File Organization

```
project/
├── logs/                          # Structured logs
│   └── mnist_fedavg_20240324_143022.log
├── results/                       # Metrics exports
│   ├── mnist_fedavg_metrics_20240324_143022.json
│   ├── mnist_fedavg_rounds_20240324_143022.csv
│   ├── mnist_fedavg_clients_20240324_143022.csv
│   └── tensorboard/               # TensorBoard logs
│       └── mnist_fedavg/
│           └── events.out.tfevents...
```

## Research Comparison

| Feature | Flower | FedML | PySyft | **AFLF (Ours)** |
|---------|--------|-------|--------|-----------------|
| Structured logging | ✅ | ✅ | ❌ | ✅ |
| Progress bar | ✅ | ✅ | ❌ | ✅ |
| JSON/CSV export | ✅ | ✅ | ❌ | ✅ |
| TensorBoard | ✅ | ✅ | ❌ | ✅ |
| Convergence tracking | ❌ | ✅ | ❌ | ✅ |
| Client-level metrics | ✅ | ✅ | ❌ | ✅ |
| No external deps (progress) | ❌ | ❌ | - | ✅ |

## Best Practices

1. **Always use logging in experiments**:
   ```python
   logger = ServerLogger(experiment_name="your_experiment")
   server = FederatedServer(..., server_logger=logger)
   ```

2. **Export metrics for reproducibility**:
   ```python
   tracker = MetricsTracker(experiment_name="your_experiment")
   server = FederatedServer(..., metrics_tracker=tracker)
   server.export_metrics()  # Always export after training
   ```

3. **Use TensorBoard for visualization**:
   ```python
   tracker = MetricsTracker(..., enable_tensorboard=True)
   # Then run: tensorboard --logdir results/tensorboard
   ```

4. **Implement early stopping**:
   ```python
   progress = ProgressTracker(convergence_window=5)
   # Check progress.has_converged() each round
   ```

5. **Log to files for long experiments**:
   ```python
   logger = ServerLogger(..., enable_file_logging=True)
   # Logs persist even if console output is lost
   ```

## Advanced Usage

### Custom Logging Levels

```python
import logging

logger = ServerLogger(
    experiment_name="debug_run",
    console_level=logging.DEBUG,  # Show everything
    file_level=logging.DEBUG
)
```

### Programmatic Metric Access

```python
# Get all round metrics
round_metrics = tracker.round_metrics
for rm in round_metrics:
    print(f"Round {rm['round']}: Loss={rm['avg_train_loss']}")

# Get summary statistics
summary = tracker.compute_summary()
print(f"Average participation rate: {summary['participation_rate_mean']:.2%}")
```

### Custom Progress Bar Width

```python
from aflf.server import ConsoleProgressLogger

progress = ConsoleProgressLogger(total_rounds=50, bar_width=70)
```

## Troubleshooting

**TensorBoard not working?**
```bash
pip install tensorboard
```

**Logs not writing to file?**
- Check `enable_file_logging=True`
- Verify log directory has write permissions
- Check disk space

**Progress bar flickering?**
- This is normal with concurrent output
- Use `show_progress=False` if it's distracting

**Large JSON/CSV files?**
- Only client-level metrics create large files
- Consider recording only summary metrics for very large experiments
- Use compression: `gzip results/*.json`
