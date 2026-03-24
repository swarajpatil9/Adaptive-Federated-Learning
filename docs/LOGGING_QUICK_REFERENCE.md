# Server Logging Quick Reference

## Quick Start

### 1. Basic Setup (2 lines)

```python
from aflf.server import FederatedServer, ServerLogger

logger = ServerLogger(experiment_name="my_experiment")
server = FederatedServer(model=model, server_logger=logger)
```

### 2. Full Setup (4 lines)

```python
from aflf.server import FederatedServer, ServerLogger, MetricsTracker

logger = ServerLogger(experiment_name="my_exp", log_dir="logs")
tracker = MetricsTracker(experiment_name="my_exp", output_dir="results")
server = FederatedServer(model=model, server_logger=logger, metrics_tracker=tracker)
server.run_training(num_rounds=50, clients=clients, show_progress=True)
server.export_metrics()
```

## Common Patterns

### Pattern 1: Logging Only

```python
logger = ServerLogger(experiment_name="exp_1")
server = FederatedServer(model=model, server_logger=logger)

for round_num in range(50):
    server.execute_round(round_num, clients)
```

**Output**: Console + file logs in `logs/exp_1_*.log`

### Pattern 2: Metrics Export Only

```python
tracker = MetricsTracker(experiment_name="exp_1", output_dir="results")
server = FederatedServer(model=model, metrics_tracker=tracker)

for round_num in range(50):
    server.execute_round(round_num, clients)

server.export_metrics()  # Creates JSON + CSV
```

**Output**: JSON/CSV files in `results/`

### Pattern 3: Progress Bar Only

```python
server = FederatedServer(model=model)
server.run_training(num_rounds=50, clients=clients, show_progress=True)
```

**Output**: Live progress bar with metrics

### Pattern 4: Everything

```python
logger = ServerLogger(experiment_name="full")
tracker = MetricsTracker(experiment_name="full", enable_tensorboard=True)
server = FederatedServer(model=model, server_logger=logger, metrics_tracker=tracker)

server.run_training(num_rounds=50, clients=clients, show_progress=True)
server.export_metrics()
```

**Output**: Logs + JSON/CSV + TensorBoard + Progress bar

### Pattern 5: Early Stopping

```python
from aflf.server import ProgressTracker

progress = ProgressTracker(convergence_window=5, convergence_threshold=0.001)

for round_num in range(100):
    result = server.execute_round(round_num, clients)
    progress.update(
        loss=result['metrics']['avg_train_loss'],
        accuracy=result['metrics']['avg_train_accuracy']
    )
    if round_num >= 10 and progress.has_converged():
        print(f"Converged at round {round_num}")
        break
```

**Output**: Stops when loss converges

## Common Tasks

### View TensorBoard

```bash
# After training with enable_tensorboard=True
tensorboard --logdir results/tensorboard
# Open http://localhost:6006
```

### Load Exported Metrics (Python)

```python
import json

with open('results/exp_1_metrics_*.json') as f:
    data = json.load(f)

print(f"Total rounds: {data['summary']['num_rounds']}")
print(f"Final loss: {data['summary']['final_train_loss']:.4f}")

# Plot loss curve
losses = [r['avg_train_loss'] for r in data['round_metrics']]
import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()
```

### Load Exported Metrics (Pandas)

```python
import pandas as pd

# Round-level metrics
rounds_df = pd.read_csv('results/exp_1_rounds_*.csv')
print(rounds_df.describe())

# Client-level metrics
clients_df = pd.read_csv('results/exp_1_clients_*.csv')
print(clients_df.groupby('client_id')['train_loss'].mean())
```

### Change Log Level

```python
import logging

# DEBUG: Show everything
logger = ServerLogger(experiment_name="debug", console_level=logging.DEBUG)

# WARNING: Only warnings and errors
logger = ServerLogger(experiment_name="quiet", console_level=logging.WARNING)
```

## API Reference

### ServerLogger

```python
ServerLogger(
    experiment_name: str = "federated_learning",
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    enable_file_logging: bool = True
)
```

### MetricsTracker

```python
MetricsTracker(
    experiment_name: str = "federated_learning",
    output_dir: str = "results",
    enable_tensorboard: bool = False,
    tensorboard_dir: Optional[str] = None
)
```

### ProgressTracker

```python
ProgressTracker(
    convergence_window: int = 5,
    convergence_threshold: float = 0.001
)
```

### FederatedServer (updated)

```python
FederatedServer(
    model: nn.Module,
    selection_strategy: Optional[SelectionStrategy] = None,
    aggregation_strategy: Optional[AggregationStrategy] = None,
    num_clients_per_round: int = 10,
    device: str = "cpu",
    server_logger: Optional[ServerLogger] = None,         # NEW
    metrics_tracker: Optional[MetricsTracker] = None,     # NEW
    enable_progress_bar: bool = False                     # NEW
)
```

## Examples

Run examples:

```bash
# Full logging system demo
python examples/server_logging_demo.py

# Original Phase 5 demo (no logging)
python examples/phase5_server_orchestration.py
```

## Troubleshooting

**Q: TensorBoard not working?**
```bash
pip install tensorboard
```

**Q: Where are my log files?**
```bash
ls logs/  # Look for experiment_name_timestamp.log
```

**Q: Progress bar not showing?**
```python
# Use run_training() with show_progress=True
server.run_training(num_rounds=50, clients=clients, show_progress=True)
```

**Q: How to disable logging?**
```python
# Simply don't pass logger/tracker
server = FederatedServer(model=model)  # No logging
```

## Tips

✅ **DO**:
- Always export metrics: `server.export_metrics()`
- Use meaningful experiment names
- Enable file logging for long experiments
- Use TensorBoard for visualization

❌ **DON'T**:
- Don't log to console in production (use file logging only)
- Don't forget to call `tracker.close()` (cleanup)
- Don't ignore convergence tracking (can save time)

## Full Documentation

See `docs/SERVER_LOGGING.md` for complete documentation.
