# Phase 5 Enhancement: Server Logging System

## Summary

Added comprehensive logging system to Phase 5 (Server Orchestration) following patterns from research repositories like **Flower**, **FedML**, and **PySyft**.

## Files Added

### Core Implementation (3 files)

1. **`aflf/server/logger.py`** (290 lines)
   - `ServerLogger`: Structured console + file logging
   - `ConsoleProgressLogger`: Real-time progress bar (no dependencies)

2. **`aflf/server/metrics_tracker.py`** (430 lines)
   - `MetricsTracker`: Metrics storage and export (JSON/CSV/TensorBoard)
   - `ProgressTracker`: Convergence detection and improvement tracking

3. **`aflf/server/server.py`** (updated)
   - Integrated logging into `FederatedServer`
   - Added `run_training()` method with progress bar
   - Added `export_metrics()` method

### Documentation & Examples (3 files)

4. **`examples/server_logging_demo.py`** (380 lines)
   - 5 complete examples demonstrating all logging features

5. **`docs/SERVER_LOGGING.md`** (350 lines)
   - Complete documentation with usage patterns
   - Research comparison table
   - Best practices and troubleshooting

6. **`tests/server/test_logging.py`** (350 lines)
   - Comprehensive tests for all logging components
   - 25+ test cases

### Module Exports (1 file updated)

7. **`aflf/server/__init__.py`** (updated)
   - Exported: `ServerLogger`, `ConsoleProgressLogger`, `MetricsTracker`, `ProgressTracker`

## Features Implemented

### 1. Structured Logging (`ServerLogger`)

```python
logger = ServerLogger(experiment_name="mnist_fedavg", log_dir="logs")
server = FederatedServer(model=model, server_logger=logger)
```

**Tracks**:
- Server initialization
- Round start/end with timing
- Client success/failure
- Training completion summary

**Output locations**:
- Console (formatted, timestamped)
- File (`logs/{experiment}_{timestamp}.log`)

### 2. Metrics Tracking (`MetricsTracker`)

```python
tracker = MetricsTracker(
    experiment_name="mnist_fedavg",
    output_dir="results",
    enable_tensorboard=True
)
server = FederatedServer(model=model, metrics_tracker=tracker)
server.run_training(num_rounds=50, clients=clients)
server.export_metrics()
```

**Tracks**:
- Round metrics (loss, accuracy, participation, timing)
- Client-level results (per-client performance)
- Summary statistics (averages, min/max, trends)

**Export formats**:
- JSON: Structured data for analysis
- CSV: Spreadsheet-compatible
- TensorBoard: Real-time visualization

### 3. Progress Bar (`ConsoleProgressLogger`)

```python
server.run_training(num_rounds=50, clients=clients, show_progress=True)
```

**Display**:
```
Round 25/50 |█████████████████████████░░░░░░░░░░░░░░░░░░░░░░░| 50% [8/10 clients] Loss: 0.2345 Acc: 0.9123 ETA: 380s
```

**Features**:
- Real-time progress visualization
- Live metrics (loss, accuracy)
- Participation tracking
- ETA calculation
- No external dependencies (pure Python)

### 4. Convergence Detection (`ProgressTracker`)

```python
progress = ProgressTracker(convergence_window=5, convergence_threshold=0.001)

for round_num in range(max_rounds):
    result = server.execute_round(round_num, clients)
    progress.update(loss=result['metrics']['avg_train_loss'],
                   accuracy=result['metrics']['avg_train_accuracy'])

    if progress.has_converged():
        print(f"Converged at round {round_num}!")
        break
```

**Features**:
- Variance-based convergence detection
- Best metrics tracking
- Improvement calculation over windows
- Early stopping support

## Integration Points

### Automatic Logging in `execute_round()`

```python
def execute_round(self, round_num, clients):
    # 1. Log round start
    if self.server_logger:
        self.server_logger.log_round_start(round_num, num_selected)

    # 2. Execute round (existing code)
    round_result = self.orchestrator.execute_round(...)

    # 3. Log round end
    if self.server_logger:
        self.server_logger.log_round_end(round_state, metrics)

    # 4. Track metrics
    if self.metrics_tracker:
        self.metrics_tracker.record_round(round_state, metrics)
        for result in round_result.results:
            self.metrics_tracker.record_client_result(...)

    # 5. Update progress bar
    if self.progress_logger:
        self.progress_logger.update_round(...)
```

### New Server Constructor Parameters

```python
FederatedServer(
    model=model,
    # ... existing parameters ...
    server_logger=ServerLogger(...),        # NEW: Optional logger
    metrics_tracker=MetricsTracker(...),    # NEW: Optional tracker
    enable_progress_bar=False,              # NEW: Progress bar flag
)
```

### New Convenience Methods

```python
# Run training with automatic progress tracking
server.run_training(num_rounds=50, clients=clients, show_progress=True)

# Export all tracked metrics
paths = server.export_metrics(export_json=True, export_csv=True)
```

## Usage Patterns

### Pattern 1: Basic Logging

```python
logger = ServerLogger(experiment_name="experiment_1")
server = FederatedServer(model=model, server_logger=logger)
server.run_training(num_rounds=50, clients=clients)
```

### Pattern 2: Full Research Setup

```python
logger = ServerLogger(experiment_name="exp_1", log_dir="logs")
tracker = MetricsTracker(experiment_name="exp_1", output_dir="results",
                        enable_tensorboard=True)

server = FederatedServer(model=model, server_logger=logger,
                        metrics_tracker=tracker)
server.run_training(num_rounds=50, clients=clients, show_progress=True)
server.export_metrics()
```

### Pattern 3: Convergence-Based Training

```python
progress = ProgressTracker(convergence_window=5)
for round_num in range(max_rounds):
    result = server.execute_round(round_num, clients)
    progress.update(loss=result['metrics']['avg_train_loss'],
                   accuracy=result['metrics']['avg_train_accuracy'])
    if progress.has_converged():
        break
```

## Research Repository Comparison

| Feature | Flower | FedML | **AFLF (Ours)** |
|---------|--------|-------|-----------------|
| Structured logging | ✅ | ✅ | ✅ |
| Progress bar | ✅ (tqdm) | ✅ (tqdm) | ✅ (native) |
| JSON export | ✅ | ✅ | ✅ |
| CSV export | ✅ | ✅ | ✅ |
| TensorBoard | ✅ | ✅ | ✅ |
| Convergence tracking | ❌ | ✅ | ✅ |
| Client-level metrics | ✅ | ✅ | ✅ |
| No external deps | ❌ | ❌ | ✅ |

**Advantages over Flower/FedML**:
- Progress bar with no external dependencies (pure Python)
- Built-in convergence detection
- Cleaner integration (optional, not required)
- More comprehensive client-level tracking

## Files Generated During Training

```
project/
├── logs/
│   └── experiment_1_20240324_143022.log        # Structured log file
├── results/
│   ├── experiment_1_metrics_*.json             # All metrics + summary
│   ├── experiment_1_rounds_*.csv               # Round-level CSV
│   ├── experiment_1_clients_*.csv              # Client-level CSV
│   └── tensorboard/
│       └── experiment_1/
│           └── events.out.tfevents.*           # TensorBoard events
```

## Backward Compatibility

✅ **Fully backward compatible** - all logging features are optional:

```python
# Old code still works (no logging)
server = FederatedServer(model=model)
server.execute_round(round_num=0, clients=clients)

# New code with logging (opt-in)
server = FederatedServer(model=model, server_logger=logger)
server.execute_round(round_num=0, clients=clients)
```

## Testing

**Test coverage**: 25+ tests across 4 test classes

```bash
pytest tests/server/test_logging.py -v
```

**Tests include**:
- ServerLogger initialization and logging methods
- ConsoleProgressLogger progress tracking
- MetricsTracker recording and export (JSON/CSV)
- ProgressTracker convergence detection
- File creation and content verification

## Examples

Run the comprehensive demo:

```bash
python examples/server_logging_demo.py
```

**Demonstrates**:
1. Basic structured logging
2. Progress bar tracking
3. Metrics export
4. Convergence detection
5. Complete logging system (all features)

## Documentation

Full documentation: `docs/SERVER_LOGGING.md`

**Includes**:
- Component overview
- Usage examples
- API reference
- Best practices
- Research comparison
- Troubleshooting

## Summary Statistics

**Lines of code added**: ~1,800
- Implementation: ~720 lines
- Tests: ~350 lines
- Examples: ~380 lines
- Documentation: ~350 lines

**Components added**: 4 classes
- `ServerLogger`
- `ConsoleProgressLogger`
- `MetricsTracker`
- `ProgressTracker`

**Integration**: Seamless
- No breaking changes
- Optional features (opt-in)
- Clean API design

## Next Steps (Optional Enhancements)

Future improvements (Phase 7+):
1. Weights & Biases integration
2. MLflow integration
3. Real-time dashboard (web UI)
4. Distributed logging (multi-server)
5. Custom metric hooks
6. Automatic checkpoint saving based on metrics

---

**Status**: ✅ Complete and ready for use

**Integration**: ✅ Merged into Phase 5 (Server Orchestration)

**Testing**: ✅ Comprehensive test coverage

**Documentation**: ✅ Full documentation provided
