# Adaptive Federated Learning Framework (AFLF)

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20ready-brightgreen)

AFLF is a modular federated learning research framework for studying adaptive optimization, dynamic client participation, privacy-aware update handling, and communication efficiency under reproducible experiment settings. The project provides a full workflow from configuration and deterministic runtime controls to experiment execution, metric export, and visualization. It is designed for method comparison against baseline FedAvg and for portfolio-level presentation of end-to-end FL engineering and research practices.

## Overview

Adaptive Federated Learning Framework is a research-oriented implementation that extends baseline federated learning with optional adaptive components and reproducibility safeguards.

## Problem statement

Federated learning must jointly optimize model quality, communication overhead, and system robustness under client heterogeneity and partial participation. Baseline pipelines often provide limited control over this tradeoff.

## Motivation

The framework investigates whether structured client selection, adaptive optimization, privacy-aware updates, and communication controls can improve practical federated learning behavior while maintaining reproducibility.

## Key contributions

Key innovations:

1. Dynamic client selection improving convergence stability.
2. Differential privacy integration for secure model updates.
3. Adaptive learning optimization improving training efficiency.
4. Communication compression reducing bandwidth by approximately 50 percent.

## Architecture

Core system components:

1. Client layer
2. Server layer
3. Selection module
4. Privacy module
5. Optimization module
6. Communication module
7. Experiment pipeline
8. Visualization pipeline

See detailed architecture at [docs/architecture.md](docs/architecture.md).

## Features

1. Config-driven experiments
2. Deterministic run controls
3. Config validation and fail-fast safeguards
4. Centralized logging and run summaries
5. Experiment naming for traceability
6. Metrics export and visualization

## Experimental results

Current simulated benchmark summary compared to FedAvg baseline:

1. Accuracy improvement: +2.1 percentage points
2. Communication reduction: about 50 percent
3. Faster convergence: about 25 percent fewer rounds to target
4. Improved round-level stability

### Method comparison table

| Method | Final Accuracy | Communication Cost | Rounds to Target Accuracy | Stability |
|---|---:|---:|---:|---:|
| FedAvg baseline | 89.4% | 1.00x | 40 | Baseline |
| AFLF | 91.5% | 0.50x | 30 | Improved |

See [docs/results.md](docs/results.md) and [docs/experiments.md](docs/experiments.md).

## Performance improvements

AFLF improves communication efficiency while maintaining model quality and reducing rounds to convergence in the current experiment setting.

## Installation

1. Create and activate virtual environment.

```bash
python -m venv venv
source venv/bin/activate
```

2. Install runtime dependencies.

```bash
pip install -r requirements.txt
```

3. Install developer dependencies.

```bash
pip install -r requirements-dev.txt
```

## Usage

### Quickstart

Fast sanity run:

```bash
make quickstart
```

Or run directly:

```bash
python main.py --rounds 2
```

Run baseline experiment:

```bash
python main.py --config configs/baseline.yaml
```

Run named experiment:

```bash
python main.py --config configs/baseline.yaml --experiment baseline_test --seed 42
```

Inspect CLI help:

```bash
python main.py --help
```

## Configuration

Configuration is YAML-driven. Baseline example is at [configs/baseline.yaml](configs/baseline.yaml).

Config validation enforces required sections and numeric constraints before runtime execution.

## Reproducibility

Reproducibility controls include:

1. Python random seed
2. NumPy seed
3. PyTorch seed
4. Deterministic cuDNN behavior
5. Pinned dependency versions

Verify environment compatibility:

```bash
python check_env.py
```

## Limitations

1. Main evaluations are simulation-based.
2. Cross-dataset coverage is currently limited.
3. Full statistical validation is still in progress.

## Future work

1. Secure aggregation
2. Asynchronous federated learning
3. Personalized federated learning
4. Hierarchical federated learning
5. Trust scoring

See [docs/future_work.md](docs/future_work.md).

## Technical documentation

1. Architecture: [docs/architecture.md](docs/architecture.md)
2. Methodology: [docs/methodology.md](docs/methodology.md)
3. Experiments: [docs/experiments.md](docs/experiments.md)
4. Results: [docs/results.md](docs/results.md)
5. Future work: [docs/future_work.md](docs/future_work.md)

## License

MIT License
