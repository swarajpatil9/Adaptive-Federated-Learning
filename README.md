# Adaptive Federated Learning Framework (AFLF)

AFLF is a research-focused federated learning framework designed for reproducible experimentation, adaptive optimization studies, and publication-ready analysis pipelines.

## Project Overview

AFLF provides a complete FL workflow:
- Multi-client federated training
- Config-driven experiment execution
- Metrics export and analysis
- Visualization pipelines
- Privacy and communication research modules

PHASE 14 hardening adds stronger safeguards for deterministic runs, config sanity checks, centralized logging, and environment validation.

## Architecture

```text
aflf/
├── aggregation/      # FedAvg and aggregation utilities
├── client/           # Local training clients
├── communication/    # Compression and communication controls
├── config/           # ConfigValidator + defaults + config utils
├── data/             # Dataset loading and partitioning
├── evaluation/       # Round/global metrics and summaries
├── logging/          # SystemLogger + log configuration
├── models/           # Model definitions and factories
├── optimization/     # Adaptive optimization controllers
├── privacy/          # Differential privacy and related modules
├── selection/        # Client selection policies
├── server/           # Server/orchestration runtime
├── system/           # Seed manager + env/dependency checks
└── training/         # Training loop orchestration
```

## Features

- Deterministic execution controls
- Early config validation with clear errors
- Centralized file + console logging
- Graceful exception handling in training runtime
- Environment and dependency verification script
- Configurable experiment naming for run tracking

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Optional developer dependencies:
```bash
pip install -r requirements-dev.txt
```

## Usage

Run baseline training:
```bash
python main.py --config configs/baseline.yaml
```

Run with explicit experiment naming and seed override:
```bash
python main.py --config configs/baseline.yaml --experiment baseline_test --seed 42
```

Inspect CLI help:
```bash
python main.py --help
```

Verify local environment:
```bash
python check_env.py
```

## Experiments

Experiment outputs are written under:
- `results/metrics/` for evaluation exports
- `results/logs/` for run logs

Use `--experiment` to separate runs in logging and downstream analysis.

## Results

The framework exports round-level and summary-level metrics suitable for:
- FL method comparison
- convergence analysis
- communication-efficiency studies
- reproducibility checks across repeated seeds

## Reproducibility

AFLF enforces deterministic settings for:
- Python `random`
- NumPy RNG
- PyTorch CPU/GPU RNG
- cuDNN deterministic mode

To maximize reproducibility:
- keep dependency versions pinned
- use identical config files
- set explicit `--seed`
- run `python check_env.py` before experiments

## Coding Rules (Repository Hardening)

- Keep training and model logic unchanged unless explicitly required
- Add safeguards around I/O, config loading, and orchestration boundaries
- Prefer config-driven values over hardcoded constants
- Use logger APIs instead of ad-hoc prints in runtime paths

## Research-Oriented Next Improvements

- Docker image and compose setup for portable execution
- CI for lint, tests, and smoke-run validation
- automated experiment matrix runs
- config schema versioning and migration checks
- expanded ML reproducibility protocol documentation

## License

MIT License
