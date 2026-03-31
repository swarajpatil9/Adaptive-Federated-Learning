# Adaptive Federated Learning Framework (AFLF)

A research-grade implementation of federated learning with adaptive optimization, privacy preservation, and communication efficiency.

## Overview

This framework implements a complete federated learning system with:
- **Baseline FL**: Standard FedAvg with multi-client simulation
- **Adaptive components**: Dynamic client selection, adaptive learning rates
- **Privacy**: Differential privacy via Opacus
- **Communication optimization**: Gradient compression and scheduling
- **Research-grade evaluation**: Comprehensive metrics and ablation studies

## Architecture

```
aflf/
├── client/           # Client-side training logic
├── server/           # Server-side aggregation
├── aggregation/      # Aggregation strategies (FedAvg, weighted, etc.)
├── selection/        # Client selection mechanisms
├── optimization/     # Adaptive learning-rate optimization
├── privacy/          # Privacy-preserving techniques
├── communication/    # Communication optimization
├── models/           # Neural network architectures
├── data/             # Dataset loaders and partitioning
├── training/         # Training loops and optimizers
├── evaluation/       # Metrics and evaluation utilities
└── utils/            # Shared utilities
```

## Installation

### Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install dependencies
```bash
pip install -e .
```

### Install development dependencies
```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
python main.py --config configs/baseline.yaml
```

## Project Status

**Phase 1**: ✓ Environment setup and project skeleton
**Phase 2**: Dataset pipeline and client data simulation
**Phase 3**: Base model architectures
**Phase 4**: Client trainer implementation
**Phase 5**: Server and aggregation
**Phase 6**: Federated learning loop
**Phase 7**: Metrics and evaluation
**Phase 8**: Dynamic client selection
**Phase 9**: Privacy mechanisms
**Phase 10**: Adaptive optimization
**Phase 11**: Communication efficiency
**Phase 12**: Experiments and baselines
**Phase 13**: Visualization
**Phase 14**: Testing and hardening
**Phase 15**: Research polish

## Development

### Code quality
```bash
# Format code
black aflf/
isort aflf/

# Lint
flake8 aflf/

# Type checking
mypy aflf/

# Run tests
pytest
```

## Research Features

- **Reproducibility**: Deterministic runs with seed control
- **Experiment tracking**: TensorBoard integration
- **Config-driven**: YAML-based experiment configuration
- **Ablation studies**: Modular design for component analysis
- **Privacy-utility tradeoff**: Configurable DP parameters

## Requirements

- Python >= 3.9
- PyTorch >= 2.1.0
- See `requirements.txt` for full dependencies

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{aflf2026,
  title={Adaptive Federated Learning Framework},
  author={Patil, Swaraj},
  year={2026},
  url={https://github.com/swarajpatil9/adaptive-federated-learning}
}
```

## License

MIT License

## Author

Swaraj Patil - ML Engineer & Federated Learning Researcher
# Adaptive-Federated-Learning
