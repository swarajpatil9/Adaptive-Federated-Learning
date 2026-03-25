"""
Adaptive Federated Learning Framework (AFLF)

A research-grade federated learning implementation with:
- Adaptive client selection
- Privacy-preserving aggregation
- Communication optimization
- Modular and extensible design

Currently implemented (Phase 2-6):
- Data pipeline: MNIST, CIFAR-10 with IID/Non-IID partitioning
- Models: SimpleCNN, CNN, CNNLarge with FL-optimized interface
- Client training: Local SGD, metrics tracking, weight management
- Server orchestration and round management
- FedAvg aggregation baseline
- Federated training loop with global evaluation

Planned (Phase 5+):
- Server aggregation and coordination
- Communication optimization
- Privacy-preserving mechanisms
"""

__version__ = "0.1.0"
__author__ = "Swaraj Patil"

__all__ = [
    "client",
    "data",
    "models",
    "training",
    "aggregation",
    "server",
    "selection",
    "evaluation",
]
