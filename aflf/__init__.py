"""
Adaptive Federated Learning Framework (AFLF)

A research-grade federated learning implementation with:
- Adaptive client selection
- Privacy-preserving aggregation
- Communication optimization
- Modular and extensible design

Currently implemented (Phase 2-4):
- Data pipeline: MNIST, CIFAR-10 with IID/Non-IID partitioning
- Models: SimpleCNN, CNN, CNNLarge with FL-optimized interface
- Client training: Local SGD, metrics tracking, weight management

Planned (Phase 5+):
- Server aggregation and coordination
- Communication optimization
- Privacy-preserving mechanisms
"""

__version__ = "0.1.0"
__author__ = "Swaraj Patil"

# Phase 2-4 Implementations (ready to use)
from aflf import client, data, models

# Phase 5+ (placeholders for future phases)
# from aflf import training, aggregation, server, communication, privacy, selection, evaluation, utils

__all__ = [
    # Implemented
    "client",
    "data",
    "models",
    # Planned for future phases
    # "training", "aggregation", "server",
    # "communication", "privacy", "selection", "evaluation", "utils"
]
