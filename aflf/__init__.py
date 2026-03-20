"""
Adaptive Federated Learning Framework (AFLF)

A research-grade federated learning implementation with:
- Adaptive client selection
- Privacy-preserving aggregation
- Communication optimization
- Modular and extensible design

Currently implemented (Phase 2-3):
- Data pipeline: MNIST, CIFAR-10 with IID/Non-IID partitioning
- Models: SimpleCNN, CNN, CNNLarge with FL-optimized interface

Planned (Phase 4+):
- Client training and local optimization
- Server aggregation and coordination
- Communication optimization
- Privacy-preserving mechanisms
"""

__version__ = "0.1.0"
__author__ = "Swaraj Patil"

# Phase 2-3 Implementations (ready to use)
from aflf import data, models

# Phase 4+ (placeholders for future phases)
# from aflf import client, training, aggregation, server, communication, privacy, selection, evaluation, utils

__all__ = [
    # Implemented
    "data",
    "models",
    # Planned for future phases
    # "client", "training", "aggregation", "server",
    # "communication", "privacy", "selection", "evaluation", "utils"
]
