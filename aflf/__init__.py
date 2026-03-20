"""
Adaptive Federated Learning Framework (AFLF)

A research-grade federated learning implementation with:
- Adaptive client selection
- Privacy-preserving aggregation
- Communication optimization
- Modular and extensible design
"""

__version__ = "0.1.0"
__author__ = "Swaraj Patil"

from aflf import (
    aggregation,
    client,
    communication,
    data,
    evaluation,
    models,
    privacy,
    selection,
    server,
    training,
    utils,
)

__all__ = [
    "aggregation",
    "client",
    "communication",
    "data",
    "evaluation",
    "models",
    "privacy",
    "selection",
    "server",
    "training",
    "utils",
]
