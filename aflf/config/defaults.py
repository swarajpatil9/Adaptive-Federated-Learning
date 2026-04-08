"""Default configuration values for AFLF runtime hardening."""

from typing import Any, Dict


def get_config_defaults() -> Dict[str, Any]:
    """Return conservative defaults used when optional config fields are missing."""
    return {
        "seed": 42,
        "federated": {
            "num_rounds": 1,
            "clients_per_round": 1,
            "device": "cpu",
        },
        "training": {
            "local_epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "sgd",
            "momentum": 0.0,
            "weight_decay": 0.0,
            "criterion": "cross_entropy",
            "verbose": False,
        },
        "evaluation": {
            "frequency": 1,
        },
    }
