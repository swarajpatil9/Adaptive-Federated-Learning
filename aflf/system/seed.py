"""Experiment reproducibility controls."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


class ExperimentSeedManager:
    """Centralized seed setup for deterministic experiment runs."""

    @staticmethod
    def set_seed(seed: int, deterministic: bool = True) -> None:
        """Set seed for Python, NumPy, and PyTorch."""
        os.environ["PYTHONHASHSEED"] = str(seed)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True, warn_only=True)
