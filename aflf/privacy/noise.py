"""Noise injection utilities for client-side differential privacy."""

from collections import OrderedDict
from typing import Dict

import torch


class NoiseAdder:
    """Add Gaussian noise to clipped client model updates."""

    def __init__(self, clip_norm: float, noise_multiplier: float):
        if clip_norm <= 0:
            raise ValueError('clip_norm must be positive')
        if noise_multiplier < 0:
            raise ValueError('noise_multiplier must be non-negative')

        self.clip_norm = float(clip_norm)
        self.noise_multiplier = float(noise_multiplier)

    @property
    def noise_std(self) -> float:
        """Gaussian standard deviation used for each floating tensor."""
        return self.clip_norm * self.noise_multiplier

    def add_noise(
        self,
        update: OrderedDict[str, torch.Tensor],
    ) -> tuple[OrderedDict[str, torch.Tensor], Dict[str, float]]:
        """Add i.i.d. Gaussian noise to floating-point tensors in update."""
        noised_update: OrderedDict[str, torch.Tensor] = OrderedDict()
        std = self.noise_std

        for name, tensor in update.items():
            if torch.is_floating_point(tensor) and std > 0:
                noise = torch.normal(
                    mean=0.0,
                    std=std,
                    size=tensor.shape,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                noised_update[name] = tensor + noise
            else:
                noised_update[name] = tensor.clone()

        metadata = {
            'noise_multiplier': float(self.noise_multiplier),
            'noise_std': float(std),
            'noise_applied': bool(std > 0),
        }
        return noised_update, metadata
