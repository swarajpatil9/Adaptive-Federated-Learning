"""Gradient/update clipping utilities for differential privacy in FL."""

from collections import OrderedDict
from typing import Dict

import torch


class GradientClipper:
    """Clip client model updates by global L2 norm."""

    def __init__(self, clip_norm: float):
        if clip_norm <= 0:
            raise ValueError('clip_norm must be positive')
        self.clip_norm = float(clip_norm)

    def compute_l2_norm(self, update: OrderedDict[str, torch.Tensor]) -> float:
        """Compute global L2 norm over all floating-point tensors in update."""
        squared_norm = 0.0
        for tensor in update.values():
            if torch.is_floating_point(tensor):
                squared_norm += float(torch.sum(tensor.detach().float() ** 2).item())
        return squared_norm ** 0.5

    def clip(
        self,
        update: OrderedDict[str, torch.Tensor],
    ) -> tuple[OrderedDict[str, torch.Tensor], Dict[str, float]]:
        """Clip update using g <- g * min(1, C / ||g||)."""
        update_norm = self.compute_l2_norm(update)
        clip_factor = min(1.0, self.clip_norm / (update_norm + 1e-12))

        clipped_update: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, tensor in update.items():
            if torch.is_floating_point(tensor):
                clipped_update[name] = tensor * clip_factor
            else:
                clipped_update[name] = tensor.clone()

        metadata = {
            'update_norm_before_clip': float(update_norm),
            'clip_factor': float(clip_factor),
            'clip_applied': bool(clip_factor < 1.0),
            'clip_norm': float(self.clip_norm),
        }
        return clipped_update, metadata
