"""Top-K magnitude sparsification routines for model updates."""

from collections import OrderedDict
from typing import Dict, Tuple

import torch


class SparseUpdateHandler:
    """Apply and reverse top-k sparsification metadata."""

    def sparsify(
        self,
        weights: OrderedDict[str, torch.Tensor],
        sparsity_ratio: float,
    ) -> Tuple[OrderedDict[str, torch.Tensor], Dict[str, float]]:
        """Keep only top magnitude entries and zero out the rest."""
        ratio = min(max(float(sparsity_ratio), 0.0), 0.99)
        if ratio <= 0.0:
            return weights, {"effective_sparsity_ratio": 0.0}

        sparse = OrderedDict()
        nonzero_total = 0
        value_total = 0

        for name, tensor in weights.items():
            flat = tensor.abs().flatten()
            k = max(1, int((1.0 - ratio) * flat.numel()))
            if k >= flat.numel():
                mask = torch.ones_like(tensor, dtype=tensor.dtype)
            else:
                threshold = torch.topk(flat, k, largest=True).values.min()
                mask = (tensor.abs() >= threshold).to(tensor.dtype)
            sparse_tensor = tensor * mask
            sparse[name] = sparse_tensor
            nonzero_total += int((sparse_tensor != 0).sum().item())
            value_total += int(sparse_tensor.numel())

        effective = 1.0 - (float(nonzero_total) / float(value_total)) if value_total else 0.0
        return sparse, {"effective_sparsity_ratio": float(effective)}

    def densify(
        self,
        sparse_weights: OrderedDict[str, torch.Tensor],
    ) -> OrderedDict[str, torch.Tensor]:
        """Return sparse tensors unchanged as dense tensors with zeros."""
        return OrderedDict((name, tensor) for name, tensor in sparse_weights.items())
