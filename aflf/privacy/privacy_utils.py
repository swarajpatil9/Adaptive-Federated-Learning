"""Utility helpers for differential privacy processing in federated learning."""

from collections import OrderedDict
from typing import Dict

import torch


def compute_model_update(
    global_weights: OrderedDict[str, torch.Tensor],
    local_weights: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Compute client model update as local minus global weights."""
    if set(global_weights.keys()) != set(local_weights.keys()):
        raise ValueError('Global and local weights must have identical parameter keys')

    update: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, global_tensor in global_weights.items():
        local_tensor = local_weights[name]
        if torch.is_floating_point(local_tensor) and torch.is_floating_point(global_tensor):
            update[name] = local_tensor - global_tensor
        else:
            update[name] = local_tensor.clone()
    return update


def apply_update_to_weights(
    global_weights: OrderedDict[str, torch.Tensor],
    update: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Apply model update to global weights and return protected client weights."""
    if set(global_weights.keys()) != set(update.keys()):
        raise ValueError('Global weights and update must have identical parameter keys')

    protected_weights: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, global_tensor in global_weights.items():
        update_tensor = update[name]
        if torch.is_floating_point(global_tensor) and torch.is_floating_point(update_tensor):
            protected_weights[name] = global_tensor + update_tensor
        else:
            protected_weights[name] = update_tensor.clone()
    return protected_weights


def privacy_budget_placeholder() -> Dict[str, float | None]:
    """Return placeholder privacy-accounting metadata for future extension."""
    return {
        'epsilon': None,
        'delta': None,
        'accountant': 'placeholder',
    }
