"""Utilities for communication payload accounting and conversion."""

from collections import OrderedDict
from typing import Dict

import torch


DTYPE_TO_BYTES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.int8: 1,
    torch.uint8: 1,
}


def tensor_nbytes(tensor: torch.Tensor) -> int:
    """Estimate byte size of a tensor with fallback handling."""
    bytes_per_element = int(DTYPE_TO_BYTES.get(tensor.dtype, tensor.element_size()))
    return int(tensor.numel() * bytes_per_element)


def state_dict_nbytes(weights: OrderedDict[str, torch.Tensor]) -> int:
    """Estimate payload size for an ordered state dict."""
    return int(sum(tensor_nbytes(param) for param in weights.values()))


def to_float32_state_dict(
    weights: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Convert all tensors to float32 for stable aggregation."""
    return OrderedDict((name, tensor.to(torch.float32)) for name, tensor in weights.items())


def to_float16_state_dict(
    weights: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Convert all tensors to float16 for reduced precision uplink."""
    return OrderedDict((name, tensor.to(torch.float16)) for name, tensor in weights.items())


def mean(values: Dict[str, float]) -> float:
    """Compute mean over mapping values with empty safety."""
    if not values:
        return 0.0
    return float(sum(values.values()) / len(values))
