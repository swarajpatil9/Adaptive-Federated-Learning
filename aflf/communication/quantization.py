"""Precision and quantization routines for model updates."""

from collections import OrderedDict
from typing import Tuple

import torch


class Quantizer:
    """Apply reduced precision or simple 8-bit quantization simulation."""

    def apply_precision(
        self,
        weights: OrderedDict[str, torch.Tensor],
        precision: str,
    ) -> OrderedDict[str, torch.Tensor]:
        """Cast model update tensors to target precision."""
        target = str(precision).lower()
        if target == "float16":
            return OrderedDict((k, v.to(torch.float16)) for k, v in weights.items())
        return OrderedDict((k, v.to(torch.float32)) for k, v in weights.items())

    def quantize_8bit(
        self,
        weights: OrderedDict[str, torch.Tensor],
    ) -> Tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor]]:
        """Quantize each tensor to signed int8 with per-tensor scale."""
        quantized = OrderedDict()
        scales = OrderedDict()

        for name, tensor in weights.items():
            max_abs = float(tensor.abs().max().item())
            scale = max(max_abs / 127.0, 1e-8)
            q = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
            quantized[name] = q
            scales[name] = torch.tensor(scale, dtype=torch.float32)

        return quantized, scales

    def dequantize_8bit(
        self,
        quantized: OrderedDict[str, torch.Tensor],
        scales: OrderedDict[str, torch.Tensor],
    ) -> OrderedDict[str, torch.Tensor]:
        """Recover float32 tensors from signed int8 representation."""
        return OrderedDict(
            (name, quantized[name].to(torch.float32) * scales[name].to(torch.float32))
            for name in quantized.keys()
        )
