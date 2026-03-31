"""Model update compression and communication accounting."""

from collections import OrderedDict
from typing import Any, Dict, Tuple

import torch

from .communication_config import CommunicationConfig
from .communication_utils import state_dict_nbytes, to_float32_state_dict
from .quantization import Quantizer
from .sparsification import SparseUpdateHandler


class CommunicationTracker:
    """Track per-update communication payload statistics."""

    def summarize(
        self,
        original_bytes: int,
        compressed_bytes: int,
    ) -> Dict[str, float]:
        """Produce compression reduction metrics for one payload."""
        saved = max(int(original_bytes) - int(compressed_bytes), 0)
        reduction = (saved / float(original_bytes) * 100.0) if original_bytes > 0 else 0.0
        return {
            "original_bytes": float(original_bytes),
            "compressed_bytes": float(compressed_bytes),
            "saved_bytes": float(saved),
            "reduction_percentage": float(reduction),
        }


class ModelCompressor:
    """Compress and decompress client updates based on communication config."""

    def __init__(self, config: CommunicationConfig):
        self.config = config
        self.quantizer = Quantizer()
        self.sparse_handler = SparseUpdateHandler()
        self.tracker = CommunicationTracker()

    def compress_model_update(
        self,
        weights: OrderedDict[str, torch.Tensor],
    ) -> Tuple[OrderedDict[str, torch.Tensor], Dict[str, Any]]:
        """Apply optional precision reduction, quantization, and sparsification."""
        original_bytes = state_dict_nbytes(weights)
        metadata: Dict[str, Any] = {
            "compression_enabled": bool(self.config.compression_enabled),
            "precision": self.config.precision,
            "quantization_enabled": bool(self.config.quantization_enabled),
            "sparsification_enabled": bool(self.config.sparsification_enabled),
            "sparsity_ratio": float(self.config.sparsity_ratio),
        }

        if not self.config.compression_enabled:
            payload_stats = self.tracker.summarize(original_bytes, original_bytes)
            metadata.update(payload_stats)
            metadata["quantization_scales"] = None
            return weights, metadata

        compressed = self.quantizer.apply_precision(weights, self.config.precision)

        if self.config.quantization_enabled:
            compressed, scales = self.quantizer.quantize_8bit(compressed)
            metadata["quantization_scales"] = scales
        else:
            metadata["quantization_scales"] = None

        if self.config.sparsification_enabled:
            compressed, sparse_meta = self.sparse_handler.sparsify(
                compressed,
                self.config.sparsity_ratio,
            )
            metadata.update(sparse_meta)
        else:
            metadata["effective_sparsity_ratio"] = 0.0

        compressed_bytes = state_dict_nbytes(compressed)
        payload_stats = self.tracker.summarize(original_bytes, compressed_bytes)
        metadata.update(payload_stats)
        return compressed, metadata

    def decompress_model_update(
        self,
        weights: OrderedDict[str, torch.Tensor],
        metadata: Dict[str, Any],
    ) -> OrderedDict[str, torch.Tensor]:
        """Convert compressed payload back to float32 tensors for aggregation."""
        if not metadata or not metadata.get("compression_enabled", False):
            return to_float32_state_dict(weights)

        output = weights

        if metadata.get("quantization_enabled", False):
            scales = metadata.get("quantization_scales", None)
            if scales is None:
                raise ValueError("Missing quantization scales for int8 update")
            output = self.quantizer.dequantize_8bit(output, scales)

        if metadata.get("sparsification_enabled", False):
            output = self.sparse_handler.densify(output)

        return to_float32_state_dict(output)
