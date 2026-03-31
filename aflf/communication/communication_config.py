"""Configuration primitives for communication-efficiency controls."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class CommunicationConfig:
    """Config for client update compression and precision control."""

    compression_enabled: bool = False
    precision: str = "float32"
    quantization_enabled: bool = False
    sparsification_enabled: bool = False
    sparsity_ratio: float = 0.0

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "CommunicationConfig":
        """Build config from raw communication block."""
        source = raw or {}
        return cls(
            compression_enabled=bool(source.get("compression_enabled", False)),
            precision=str(source.get("precision", "float32")).lower(),
            quantization_enabled=bool(source.get("quantization_enabled", False)),
            sparsification_enabled=bool(source.get("sparsification_enabled", False)),
            sparsity_ratio=float(source.get("sparsity_ratio", 0.0)),
        )
