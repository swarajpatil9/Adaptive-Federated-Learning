"""Privacy configuration models for client-side DP processing."""

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class PrivacyConfig:
    """Configuration for optional client-side differential privacy."""

    privacy_enabled: bool = False
    clip_norm: float = 1.0
    noise_multiplier: float = 0.05
    secure_aggregation_enabled: bool = False
    accountant_type: str = 'placeholder'

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'PrivacyConfig':
        """Create PrivacyConfig from training config dictionary."""
        if config is None:
            return cls()

        privacy_block = config.get('privacy', config)
        return cls(
            privacy_enabled=bool(privacy_block.get('privacy_enabled', False)),
            clip_norm=float(privacy_block.get('clip_norm', 1.0)),
            noise_multiplier=float(privacy_block.get('noise_multiplier', 0.05)),
            secure_aggregation_enabled=bool(
                privacy_block.get('secure_aggregation_enabled', False)
            ),
            accountant_type=str(privacy_block.get('accountant_type', 'placeholder')),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for logs/config propagation."""
        return asdict(self)
