"""Differential privacy mechanism wrapper for client weight protection."""

import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Dict

import torch

from .clipping import GradientClipper
from .noise import NoiseAdder
from .privacy_config import PrivacyConfig
from .privacy_utils import (
    apply_update_to_weights,
    compute_model_update,
    privacy_budget_placeholder,
)


@dataclass
class PrivacyProcessingResult:
    """Output from DP client-weight protection."""

    protected_weights: OrderedDict[str, torch.Tensor]
    metadata: Dict[str, object]
    processing_time: float

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary for logging/serialization helpers."""
        payload = asdict(self)
        payload['protected_weights'] = self.protected_weights
        return payload


class PrivacyEngine:
    """Apply clipping + Gaussian noise to client model updates."""

    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.clipper = GradientClipper(config.clip_norm)
        self.noise_adder = NoiseAdder(
            clip_norm=config.clip_norm,
            noise_multiplier=config.noise_multiplier,
        )

    def protect_weights(
        self,
        global_weights: OrderedDict[str, torch.Tensor],
        local_weights: OrderedDict[str, torch.Tensor],
    ) -> PrivacyProcessingResult:
        """Protect local client weights by privatizing model updates."""
        start_time = time.time()

        if not self.config.privacy_enabled:
            metadata: Dict[str, object] = {
                'privacy_enabled': False,
                'clip_applied': False,
                'clip_factor': 1.0,
                'clip_norm': float(self.config.clip_norm),
                'noise_multiplier': float(self.config.noise_multiplier),
                'noise_std': 0.0,
                'noise_applied': False,
                **privacy_budget_placeholder(),
            }
            return PrivacyProcessingResult(
                protected_weights=local_weights,
                metadata=metadata,
                processing_time=time.time() - start_time,
            )

        model_update = compute_model_update(global_weights, local_weights)
        clipped_update, clipping_metadata = self.clipper.clip(model_update)
        noised_update, noise_metadata = self.noise_adder.add_noise(clipped_update)
        protected_weights = apply_update_to_weights(global_weights, noised_update)

        metadata = {
            'privacy_enabled': True,
            **clipping_metadata,
            **noise_metadata,
            **privacy_budget_placeholder(),
        }

        return PrivacyProcessingResult(
            protected_weights=protected_weights,
            metadata=metadata,
            processing_time=time.time() - start_time,
        )
