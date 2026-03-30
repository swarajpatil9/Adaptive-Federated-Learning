"""Secure aggregation preparation utilities.

This module provides a preparation-only layer so future secure aggregation
protocols can be plugged in without changing aggregation math.
"""

from dataclasses import dataclass
from typing import Any, Dict

from ..client.client import TrainingResult


@dataclass
class SecureAggregationPreparationResult:
    """Container with original result and secure-aggregation metadata."""

    training_result: TrainingResult
    metadata: Dict[str, Any]


class SecureAggregationPreparer:
    """Preparation-stage helper for future secure aggregation protocols."""

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)

    def prepare(self, result: TrainingResult) -> SecureAggregationPreparationResult:
        """Attach secure-aggregation metadata without changing model weights."""
        metadata = {
            'secure_aggregation_enabled': self.enabled,
            'masking_applied': False,
            'protocol': 'preparation_only',
            'notes': 'No cryptographic masking applied in baseline Phase 9.',
        }
        return SecureAggregationPreparationResult(training_result=result, metadata=metadata)
