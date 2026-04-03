"""Ablation experiment generation and grouping."""

from __future__ import annotations

from typing import Dict

from .experiment_config import ExperimentConfig


class AblationManager:
    """Factory for baseline/AFLF ablations and grouped execution sets."""

    @staticmethod
    def full_ablation_suite(seed: int = 42) -> Dict[str, ExperimentConfig]:
        """
        Required Phase-12 ablations:
        - FedAvg + Selection
        - FedAvg + Privacy
        - FedAvg + Adaptive
        - FedAvg + Communication
        - AFLF full
        """
        return {
            "selection_only": ExperimentConfig(
                name="selection_only",
                description="FedAvg + dynamic selection",
                selection_enabled=True,
                privacy_enabled=False,
                adaptive_lr_enabled=False,
                compression_enabled=False,
                seed=seed,
            ),
            "privacy_only": ExperimentConfig(
                name="privacy_only",
                description="FedAvg + privacy",
                selection_enabled=False,
                privacy_enabled=True,
                adaptive_lr_enabled=False,
                compression_enabled=False,
                seed=seed,
            ),
            "adaptive_only": ExperimentConfig(
                name="adaptive_only",
                description="FedAvg + adaptive learning rate",
                selection_enabled=False,
                privacy_enabled=False,
                adaptive_lr_enabled=True,
                compression_enabled=False,
                seed=seed,
            ),
            "communication_only": ExperimentConfig(
                name="communication_only",
                description="FedAvg + communication compression",
                selection_enabled=False,
                privacy_enabled=False,
                adaptive_lr_enabled=False,
                compression_enabled=True,
                seed=seed,
            ),
            "aflf_full": ExperimentConfig(
                name="aflf_full",
                description="AFLF full stack",
                selection_enabled=True,
                privacy_enabled=True,
                adaptive_lr_enabled=True,
                compression_enabled=True,
                seed=seed,
            ),
        }

    @staticmethod
    def immediate_phase12_subset(seed: int = 42) -> Dict[str, ExperimentConfig]:
        """The 4 runs requested immediately after implementation."""
        configs = AblationManager.full_ablation_suite(seed=seed)
        baseline = ExperimentConfig(
            name="baseline",
            description="FedAvg baseline only",
            selection_enabled=False,
            privacy_enabled=False,
            adaptive_lr_enabled=False,
            compression_enabled=False,
            seed=seed,
        )
        return {
            "baseline": baseline,
            "aflf_full": configs["aflf_full"],
            "selection_only": configs["selection_only"],
            "privacy_only": configs["privacy_only"],
        }
