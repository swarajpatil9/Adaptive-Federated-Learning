"""Configuration models for reproducible federated experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from aflf.training import load_yaml_config


@dataclass
class ExperimentConfig:
    """Config-driven experiment definition with AFLF feature toggles."""

    name: str
    description: str
    base_config_path: str = "configs/baseline.yaml"
    selection_enabled: bool = False
    privacy_enabled: bool = False
    adaptive_lr_enabled: bool = False
    compression_enabled: bool = False
    seed: int = 42
    num_runs: int = 1
    tags: Dict[str, str] = field(default_factory=dict)
    output_subdir: str = "results/experiments"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a dictionary."""
        return asdict(self)

    def load_base_runtime_config(self) -> Dict[str, Any]:
        """Load the baseline YAML used as experiment template."""
        return load_yaml_config(self.base_config_path)

    def build_runtime_config(self) -> Dict[str, Any]:
        """
        Build a training config dictionary for FederatedTrainer.

        This function only mutates a copied config dictionary and does not
        modify the existing training system.
        """
        config = self.load_base_runtime_config()

        config["seed"] = int(self.seed)

        selection_block = config.setdefault("selection", {})
        if self.selection_enabled:
            if str(selection_block.get("strategy", "random")).lower() == "random":
                selection_block["strategy"] = "dynamic"
        else:
            selection_block["strategy"] = "random"

        privacy_block = config.setdefault("privacy", {})
        privacy_block["privacy_enabled"] = bool(self.privacy_enabled)

        communication_block = config.setdefault("communication", {})
        communication_block["compression_enabled"] = bool(self.compression_enabled)

        optimization_block = config.setdefault("optimization", {})
        adaptive_block = optimization_block.setdefault("adaptive_lr", {})
        adaptive_block["enabled"] = bool(self.adaptive_lr_enabled)

        return config

    def output_dir(self) -> Path:
        """Directory where experiment artifacts will be written."""
        return Path(self.output_subdir)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Build config from plain dictionary data."""
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", data["name"])),
            base_config_path=str(data.get("base_config_path", "configs/baseline.yaml")),
            selection_enabled=bool(data.get("selection_enabled", False)),
            privacy_enabled=bool(data.get("privacy_enabled", False)),
            adaptive_lr_enabled=bool(data.get("adaptive_lr_enabled", False)),
            compression_enabled=bool(data.get("compression_enabled", False)),
            seed=int(data.get("seed", 42)),
            num_runs=int(data.get("num_runs", 1)),
            tags=dict(data.get("tags", {})),
            output_subdir=str(data.get("output_subdir", "results/experiments")),
        )


def load_experiment_config_from_file(path: str) -> ExperimentConfig:
    """Load one experiment definition from YAML file."""
    payload = load_yaml_config(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in experiment config file: {path}")

    if "experiment" in payload:
        body = payload["experiment"]
    else:
        body = payload

    if not isinstance(body, dict):
        raise ValueError(f"Expected 'experiment' to be a mapping in: {path}")

    config = ExperimentConfig.from_dict(body)

    if not Path(config.base_config_path).exists():
        raise FileNotFoundError(
            f"Base config path does not exist: {config.base_config_path}"
        )

    return config


def default_phase12_core_configs(seed: int = 42) -> Dict[str, ExperimentConfig]:
    """Canonical phase-12 core experiments: baseline and AFLF full."""
    baseline = ExperimentConfig(
        name="baseline",
        description="FedAvg baseline only",
        selection_enabled=False,
        privacy_enabled=False,
        adaptive_lr_enabled=False,
        compression_enabled=False,
        seed=seed,
    )

    aflf_full = ExperimentConfig(
        name="aflf_full",
        description="AFLF full stack: selection + privacy + adaptive + communication",
        selection_enabled=True,
        privacy_enabled=True,
        adaptive_lr_enabled=True,
        compression_enabled=True,
        seed=seed,
    )

    return {
        baseline.name: baseline,
        aflf_full.name: aflf_full,
    }


def default_phase12_requested_four(seed: int = 42) -> Dict[str, ExperimentConfig]:
    """Required immediate runs requested for phase 12 completion."""
    configs = default_phase12_core_configs(seed=seed)
    configs["selection_only"] = ExperimentConfig(
        name="selection_only",
        description="FedAvg + dynamic selection",
        selection_enabled=True,
        privacy_enabled=False,
        adaptive_lr_enabled=False,
        compression_enabled=False,
        seed=seed,
    )
    configs["privacy_only"] = ExperimentConfig(
        name="privacy_only",
        description="FedAvg + privacy",
        selection_enabled=False,
        privacy_enabled=True,
        adaptive_lr_enabled=False,
        compression_enabled=False,
        seed=seed,
    )
    return configs
