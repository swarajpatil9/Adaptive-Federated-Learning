"""Logging configuration objects."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoggingConfig:
    """Configuration for system-wide logging handlers."""

    experiment_name: str
    output_dir: Path = Path("results/logs")
    level: str = "INFO"
    filename: str = "training.log"
