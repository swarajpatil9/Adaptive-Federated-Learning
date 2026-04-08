"""Centralized logger setup used by CLI and training runtime."""

from __future__ import annotations

import logging
from pathlib import Path

from .log_config import LoggingConfig
from .logging_utils import sanitize_experiment_name


class SystemLogger:
    """Factory for configuring shared AFLF logging handlers."""

    ROOT_LOGGER_NAME = "aflf"

    @classmethod
    def configure(cls, config: LoggingConfig) -> logging.Logger:
        """Configure root AFLF logger with console and file handlers."""
        logger = logging.getLogger(cls.ROOT_LOGGER_NAME)
        level = getattr(logging, config.level.upper(), logging.INFO)
        logger.setLevel(level)

        logger.handlers = []
        logger.propagate = False

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        slug = sanitize_experiment_name(config.experiment_name)
        file_path = output_dir / f"{slug}_{config.filename}"
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info("Logger initialized. File output: %s", file_path)
        return logger

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Return a child logger under AFLF root."""
        return logging.getLogger(f"{cls.ROOT_LOGGER_NAME}.{name}")
