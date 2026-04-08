"""System utility helpers for hardening scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_directories(paths: Iterable[Path]) -> None:
    """Create all directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
