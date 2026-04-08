"""Utilities for logger-safe formatting."""

from __future__ import annotations

import re


def sanitize_experiment_name(name: str) -> str:
    """Convert an arbitrary experiment name into a filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", name.strip())
    return slug or "experiment"
