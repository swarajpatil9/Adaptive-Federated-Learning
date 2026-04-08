"""Utility helpers for the AFLF research dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

RESULTS_ROOT = Path("results")
CONFIGS_ROOT = Path("configs")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def to_display_name(run_key: str) -> str:
    """Create a compact display label for run identifiers."""
    label = run_key.replace("_phase", " | phase ")
    return label.replace("_", " ")


def infer_method_from_name(name: str) -> str:
    lower = name.lower()
    if "fedavg" in lower:
        return "FedAvg"
    if "baseline" in lower:
        return "FedAvg"
    if "aflf" in lower:
        return "AFLF"
    return "Other"


def safe_read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_read_yaml(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def find_results_dirs() -> list[Path]:
    """Return available result roots, preferring canonical results/ first."""
    roots: list[Path] = []
    primary = PROJECT_ROOT / "results"
    if primary.exists():
        roots.append(primary)

    for candidate in PROJECT_ROOT.glob("results/**"):
        if candidate.is_dir() and candidate.name in {"experiments", "system_check"}:
            if candidate not in roots:
                roots.append(candidate)
    return roots


def format_float(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{val:.{digits}f}"


def format_percent(value: Any, digits: int = 2) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if val <= 1.0:
        val *= 100.0
    return f"{val:.{digits}f}%"


def format_seconds(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}s"
    except (TypeError, ValueError):
        return "N/A"


def bytes_to_mb(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value) / (1024 * 1024):.{digits}f} MB"
    except (TypeError, ValueError):
        return "N/A"
