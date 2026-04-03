"""Utility helpers for experiment reproducibility and metrics extraction."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


def ensure_dir(path: Path) -> Path:
    """Create directory if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_global_reproducibility(seed: int) -> None:
    """Set all relevant RNG seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Keep deterministic behavior when possible.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def dump_json(data: Dict[str, Any], output_path: Path) -> Path:
    """Write JSON artifact with stable formatting."""
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    return output_path


def _round_to(value: float, ndigits: int = 6) -> float:
    return float(round(float(value), ndigits))


def _infer_convergence_round(history: List[Dict[str, Any]], threshold_ratio: float = 0.98) -> int:
    """
    Estimate convergence round as first round reaching threshold_ratio of final accuracy.
    """
    if not history:
        return 0

    final_accuracy = float(history[-1].get("global_accuracy", 0.0))
    if final_accuracy <= 0.0:
        return len(history)

    threshold = final_accuracy * float(threshold_ratio)
    for item in history:
        if float(item.get("global_accuracy", 0.0)) >= threshold:
            return int(item.get("round_num", 0)) + 1
    return len(history)


def _aggregate_privacy_overhead(history: List[Dict[str, Any]]) -> float:
    return _round_to(sum(float(item.get("privacy_overhead_time_total", 0.0)) for item in history))


def _stability_metrics(history: List[Dict[str, Any]]) -> Dict[str, float]:
    accuracies = [float(item.get("global_accuracy", 0.0)) for item in history]
    if not accuracies:
        return {
            "accuracy_std": 0.0,
            "accuracy_mean": 0.0,
            "accuracy_cv": 0.0,
        }

    acc_mean = float(mean(accuracies))
    acc_std = float(pstdev(accuracies)) if len(accuracies) > 1 else 0.0
    cv = (acc_std / acc_mean) if acc_mean > 0 else 0.0
    return {
        "accuracy_std": _round_to(acc_std),
        "accuracy_mean": _round_to(acc_mean),
        "accuracy_cv": _round_to(cv),
    }


def extract_primary_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract comparable headline metrics from one training result payload."""
    summary = result.get("summary", {})
    history = result.get("history", [])

    final_accuracy = float(summary.get("final_global_accuracy", 0.0))
    total_training_time = float(summary.get("total_training_time", 0.0))

    compressed_bytes_total = sum(
        float(item.get("communication_compressed_cost_bytes", 0.0)) for item in history
    )
    if compressed_bytes_total > 0.0:
        comm_cost_mb = compressed_bytes_total / float(1024 * 1024)
    elif "total_communication_mb" in summary:
        comm_cost_mb = float(summary.get("total_communication_mb", 0.0))
    else:
        comm_cost_mb = sum(float(item.get("communication_cost_mb", 0.0)) for item in history)

    metrics = {
        "final_accuracy": _round_to(final_accuracy),
        "convergence_rounds": int(_infer_convergence_round(history)),
        "communication_cost_mb": _round_to(comm_cost_mb),
        "training_time_sec": _round_to(total_training_time),
        "privacy_overhead_sec": _round_to(_aggregate_privacy_overhead(history)),
    }
    metrics.update(_stability_metrics(history))
    return metrics


def confidence_interval_95(values: Iterable[float]) -> Dict[str, float]:
    """Compute normal-approximation 95% confidence interval for run averaging."""
    arr = [float(v) for v in values]
    if not arr:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    m = float(mean(arr))
    if len(arr) == 1:
        return {"mean": _round_to(m), "ci_low": _round_to(m), "ci_high": _round_to(m)}

    std = float(pstdev(arr))
    margin = 1.96 * (std / math.sqrt(len(arr)))
    return {
        "mean": _round_to(m),
        "ci_low": _round_to(m - margin),
        "ci_high": _round_to(m + margin),
    }


def dataclass_to_dict(data: Any) -> Dict[str, Any]:
    """Safe converter for dataclass instances."""
    return asdict(data)
