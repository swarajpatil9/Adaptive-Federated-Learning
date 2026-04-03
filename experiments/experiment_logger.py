"""Experiment tracking and persistence utilities for Phase 12."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .experiment_utils import dump_json, ensure_dir


@dataclass
class ExperimentRunRecord:
    """One executed run of one experiment configuration."""

    experiment_name: str
    run_index: int
    seed: int
    toggles: Dict[str, bool]
    metrics: Dict[str, Any]
    started_at: str
    finished_at: str


@dataclass
class ExperimentTracker:
    """Collects and summarizes experiment run records."""

    records: List[ExperimentRunRecord] = field(default_factory=list)

    def add(self, record: ExperimentRunRecord) -> None:
        self.records.append(record)

    def records_for(self, experiment_name: str) -> List[ExperimentRunRecord]:
        return [record for record in self.records if record.experiment_name == experiment_name]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_records": len(self.records),
            "records": [
                {
                    "experiment_name": item.experiment_name,
                    "run_index": item.run_index,
                    "seed": item.seed,
                    "toggles": item.toggles,
                    "metrics": item.metrics,
                    "started_at": item.started_at,
                    "finished_at": item.finished_at,
                }
                for item in self.records
            ],
        }


class ExperimentLogger:
    """Persists experiment outputs and generates comparison table data."""

    def __init__(self, output_dir: str = "results/experiments"):
        self.output_dir = ensure_dir(Path(output_dir))

    @staticmethod
    def now_iso() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def write_named_results(self, name: str, payload: Dict[str, Any]) -> Path:
        """Write canonical result JSON like baseline_results.json."""
        return dump_json(payload, self.output_dir / f"{name}_results.json")

    def write_ablation_results(self, payload: Dict[str, Any]) -> Path:
        return dump_json(payload, self.output_dir / "ablation_results.json")

    def write_tracker(self, tracker: ExperimentTracker) -> Path:
        return dump_json(tracker.to_dict(), self.output_dir / "experiment_tracker.json")

    def build_comparison_table(self, aggregated: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build comparison-table records for reporting and future visualization."""
        rows: List[Dict[str, Any]] = []
        for method, data in aggregated.items():
            metrics = data.get("metrics", {})
            rows.append(
                {
                    "method": method,
                    "accuracy": metrics.get("final_accuracy", 0.0),
                    "comm_cost_mb": metrics.get("communication_cost_mb", 0.0),
                    "rounds": metrics.get("convergence_rounds", 0),
                    "time_sec": metrics.get("training_time_sec", 0.0),
                    "privacy_overhead_sec": metrics.get("privacy_overhead_sec", 0.0),
                    "stability_std": metrics.get("accuracy_std", 0.0),
                }
            )
        return rows

    def write_comparison_table(self, table_rows: List[Dict[str, Any]]) -> Path:
        payload = {
            "generated_at": self.now_iso(),
            "columns": [
                "method",
                "accuracy",
                "comm_cost_mb",
                "rounds",
                "time_sec",
                "privacy_overhead_sec",
                "stability_std",
            ],
            "rows": table_rows,
        }
        return dump_json(payload, self.output_dir / "comparison_table.json")
