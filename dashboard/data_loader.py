"""Data loading layer for the AFLF Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.dashboard_utils import (
    CONFIGS_ROOT,
    PROJECT_ROOT,
    find_results_dirs,
    infer_method_from_name,
    safe_read_csv,
    safe_read_json,
    safe_read_yaml,
)


@dataclass
class RunData:
    run_key: str
    run_name: str
    method: str
    root: Path
    rounds_csv: Path | None
    clients_csv: Path | None
    experiment_json: Path | None
    round_metrics: pd.DataFrame
    client_metrics: pd.DataFrame
    experiment_payload: dict[str, Any]


@st.cache_data(show_spinner=False)
def discover_runs() -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for root in find_results_dirs():
        for rounds_csv in root.rglob("*_rounds.csv"):
            stem = rounds_csv.stem[: -len("_rounds")]
            clients_csv = rounds_csv.with_name(f"{stem}_clients.csv")
            experiment_json = rounds_csv.with_name(f"{stem}_experiment.json")

            run_name = stem
            method = infer_method_from_name(run_name)
            run_key = f"{method}::{run_name}::{rounds_csv.parent.relative_to(PROJECT_ROOT)}"

            runs[run_key] = {
                "run_key": run_key,
                "run_name": run_name,
                "method": method,
                "root": root,
                "rounds_csv": rounds_csv,
                "clients_csv": clients_csv if clients_csv.exists() else None,
                "experiment_json": experiment_json if experiment_json.exists() else None,
            }
    return dict(sorted(runs.items(), key=lambda x: x[0]))


@st.cache_data(show_spinner=False)
def load_run(run_key: str) -> RunData | None:
    run_map = discover_runs()
    if run_key not in run_map:
        return None

    info = run_map[run_key]
    rounds_csv = Path(info["rounds_csv"])
    clients_csv = Path(info["clients_csv"]) if info["clients_csv"] else None
    experiment_json = Path(info["experiment_json"]) if info["experiment_json"] else None

    round_df = safe_read_csv(rounds_csv)
    client_df = safe_read_csv(clients_csv) if clients_csv else pd.DataFrame()

    payload = safe_read_json(experiment_json) if experiment_json else None
    experiment_payload: dict[str, Any] = payload if isinstance(payload, dict) else {}

    if not round_df.empty and "round_num" in round_df.columns:
        round_df = round_df.sort_values("round_num").reset_index(drop=True)

    return RunData(
        run_key=info["run_key"],
        run_name=info["run_name"],
        method=info["method"],
        root=Path(info["root"]),
        rounds_csv=rounds_csv,
        clients_csv=clients_csv,
        experiment_json=experiment_json,
        round_metrics=round_df,
        client_metrics=client_df,
        experiment_payload=experiment_payload,
    )


@st.cache_data(show_spinner=False)
def load_experiment_tracker_df() -> pd.DataFrame:
    candidate_files = [
        PROJECT_ROOT / "results/experiments/experiment_tracker.json",
        PROJECT_ROOT / "results/system_check/experiments/experiment_tracker.json",
    ]

    for path in candidate_files:
        payload = safe_read_json(path)
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            return pd.json_normalize(payload["records"])
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_comparison_df() -> pd.DataFrame:
    candidate_files = [
        PROJECT_ROOT / "results/experiments/comparison_table.json",
        PROJECT_ROOT / "results/system_check/experiments/comparison_table.json",
    ]

    for path in candidate_files:
        payload = safe_read_json(path)
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return pd.DataFrame(payload["rows"])

    records: list[dict[str, Any]] = []
    for path in [
        PROJECT_ROOT / "results/experiments/baseline_results.json",
        PROJECT_ROOT / "results/experiments/aflf_results.json",
        PROJECT_ROOT / "results/system_check/experiments/baseline_results.json",
        PROJECT_ROOT / "results/system_check/experiments/aflf_results.json",
    ]:
        payload = safe_read_json(path)
        if not isinstance(payload, dict):
            continue
        metrics = payload.get("metrics", {})
        experiment = payload.get("experiment", {})
        if not isinstance(metrics, dict) or not isinstance(experiment, dict):
            continue
        records.append(
            {
                "method": experiment.get("name", path.stem),
                "accuracy": metrics.get("final_accuracy"),
                "comm_cost_mb": metrics.get("communication_cost_mb"),
                "rounds": metrics.get("convergence_rounds"),
                "time_sec": metrics.get("training_time_sec"),
            }
        )
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def load_available_configs() -> list[Path]:
    if not CONFIGS_ROOT.exists():
        return []
    return sorted(CONFIGS_ROOT.rglob("*.yaml"))


@st.cache_data(show_spinner=False)
def load_config_content(path_str: str) -> dict[str, Any] | list[Any] | None:
    path = Path(path_str)
    return safe_read_yaml(path)
