"""Streamlit dashboard for AFLF experiment visualization (read-only)."""

from __future__ import annotations

import platform
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import torch
except Exception:  # pragma: no cover - torch may not be available in all environments
    torch = None

from aflf import __version__ as aflf_version
from dashboard.components import (
    extract_overview_metrics,
    metric_row,
    no_data,
    render_summary_table,
)
from dashboard.config_viewer import render_config_viewer
from dashboard.dashboard_utils import format_percent, to_display_name
from dashboard.data_loader import (
    discover_runs,
    load_comparison_df,
    load_experiment_tracker_df,
    load_run,
)
from dashboard.plots import (
    client_accuracy_histogram,
    communication_per_round,
    comparison_bar,
    training_curves,
)

st.set_page_config(
    page_title="AFLF Research Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _methods_df_from_runs() -> pd.DataFrame:
    runs = discover_runs()
    rows: list[dict[str, str]] = []
    for item in runs.values():
        rows.append(
            {
                "run_name": item["run_name"],
                "method": item["method"],
                "path": str(item["rounds_csv"]),
            }
        )
    return pd.DataFrame(rows)


def render_overview_page(run_key: str) -> None:
    st.title("Adaptive Federated Learning Framework")
    st.caption("Research dashboard for federated experiment visualization and system metrics")

    run = load_run(run_key)
    if run is None:
        no_data()
        return

    metrics = extract_overview_metrics(run.round_metrics)
    metric_row(
        metrics["final_accuracy"],
        metrics["comm_reduction"],
        metrics["rounds"],
        metrics["training_time"],
    )

    tracker_df = load_experiment_tracker_df()
    if not tracker_df.empty:
        with st.expander("Experiment Summary", expanded=True):
            view_cols = [
                col
                for col in [
                    "experiment_name",
                    "run_index",
                    "metrics.final_accuracy",
                    "metrics.communication_cost_mb",
                    "metrics.convergence_rounds",
                    "metrics.training_time_sec",
                    "finished_at",
                ]
                if col in tracker_df.columns
            ]
            render_summary_table(tracker_df[view_cols], "Tracked experiments")
    else:
        no_data("No experiment summary table found")


def render_training_page(run_key: str) -> None:
    st.title("Training Analytics")

    run = load_run(run_key)
    if run is None or run.round_metrics.empty:
        no_data()
        return

    tabs = st.tabs(["Training Curves", "Client Participation"]) 

    with tabs[0]:
        figures = training_curves(run.round_metrics)
        if not figures:
            no_data("No training metric columns found")
            return

        c1, c2 = st.columns(2)
        if "accuracy" in figures:
            c1.plotly_chart(figures["accuracy"], use_container_width=True)
        if "loss" in figures:
            c2.plotly_chart(figures["loss"], use_container_width=True)

        c3, c4 = st.columns(2)
        if "lr" in figures:
            c3.plotly_chart(figures["lr"], use_container_width=True)
        if "time" in figures:
            c4.plotly_chart(figures["time"], use_container_width=True)

    with tabs[1]:
        if run.client_metrics.empty:
            no_data("No client participation data found")
        else:
            c1, c2 = st.columns(2)
            if "client_id" in run.client_metrics.columns:
                participation = (
                    run.client_metrics.groupby("client_id")["round_num"].nunique().sort_values(ascending=False)
                )
                c1.bar_chart(participation)
                c1.caption("Client participation count (number of rounds joined)")
            fig = client_accuracy_histogram(run.client_metrics)
            if fig is not None:
                c2.plotly_chart(fig, use_container_width=True)
            else:
                c2.info("No data found")


def render_experiments_page() -> None:
    st.title("Experiment Comparison")

    comparison_df = load_comparison_df()
    if comparison_df.empty:
        no_data("No comparison data found")
        return

    normalized = comparison_df.copy()
    if "method" in normalized.columns:
        normalized["method"] = normalized["method"].replace({"baseline": "FedAvg", "aflf_full": "AFLF"})

    c1, c2, c3 = st.columns(3)

    if "accuracy" in normalized.columns:
        c1.plotly_chart(comparison_bar(normalized, "accuracy", "Final Accuracy"), use_container_width=True)
    else:
        c1.info("No data found")

    if "comm_cost_mb" in normalized.columns:
        c2.plotly_chart(
            comparison_bar(normalized, "comm_cost_mb", "Communication Cost (MB)"),
            use_container_width=True,
        )
    else:
        c2.info("No data found")

    if "rounds" in normalized.columns:
        c3.plotly_chart(comparison_bar(normalized, "rounds", "Rounds"), use_container_width=True)
    else:
        c3.info("No data found")

    st.caption("FedAvg and AFLF can be compared directly when present in experiment outputs")


def render_communication_page(run_key: str) -> None:
    st.title("Communication Analysis")

    run = load_run(run_key)
    if run is None or run.round_metrics.empty:
        no_data()
        return

    last = run.round_metrics.iloc[-1]
    total_mb = last.get("communication_cost_mb")
    reduction = last.get("communication_reduction_percentage")
    precision = last.get("communication_precision_mode", "N/A")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Communication", f"{float(total_mb):.2f} MB" if pd.notna(total_mb) else "N/A")
    c2.metric("Reduction", format_percent(reduction, 2))
    c3.metric("Precision", str(precision))

    fig = communication_per_round(run.round_metrics)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        no_data("No per-round communication metrics found")


def render_system_info_page(run_key: str) -> None:
    st.title("System Info")

    runtime_tab, config_tab = st.tabs(["Runtime", "Config Viewer"])

    with runtime_tab:
        run = load_run(run_key)

        device = "N/A"
        torch_version = "Not installed"
        if torch is not None:
            torch_version = torch.__version__
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        info_cols = st.columns(2)
        info_cols[0].metric("Python Version", platform.python_version())
        info_cols[0].metric("Torch Version", torch_version)
        info_cols[0].metric("Device", device)

        info_cols[1].metric("AFLF Version", aflf_version)
        info_cols[1].metric("Run Environment", platform.platform())
        info_cols[1].metric("Selected Run Method", run.method if run else "N/A")

        if run and run.experiment_json:
            with st.expander("Raw Experiment JSON", expanded=False):
                st.caption(str(Path(run.experiment_json)))
                st.json(run.experiment_payload or {"message": "No data found"}, expanded=False)

    with config_tab:
        default_config = None
        run = load_run(run_key)
        if run and run.experiment_payload:
            experiment = run.experiment_payload.get("experiment", {})
            if isinstance(experiment, dict):
                base_path = experiment.get("base_config_path")
                if isinstance(base_path, str):
                    default_config = str((Path.cwd() / base_path).resolve())
        render_config_viewer(default_path=default_config)


def main() -> None:
    runs = discover_runs()

    st.sidebar.title("AFLF Dashboard")
    page = st.sidebar.selectbox(
        "Pages",
        options=["Overview", "Training Metrics", "Experiments", "Communication", "System Info"],
    )

    if not runs:
        st.title("Adaptive Federated Learning Framework")
        no_data("No data found. Expected results under results/metrics and experiment outputs.")
        return

    run_options = list(runs.keys())
    selected_run = st.sidebar.selectbox(
        "Select experiment run",
        options=run_options,
        format_func=to_display_name,
    )

    method_df = _methods_df_from_runs()
    with st.sidebar.expander("Detected Runs", expanded=False):
        st.dataframe(method_df, use_container_width=True, hide_index=True)

    if page == "Overview":
        render_overview_page(selected_run)
    elif page == "Training Metrics":
        render_training_page(selected_run)
    elif page == "Experiments":
        render_experiments_page()
    elif page == "Communication":
        render_communication_page(selected_run)
    else:
        render_system_info_page(selected_run)


if __name__ == "__main__":
    main()
