"""Reusable Streamlit UI components for the AFLF dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from dashboard.dashboard_utils import format_float, format_percent, format_seconds


def no_data(message: str = "No data found") -> None:
    st.info(message)


def metric_row(final_accuracy: Any, comm_reduction: Any, rounds: Any, training_time: Any) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Accuracy", format_percent(final_accuracy, 2))
    c2.metric("Communication Reduction", format_percent(comm_reduction, 2))
    c3.metric("Rounds", format_float(rounds, 0))
    c4.metric("Training Time", format_seconds(training_time, 2))


def render_summary_table(df: pd.DataFrame, caption: str) -> None:
    if df.empty:
        no_data("No experiment summary available")
        return
    st.caption(caption)
    st.dataframe(df, use_container_width=True, hide_index=True)


def extract_overview_metrics(round_df: pd.DataFrame) -> dict[str, Any]:
    if round_df.empty:
        return {
            "final_accuracy": None,
            "comm_reduction": None,
            "rounds": None,
            "training_time": None,
        }

    last = round_df.iloc[-1]
    return {
        "final_accuracy": last.get("global_accuracy"),
        "comm_reduction": last.get("communication_reduction_percentage"),
        "rounds": (last.get("round_num") + 1) if pd.notna(last.get("round_num")) else None,
        "training_time": last.get("total_training_time"),
    }
