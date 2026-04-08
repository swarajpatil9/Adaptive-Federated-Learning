"""Plotting helpers for the AFLF Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _line(df: pd.DataFrame, x: str, y: str, title: str, color: str = "#1f77b4") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y],
            mode="lines+markers",
            line={"width": 2.5, "color": color},
            marker={"size": 6},
            name=y,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x.replace("_", " ").title(),
        yaxis_title=y.replace("_", " ").title(),
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        height=360,
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def training_curves(round_df: pd.DataFrame) -> dict[str, go.Figure]:
    figures: dict[str, go.Figure] = {}
    if round_df.empty or "round_num" not in round_df.columns:
        return figures

    if "global_accuracy" in round_df.columns:
        figures["accuracy"] = _line(round_df, "round_num", "global_accuracy", "Accuracy vs Rounds", "#0f766e")
    if "global_loss" in round_df.columns:
        figures["loss"] = _line(round_df, "round_num", "global_loss", "Loss vs Rounds", "#dc2626")
    if "learning_rate" in round_df.columns:
        figures["lr"] = _line(round_df, "round_num", "learning_rate", "Learning Rate Adaptation", "#7c3aed")
    if "round_time" in round_df.columns:
        figures["time"] = _line(round_df, "round_num", "round_time", "Round Duration", "#2563eb")

    return figures


def comparison_bar(comparison_df: pd.DataFrame, y_col: str, title: str) -> go.Figure:
    fig = px.bar(
        comparison_df,
        x="method",
        y=y_col,
        color="method",
        title=title,
        text_auto=".3f",
        template="plotly_white",
        color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    )
    fig.update_layout(showlegend=False, margin={"l": 20, "r": 20, "t": 50, "b": 20}, height=360)
    return fig


def communication_per_round(round_df: pd.DataFrame) -> go.Figure | None:
    if round_df.empty or "round_num" not in round_df.columns:
        return None

    y_col = None
    for col in ["communication_cost_mb", "communication_cost_bytes"]:
        if col in round_df.columns:
            y_col = col
            break
    if y_col is None:
        return None

    fig = px.area(
        round_df,
        x="round_num",
        y=y_col,
        title="Per-Round Communication",
        template="plotly_white",
    )
    fig.update_layout(margin={"l": 20, "r": 20, "t": 50, "b": 20}, height=360)
    return fig


def client_accuracy_histogram(client_df: pd.DataFrame) -> go.Figure | None:
    if client_df.empty or "train_accuracy" not in client_df.columns:
        return None
    fig = px.histogram(
        client_df,
        x="train_accuracy",
        nbins=15,
        title="Client Accuracy Distribution",
        template="plotly_white",
        color_discrete_sequence=["#14b8a6"],
    )
    fig.update_layout(margin={"l": 20, "r": 20, "t": 50, "b": 20}, height=360)
    return fig
