"""Shared utilities for data preparation and plotting."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .visualization_config import VisualizationConfig


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def apply_plot_style(config: VisualizationConfig) -> None:
    plt.style.use(config.style)
    plt.rcParams.update(
        {
            "font.size": config.font_size,
            "axes.titlesize": config.title_size,
            "axes.labelsize": config.font_size,
            "legend.fontsize": config.legend_font_size,
            "xtick.labelsize": config.font_size - 1,
            "ytick.labelsize": config.font_size - 1,
            "axes.grid": True,
            "grid.alpha": config.grid_alpha,
            "savefig.dpi": config.dpi,
        }
    )


def smooth_series(values: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return values
    return values.rolling(window=window, min_periods=1).mean()


def summarize_metric_by_round(
    df: pd.DataFrame,
    round_col: str,
    value_col: str,
    smoothing_window: int,
) -> pd.DataFrame:
    grouped = df.groupby(round_col)[value_col].agg(mean="mean", std="std", count="count").reset_index()
    grouped[round_col] = pd.to_numeric(grouped[round_col], errors="coerce")
    grouped["mean"] = pd.to_numeric(grouped["mean"], errors="coerce")

    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["ci"] = grouped.apply(
        lambda row: 1.96 * row["std"] / math.sqrt(row["count"]) if row["count"] > 1 else 0.0,
        axis=1,
    )
    grouped["mean_smoothed"] = smooth_series(grouped["mean"], smoothing_window)
    grouped["ci_smoothed"] = smooth_series(grouped["ci"], smoothing_window)
    grouped["low"] = grouped["mean_smoothed"] - grouped["ci_smoothed"]
    grouped["high"] = grouped["mean_smoothed"] + grouped["ci_smoothed"]
    return grouped


def method_color(config: VisualizationConfig, method: str) -> str:
    return config.color_palette.get(method, "#4c4c4c")


def final_value_per_run(
    df: pd.DataFrame,
    run_col: str,
    round_col: str,
    value_col: str,
) -> pd.DataFrame:
    sorted_df = df.sort_values([run_col, round_col])
    return sorted_df.groupby(run_col, as_index=False).tail(1)[[run_col, value_col]]


def save_figure(fig: plt.Figure, output_path: Path, config: VisualizationConfig) -> None:
    if config.tight_layout:
        fig.tight_layout()
    fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


def optional_numeric_column(df: pd.DataFrame, column: str) -> Optional[pd.Series]:
    if column not in df.columns:
        return None
    return pd.to_numeric(df[column], errors="coerce")


def combine_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    valid = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True)


def percentage_reduction(baseline: float, improved: float) -> float:
    if baseline <= 0:
        return 0.0
    return ((baseline - improved) / baseline) * 100.0


def compute_ci(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(1.96 * np.std(values, ddof=0) / np.sqrt(values.size))


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
