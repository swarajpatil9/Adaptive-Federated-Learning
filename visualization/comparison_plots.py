"""Method-comparison plots for publication summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .visualization_config import VisualizationConfig
from .visualization_utils import (
    compute_ci,
    final_value_per_run,
    method_color,
    save_figure,
    summarize_metric_by_round,
)


class ComparisonPlotter:
    """Create high-level baseline vs AFLF comparison plots."""

    def __init__(self, config: VisualizationConfig):
        self.config = config

    def plot_baseline_vs_aflf_accuracy(
        self,
        method_rounds: Dict[str, pd.DataFrame],
        output_dir: Path,
    ) -> Path | None:
        methods = [self.config.method_baseline, self.config.method_aflf]
        labels: List[str] = []
        means: List[float] = []
        cis: List[float] = []
        colors: List[str] = []

        for method in methods:
            frame = method_rounds.get(method, pd.DataFrame())
            if frame.empty or "global_accuracy" not in frame.columns:
                continue

            finals = final_value_per_run(
                frame,
                run_col="run_id",
                round_col="round_num",
                value_col="global_accuracy",
            )["global_accuracy"].to_numpy(dtype=float)
            if finals.size == 0:
                continue

            labels.append(self.config.method_label(method))
            means.append(float(np.mean(finals)))
            cis.append(compute_ci(finals))
            colors.append(method_color(self.config, method))

        if not labels:
            return None

        fig, ax = plt.subplots(figsize=self.config.figure_size)
        bars = ax.bar(labels, means, yerr=cis, capsize=7, color=colors, alpha=0.92)
        ax.set_xlabel("Method")
        ax.set_ylabel("Final Accuracy")
        ax.set_title("Baseline vs AFLF: Final Accuracy")
        ax.grid(True, axis="y")

        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        output_path = output_dir / f"aflf_vs_baseline.{self.config.save_format}"
        save_figure(fig, output_path, self.config)
        return output_path

    def plot_convergence_comparison(
        self,
        method_rounds: Dict[str, pd.DataFrame],
        output_dir: Path,
    ) -> Path | None:
        methods = [self.config.method_baseline, self.config.method_aflf]
        rows: List[Dict[str, float]] = []
        target = self.config.convergence_target_accuracy

        for method in methods:
            frame = method_rounds.get(method, pd.DataFrame())
            if frame.empty or "global_accuracy" not in frame.columns:
                continue

            run_values: List[float] = []
            for run_id, run_df in frame.groupby("run_id"):
                sorted_run = run_df.sort_values("round_num")
                reached = sorted_run[sorted_run["global_accuracy"] >= target]
                if reached.empty:
                    run_values.append(float(sorted_run["round_num"].max() + 1))
                else:
                    run_values.append(float(reached["round_num"].iloc[0]))

            if not run_values:
                continue

            run_arr = np.asarray(run_values, dtype=float)
            rows.append(
                {
                    "method": method,
                    "label": self.config.method_label(method),
                    "mean_round": float(np.mean(run_arr)),
                    "ci": compute_ci(run_arr),
                    "color": method_color(self.config, method),
                }
            )

        if not rows:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(self.config.figure_size[0] * 1.8, self.config.figure_size[1]))

        for row in rows:
            frame = method_rounds[row["method"]]
            summary = summarize_metric_by_round(
                frame,
                round_col="round_num",
                value_col="global_accuracy",
                smoothing_window=self.config.smoothing_window,
            )
            axes[0].plot(
                summary["round_num"],
                summary["mean_smoothed"],
                label=row["label"],
                color=row["color"],
                linewidth=self.config.line_width,
            )
            axes[0].fill_between(
                summary["round_num"],
                summary["low"],
                summary["high"],
                color=row["color"],
                alpha=self.config.ci_alpha,
            )

        axes[0].axhline(target, linestyle="--", color="#2f2f2f", label=f"Target={target:.2f}")
        axes[0].set_xlabel("Communication Round")
        axes[0].set_ylabel("Global Accuracy")
        axes[0].set_title("Convergence Curves")
        axes[0].grid(True)

        labels = [row["label"] for row in rows]
        means = [row["mean_round"] for row in rows]
        cis = [row["ci"] for row in rows]
        colors = [row["color"] for row in rows]
        bars = axes[1].bar(labels, means, yerr=cis, capsize=7, color=colors, alpha=0.92)
        axes[1].set_xlabel("Method")
        axes[1].set_ylabel("Rounds to Reach Target")
        axes[1].set_title(f"Rounds to Reach {target:.2f} Accuracy")
        axes[1].grid(True, axis="y")

        for bar, value in zip(bars, means):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        if self.config.legend_outside:
            axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        else:
            axes[0].legend()

        output_path = output_dir / f"convergence_comparison.{self.config.save_format}"
        save_figure(fig, output_path, self.config)
        return output_path
