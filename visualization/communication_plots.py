"""Communication and participation visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .visualization_config import VisualizationConfig
from .visualization_utils import compute_ci, method_color, percentage_reduction, save_figure


class CommunicationPlotter:
    """Plot communication-cost and participation analyses."""

    def __init__(self, config: VisualizationConfig):
        self.config = config

    def plot_communication_comparison(
        self,
        method_rounds: Dict[str, pd.DataFrame],
        output_dir: Path,
    ) -> Path | None:
        methods = [self.config.method_baseline, self.config.method_aflf]
        rows: List[Dict[str, float]] = []

        for method in methods:
            frame = method_rounds.get(method, pd.DataFrame())
            if frame.empty or "communication_compressed_cost_bytes" not in frame.columns:
                continue

            per_run = (
                frame.groupby("run_id", as_index=False)["communication_compressed_cost_bytes"]
                .sum()
                .rename(columns={"communication_compressed_cost_bytes": "compressed_bytes_total"})
            )
            values_mb = per_run["compressed_bytes_total"].to_numpy(dtype=float) / float(1024 * 1024)
            if values_mb.size == 0:
                continue

            rows.append(
                {
                    "method": method,
                    "label": self.config.method_label(method),
                    "mean_mb": float(np.mean(values_mb)),
                    "ci": compute_ci(values_mb),
                    "color": method_color(self.config, method),
                }
            )

        if not rows:
            return None

        baseline_row = next((row for row in rows if row["method"] == self.config.method_baseline), None)
        aflf_row = next((row for row in rows if row["method"] == self.config.method_aflf), None)
        reduction = 0.0
        if baseline_row and aflf_row:
            reduction = percentage_reduction(baseline_row["mean_mb"], aflf_row["mean_mb"])

        fig, axes = plt.subplots(1, 2, figsize=(self.config.figure_size[0] * 1.8, self.config.figure_size[1]))

        labels = [row["label"] for row in rows]
        means = [row["mean_mb"] for row in rows]
        cis = [row["ci"] for row in rows]
        colors = [row["color"] for row in rows]
        bars = axes[0].bar(labels, means, yerr=cis, capsize=7, color=colors, alpha=0.92)
        axes[0].set_xlabel("Method")
        axes[0].set_ylabel("Total Communication (MB)")
        axes[0].set_title("Communication Cost Comparison")
        axes[0].grid(True, axis="y")
        if self.config.communication_log_scale:
            axes[0].set_yscale("log")

        for bar, value in zip(bars, means):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        if baseline_row and aflf_row:
            axes[1].bar(
                ["Reduction %"],
                [reduction],
                color=method_color(self.config, self.config.method_aflf),
                alpha=0.92,
            )
            axes[1].set_ylim(0, max(100, reduction * 1.25 + 5))
            axes[1].text(0, reduction, f"{reduction:.2f}%", ha="center", va="bottom")
        else:
            axes[1].bar(["Reduction %"], [0.0], color="#7f7f7f", alpha=0.92)
            axes[1].text(0, 0.0, "N/A", ha="center", va="bottom")

        axes[1].set_ylabel("Communication Reduction (%)")
        axes[1].set_title("AFLF Communication Reduction")
        axes[1].grid(True, axis="y")

        output_path = output_dir / f"communication_comparison.{self.config.save_format}"
        save_figure(fig, output_path, self.config)
        return output_path

    def plot_client_participation_histogram(
        self,
        method_clients: Dict[str, pd.DataFrame],
        output_dir: Path,
    ) -> Path | None:
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        plotted = False

        for method in [self.config.method_baseline, self.config.method_aflf]:
            frame = method_clients.get(method, pd.DataFrame())
            if frame.empty or "round_num" not in frame.columns or "client_id" not in frame.columns:
                continue

            counts = frame.groupby(["run_id", "round_num"], as_index=False)["client_id"].nunique()
            values = counts["client_id"].to_numpy(dtype=float)
            if values.size == 0:
                continue

            ax.hist(
                values,
                bins=max(3, int(len(np.unique(values)))),
                alpha=0.6,
                label=self.config.method_label(method),
                color=method_color(self.config, method),
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            return None

        ax.set_xlabel("Participating Clients per Round")
        ax.set_ylabel("Frequency")
        ax.set_title("Client Participation Histogram")
        ax.grid(True)
        if self.config.legend_outside:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        else:
            ax.legend()

        output_path = output_dir / f"client_participation_histogram.{self.config.save_format}"
        save_figure(fig, output_path, self.config)
        return output_path
