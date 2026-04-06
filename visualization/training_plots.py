"""Training-trajectory plots (accuracy, loss, learning rate)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from .visualization_config import VisualizationConfig
from .visualization_utils import method_color, save_figure, summarize_metric_by_round


class TrainingPlotter:
    """Create training dynamics plots from round-level experiment metrics."""

    def __init__(self, config: VisualizationConfig):
        self.config = config

    def plot_accuracy_vs_rounds(
        self,
        method_rounds: Dict[str, pd.DataFrame],
        output_dir: Path,
    ) -> Path | None:
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        plotted = False

        for method in [self.config.method_baseline, self.config.method_aflf]:
            frame = method_rounds.get(method, pd.DataFrame())
            if frame.empty or "global_accuracy" not in frame.columns:
                continue

            summary = summarize_metric_by_round(
                frame,
                round_col="round_num",
                value_col="global_accuracy",
                smoothing_window=self.config.smoothing_window,
            )
            color = method_color(self.config, method)
            label = self.config.method_label(method)

            ax.plot(
                summary["round_num"],
                summary["mean_smoothed"],
                label=label,
                color=color,
                linewidth=self.config.line_width,
                marker="o",
                markersize=self.config.marker_size,
            )
            ax.fill_between(
                summary["round_num"],
                summary["low"],
                summary["high"],
                color=color,
                alpha=self.config.ci_alpha,
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            return None

        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Global Accuracy")
        ax.set_title("Accuracy vs Communication Rounds")
        ax.grid(True)
        if self.config.legend_outside:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        else:
            ax.legend()

        output_path = output_dir / f"accuracy_vs_rounds.{self.config.save_format}"
        save_figure(fig, output_path, self.config)
        return output_path

    def plot_loss_vs_rounds(
        self,
        method_rounds: Dict[str, pd.DataFrame],
        output_dir: Path,
    ) -> Path | None:
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        plotted = False

        for method in [self.config.method_baseline, self.config.method_aflf]:
            frame = method_rounds.get(method, pd.DataFrame())
            if frame.empty or "global_loss" not in frame.columns:
                continue

            summary = summarize_metric_by_round(
                frame,
                round_col="round_num",
                value_col="global_loss",
                smoothing_window=self.config.smoothing_window,
            )
            color = method_color(self.config, method)
            label = self.config.method_label(method)

            ax.plot(
                summary["round_num"],
                summary["mean_smoothed"],
                label=label,
                color=color,
                linewidth=self.config.line_width,
                marker="o",
                markersize=self.config.marker_size,
            )
            ax.fill_between(
                summary["round_num"],
                summary["low"],
                summary["high"],
                color=color,
                alpha=self.config.ci_alpha,
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            return None

        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Global Loss")
        ax.set_title("Loss vs Communication Rounds")
        ax.grid(True)
        if self.config.legend_outside:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        else:
            ax.legend()

        output_path = output_dir / f"loss_vs_rounds.{self.config.save_format}"
        save_figure(fig, output_path, self.config)
        return output_path

    def plot_lr_adaptation_curve(
        self,
        method_rounds: Dict[str, pd.DataFrame],
        output_dir: Path,
    ) -> Path | None:
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        plotted = False

        for method in [self.config.method_baseline, self.config.method_aflf]:
            frame = method_rounds.get(method, pd.DataFrame())
            if frame.empty or "learning_rate" not in frame.columns:
                continue

            summary = summarize_metric_by_round(
                frame,
                round_col="round_num",
                value_col="learning_rate",
                smoothing_window=1,
            )
            color = method_color(self.config, method)
            label = self.config.method_label(method)

            ax.plot(
                summary["round_num"],
                summary["mean_smoothed"],
                label=label,
                color=color,
                linewidth=self.config.line_width,
                marker="o",
                markersize=self.config.marker_size,
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            return None

        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning-Rate Adaptation Curve")
        ax.grid(True)
        if self.config.legend_outside:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        else:
            ax.legend()

        output_path = output_dir / f"lr_adaptation_curve.{self.config.save_format}"
        save_figure(fig, output_path, self.config)
        return output_path
