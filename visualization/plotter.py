"""Top-level visualization manager for PHASE 13 plotting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .communication_plots import CommunicationPlotter
from .comparison_plots import ComparisonPlotter
from .training_plots import TrainingPlotter
from .visualization_config import VisualizationConfig
from .visualization_utils import apply_plot_style, ensure_dir


@dataclass
class PlotArtifacts:
    """Holds generated output artifact paths by logical plot name."""

    paths: Dict[str, Path]


class PlotManager:
    """Reads experiment CSV outputs and orchestrates publication-style plots."""

    def __init__(self, config: VisualizationConfig | None = None):
        self.config = config or VisualizationConfig()
        self.training_plotter = TrainingPlotter(self.config)
        self.comparison_plotter = ComparisonPlotter(self.config)
        self.communication_plotter = CommunicationPlotter(self.config)

    @staticmethod
    def _method_and_run_id(rounds_csv_path: Path) -> tuple[str, str]:
        # File format expected: <method>_runX_<...>_rounds.csv
        stem = rounds_csv_path.stem.replace("_rounds", "")
        if "_run" not in stem:
            return stem, "run0"
        method, run_tail = stem.split("_run", 1)
        return method, f"run{run_tail}"

    def _load_method_frames(self) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        method_rounds_frames: Dict[str, List[pd.DataFrame]] = {}
        method_clients_frames: Dict[str, List[pd.DataFrame]] = {}

        for rounds_path in sorted(self.config.metrics_dir.glob("*_rounds.csv")):
            method, run_id = self._method_and_run_id(rounds_path)
            rounds_df = pd.read_csv(rounds_path)
            if rounds_df.empty:
                continue

            rounds_df["run_id"] = run_id
            rounds_df["method"] = method
            method_rounds_frames.setdefault(method, []).append(rounds_df)

            clients_path = rounds_path.with_name(rounds_path.name.replace("_rounds.csv", "_clients.csv"))
            if clients_path.exists():
                clients_df = pd.read_csv(clients_path)
                if not clients_df.empty:
                    clients_df["run_id"] = run_id
                    clients_df["method"] = method
                    method_clients_frames.setdefault(method, []).append(clients_df)

        method_rounds = {
            key: pd.concat(frames, ignore_index=True)
            for key, frames in method_rounds_frames.items()
            if frames
        }
        method_clients = {
            key: pd.concat(frames, ignore_index=True)
            for key, frames in method_clients_frames.items()
            if frames
        }
        return method_rounds, method_clients

    def generate_all_plots(self) -> PlotArtifacts:
        apply_plot_style(self.config)
        output_dir = ensure_dir(self.config.output_dir)
        method_rounds, method_clients = self._load_method_frames()

        outputs: Dict[str, Path] = {}

        if self.config.plot_accuracy:
            out = self.training_plotter.plot_accuracy_vs_rounds(method_rounds, output_dir)
            if out:
                outputs["accuracy_vs_rounds"] = out

        if self.config.plot_loss:
            out = self.training_plotter.plot_loss_vs_rounds(method_rounds, output_dir)
            if out:
                outputs["loss_vs_rounds"] = out

        if self.config.plot_communication:
            out = self.communication_plotter.plot_communication_comparison(method_rounds, output_dir)
            if out:
                outputs["communication_comparison"] = out

        if self.config.plot_accuracy_comparison:
            out = self.comparison_plotter.plot_baseline_vs_aflf_accuracy(method_rounds, output_dir)
            if out:
                outputs["aflf_vs_baseline"] = out

        if self.config.plot_convergence:
            out = self.comparison_plotter.plot_convergence_comparison(method_rounds, output_dir)
            if out:
                outputs["convergence_comparison"] = out

        if self.config.plot_lr_curve:
            out = self.training_plotter.plot_lr_adaptation_curve(method_rounds, output_dir)
            if out:
                outputs["lr_adaptation_curve"] = out

        if self.config.plot_client_participation_histogram:
            out = self.communication_plotter.plot_client_participation_histogram(method_clients, output_dir)
            if out:
                outputs["client_participation_histogram"] = out

        return PlotArtifacts(paths=outputs)


def main() -> None:
    manager = PlotManager()
    artifacts = manager.generate_all_plots()
    print("Generated plots:")
    for key, path in artifacts.paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
