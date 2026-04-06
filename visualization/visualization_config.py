"""Configuration for experiment visualization generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class VisualizationConfig:
    """Config-driven switches and style knobs for research plots."""

    metrics_dir: Path = Path("results/experiments/metrics")
    comparison_table_json: Path = Path("results/experiments/comparison_table.json")
    output_dir: Path = Path("results/plots")

    method_baseline: str = "baseline"
    method_aflf: str = "aflf_full"
    method_label_map: Dict[str, str] = field(
        default_factory=lambda: {
            "baseline": "FedAvg",
            "aflf_full": "AFLF",
            "selection_only": "Selection-Only",
            "privacy_only": "Privacy-Only",
        }
    )

    plot_accuracy: bool = True
    plot_loss: bool = True
    plot_communication: bool = True
    plot_accuracy_comparison: bool = True
    plot_convergence: bool = True
    plot_lr_curve: bool = False
    plot_client_participation_histogram: bool = False

    figure_size: Tuple[int, int] = (10, 6)
    dpi: int = 300
    font_size: int = 14
    title_size: int = 16
    legend_font_size: int = 12
    grid_alpha: float = 0.35
    line_width: float = 2.3
    marker_size: int = 5
    ci_alpha: float = 0.20
    smoothing_window: int = 1
    convergence_target_accuracy: float = 0.95
    legend_outside: bool = True
    tight_layout: bool = True
    communication_log_scale: bool = True

    color_palette: Dict[str, str] = field(
        default_factory=lambda: {
            "baseline": "#1f77b4",
            "aflf_full": "#d62728",
            "selection_only": "#2ca02c",
            "privacy_only": "#9467bd",
        }
    )

    style: str = "seaborn-v0_8-whitegrid"
    save_format: str = "png"

    def method_label(self, method: str) -> str:
        return self.method_label_map.get(method, method)
