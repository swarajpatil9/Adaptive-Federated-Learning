"""Phase 12 reproducible experiment runner for baseline vs AFLF evaluation."""

from __future__ import annotations

import argparse
from typing import Any, Dict, Iterable, List

from aflf.training import FederatedTrainer, build_federated_config

from .ablation import AblationManager
from .experiment_config import ExperimentConfig
from .experiment_logger import ExperimentLogger, ExperimentRunRecord, ExperimentTracker
from .experiment_utils import (
    confidence_interval_95,
    extract_primary_metrics,
    set_global_reproducibility,
)


class ExperimentRunner:
    """Orchestrates repeatable, config-driven FL experiment runs."""

    def __init__(self, output_dir: str = "results/experiments"):
        self.logger = ExperimentLogger(output_dir=output_dir)
        self.tracker = ExperimentTracker()

    @staticmethod
    def _toggles(config: ExperimentConfig) -> Dict[str, bool]:
        return {
            "selection_enabled": config.selection_enabled,
            "privacy_enabled": config.privacy_enabled,
            "adaptive_lr_enabled": config.adaptive_lr_enabled,
            "compression_enabled": config.compression_enabled,
        }

    def _execute_single_run(
        self,
        config: ExperimentConfig,
        run_index: int,
        run_seed: int,
    ) -> Dict[str, Any]:
        set_global_reproducibility(run_seed)

        runtime_config = config.build_runtime_config()
        runtime_config["seed"] = int(run_seed)

        federated_config = build_federated_config(runtime_config)

        trainer = FederatedTrainer.from_components(
            data_config=runtime_config["data"],
            model_config=runtime_config["model"],
            federated_config=federated_config,
            selection_config=runtime_config.get("selection"),
            experiment_name=f"{config.name}_run{run_index}",
            metrics_output_dir=str(config.output_dir() / "metrics"),
        )

        raw_result = trainer.fit()
        primary_metrics = extract_primary_metrics(raw_result)

        return {
            "run_index": run_index,
            "seed": run_seed,
            "toggles": self._toggles(config),
            "metrics": primary_metrics,
            "raw_result": raw_result,
        }

    def run_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run one experiment (possibly multiple seeds) and aggregate results."""
        runs: List[Dict[str, Any]] = []

        for run_index in range(1, config.num_runs + 1):
            run_seed = int(config.seed) + (run_index - 1)
            started_at = self.logger.now_iso()
            run_out = self._execute_single_run(
                config=config,
                run_index=run_index,
                run_seed=run_seed,
            )
            finished_at = self.logger.now_iso()
            runs.append(run_out)

            self.tracker.add(
                ExperimentRunRecord(
                    experiment_name=config.name,
                    run_index=run_index,
                    seed=run_seed,
                    toggles=run_out["toggles"],
                    metrics=run_out["metrics"],
                    started_at=started_at,
                    finished_at=finished_at,
                )
            )

        metric_keys = list(runs[0]["metrics"].keys()) if runs else []
        aggregated_metrics: Dict[str, Any] = {}

        for key in metric_keys:
            values = [float(run["metrics"][key]) for run in runs]
            aggregated_metrics[key] = confidence_interval_95(values)

        summary_metrics = {
            key: aggregated_metrics[key]["mean"] for key in metric_keys
        }

        payload = {
            "experiment": config.to_dict(),
            "num_runs": len(runs),
            "runs": [
                {
                    "run_index": run["run_index"],
                    "seed": run["seed"],
                    "toggles": run["toggles"],
                    "metrics": run["metrics"],
                }
                for run in runs
            ],
            "aggregated_metrics": aggregated_metrics,
            "metrics": summary_metrics,
            "raw_results": [run["raw_result"] for run in runs],
        }

        return payload

    def run_many(self, configs: Iterable[ExperimentConfig]) -> Dict[str, Dict[str, Any]]:
        """Run multiple configs automatically and persist all required artifacts."""
        results: Dict[str, Dict[str, Any]] = {}
        for config in configs:
            payload = self.run_experiment(config)
            results[config.name] = payload

            if config.name == "baseline":
                self.logger.write_named_results("baseline", payload)
            elif config.name == "aflf_full":
                self.logger.write_named_results("aflf", payload)

        ablation_payload = {
            "experiments": {
                name: payload
                for name, payload in results.items()
                if name in {
                    "selection_only",
                    "privacy_only",
                    "adaptive_only",
                    "communication_only",
                    "aflf_full",
                }
            }
        }
        self.logger.write_ablation_results(ablation_payload)

        table_rows = self.logger.build_comparison_table(
            {name: payload for name, payload in results.items()}
        )
        self.logger.write_comparison_table(table_rows)
        self.logger.write_tracker(self.tracker)

        return results


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 12 experiment runner (baseline vs AFLF + ablations)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed used for reproducible runs",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of repeated runs per experiment",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="phase12_required",
        choices=["baseline_vs_aflf", "phase12_required", "full_ablation"],
        help="Which experiment set to execute",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/experiments",
        help="Experiment artifact directory",
    )
    return parser


def _configs_for_mode(mode: str, seed: int, num_runs: int) -> List[ExperimentConfig]:
    if mode == "baseline_vs_aflf":
        baseline = ExperimentConfig(
            name="baseline",
            description="FedAvg baseline only",
            selection_enabled=False,
            privacy_enabled=False,
            adaptive_lr_enabled=False,
            compression_enabled=False,
            seed=seed,
            num_runs=num_runs,
        )
        aflf_full = ExperimentConfig(
            name="aflf_full",
            description="AFLF full stack",
            selection_enabled=True,
            privacy_enabled=True,
            adaptive_lr_enabled=True,
            compression_enabled=True,
            seed=seed,
            num_runs=num_runs,
        )
        return [baseline, aflf_full]

    if mode == "full_ablation":
        suite = AblationManager.full_ablation_suite(seed=seed)
        baseline = ExperimentConfig(
            name="baseline",
            description="FedAvg baseline only",
            selection_enabled=False,
            privacy_enabled=False,
            adaptive_lr_enabled=False,
            compression_enabled=False,
            seed=seed,
            num_runs=num_runs,
        )
        configs = [baseline]
        for cfg in suite.values():
            cfg.num_runs = num_runs
            configs.append(cfg)
        return configs

    required = AblationManager.immediate_phase12_subset(seed=seed)
    configs = []
    for cfg in required.values():
        cfg.num_runs = num_runs
        configs.append(cfg)
    return configs


def main() -> None:
    args = _arg_parser().parse_args()

    configs = _configs_for_mode(
        mode=args.mode,
        seed=int(args.seed),
        num_runs=int(args.num_runs),
    )

    runner = ExperimentRunner(output_dir=args.output_dir)
    results = runner.run_many(configs)

    print("=" * 88)
    print("PHASE 12 EXPERIMENTS COMPLETE")
    print("=" * 88)
    for name, payload in results.items():
        metrics = payload.get("metrics", {})
        print(
            f"{name:18s} "
            f"acc={metrics.get('final_accuracy', 0.0):.4f} "
            f"comm={metrics.get('communication_cost_mb', 0.0):.3f}MB "
            f"rounds={int(metrics.get('convergence_rounds', 0))} "
            f"time={metrics.get('training_time_sec', 0.0):.2f}s"
        )
    print("=" * 88)


if __name__ == "__main__":
    main()
