"""
Main entry point for federated learning experiments.

Usage:
    python main.py
    python main.py --config configs/baseline.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

from aflf.config import ConfigValidator, load_and_validate_config
from aflf.logging import LoggingConfig, SystemLogger
from aflf.system import DependencyChecker, EnvironmentChecker, ExperimentSeedManager
from aflf.training import (
    FederatedTrainer,
    build_federated_config,
)

AFLF_VERSION = "1.0"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Adaptive Federated Learning Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"AFLF Framework v{AFLF_VERSION}",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Optional configuration YAML path (defaults to baseline)",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override federated.num_rounds from config for fast sanity runs",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (overrides config)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use for training (overrides config)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for outputs",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name used for log/metric file prefixes",
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    experiment_name = args.experiment or Path(args.config).stem

    logger = SystemLogger.configure(
        LoggingConfig(
            experiment_name=experiment_name,
            output_dir=Path(args.output_dir) / "logs",
            level="DEBUG" if args.verbose else "INFO",
        )
    )

    try:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error("Config file not found: %s", config_path)
            return 1

        logger.info("=" * 80)
        logger.info("Adaptive Federated Learning Framework (AFLF)")
        logger.info("=" * 80)
        logger.info("Config: %s", config_path)
        logger.info("Output directory: %s", args.output_dir)
        logger.info("Experiment: %s", experiment_name)
        if args.seed is not None:
            logger.info("Random seed override: %s", args.seed)
        if args.device is not None:
            logger.info("Device override: %s", args.device)
        logger.info("=" * 80)

        env_report = EnvironmentChecker.check()
        logger.info(
            "Environment: python=%s platform=%s torch=%s cuda=%s mps=%s",
            env_report.python_version,
            env_report.platform,
            env_report.has_torch,
            env_report.has_cuda,
            env_report.has_mps,
        )

        dependency_issues = DependencyChecker.validate("requirements.txt")
        if dependency_issues:
            logger.warning("Dependency mismatches detected:")
            for issue in dependency_issues:
                logger.warning("- %s", issue)

        config = load_and_validate_config(
            config_path=config_path,
            validator=ConfigValidator(),
        )

        if args.seed is not None:
            config["seed"] = args.seed
        if args.rounds is not None:
            config.setdefault("federated", {})["num_rounds"] = int(args.rounds)

        ExperimentSeedManager.set_seed(int(config.get("seed", 42)), deterministic=True)

        federated_config = build_federated_config(config)

        if args.seed is not None:
            federated_config.seed = args.seed
        if args.rounds is not None:
            federated_config.num_rounds = int(args.rounds)

        if args.device is not None:
            federated_config.device = args.device

        trainer = FederatedTrainer.from_components(
            data_config=config["data"],
            model_config=config["model"],
            federated_config=federated_config,
            selection_config=config.get("selection"),
            experiment_name=experiment_name,
            metrics_output_dir=str(Path(args.output_dir) / "metrics"),
        )

        logger.info("Starting federated training loop...")
        results = trainer.fit()

        summary = results.get("summary", {})
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info("Rounds: %s", summary.get("num_rounds", 0))
        logger.info("Total time: %.2fs", summary.get("total_training_time", 0.0))
        logger.info(
            "Final global accuracy: %.4f",
            summary.get("final_global_accuracy", 0.0),
        )
        logger.info("Final global loss: %.4f", summary.get("final_global_loss", 0.0))
        if "evaluation_logs" in results:
            logger.info("Evaluation logs:")
            for name, path in results["evaluation_logs"].items():
                logger.info("  - %s: %s", name, path)
        logger.info("=" * 80)
        return 0
    except Exception as exc:
        logging.getLogger("aflf.main").exception("Fatal execution error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
