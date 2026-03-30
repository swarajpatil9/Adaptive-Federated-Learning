"""
Main entry point for federated learning experiments.

Usage:
    python main.py --config configs/baseline.yaml
"""

import argparse
import sys
from pathlib import Path

from aflf.training import (
    FederatedTrainer,
    build_federated_config,
    load_yaml_config,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Adaptive Federated Learning Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
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

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Verify config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print("=" * 80)
    print("Adaptive Federated Learning Framework (AFLF)")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Output directory: {args.output_dir}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    if args.device is not None:
        print(f"Device: {args.device}")
    print("=" * 80)

    config = load_yaml_config(str(config_path))

    if 'data' not in config or 'model' not in config:
        print("Error: Config must contain 'data' and 'model' sections.")
        sys.exit(1)

    federated_config = build_federated_config(config)

    if args.seed is not None:
        federated_config.seed = args.seed

    if args.device is not None:
        federated_config.device = args.device

    trainer = FederatedTrainer.from_components(
        data_config=config['data'],
        model_config=config['model'],
        federated_config=federated_config,
        selection_config=config.get('selection'),
        experiment_name=config_path.stem,
        metrics_output_dir=str(Path(args.output_dir) / 'metrics'),
    )

    print("\nStarting Phase 9 federated training loop...")
    results = trainer.fit()

    summary = results['summary']
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Rounds: {summary.get('num_rounds', 0)}")
    print(f"Total time: {summary.get('total_training_time', 0.0):.2f}s")
    print(f"Final global accuracy: {summary.get('final_global_accuracy', 0.0):.4f}")
    print(f"Final global loss: {summary.get('final_global_loss', 0.0):.4f}")
    if 'evaluation_logs' in results:
        print("Evaluation logs:")
        for name, path in results['evaluation_logs'].items():
            print(f"  - {name}: {path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
