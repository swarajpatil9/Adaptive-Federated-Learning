"""
Main entry point for federated learning experiments.

Usage:
    python main.py --config configs/baseline.yaml
    python main.py --config configs/adaptive.yaml --seed 42
"""

import argparse
import sys
from pathlib import Path


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

    print("="*80)
    print("Adaptive Federated Learning Framework (AFLF)")
    print("="*80)
    print(f"Config: {config_path}")
    print(f"Output directory: {args.output_dir}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    if args.device is not None:
        print(f"Device: {args.device}")
    print("="*80)

    # TODO: Load config, initialize system, run federated learning
    print("\n[PHASE 1 COMPLETE] Project skeleton ready.")
    print("Next phases will implement the full FL pipeline.\n")


if __name__ == "__main__":
    main()
