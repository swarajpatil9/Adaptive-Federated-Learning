"""
Logging system for federated learning server.

Provides structured logging for:
- Round execution
- Client participation
- Training metrics
- System events

Supports multiple outputs:
- Console (with progress bars)
- File (structured logs)
- TensorBoard (metrics visualization)
- JSON/CSV (data export)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from .round_manager import RoundState


class ServerLogger:
    """
    Structured logger for federated server.

    Handles logging at multiple levels:
    - INFO: Round start/end, major events
    - DEBUG: Detailed client operations
    - WARNING: Failures, degraded performance
    - ERROR: Critical failures

    Example:
        >>> logger = ServerLogger(
        ...     experiment_name="mnist_fedavg",
        ...     log_dir="logs",
        ...     console_level=logging.INFO
        ... )
        >>> logger.log_round_start(round_num=0, num_selected=10)
        >>> logger.log_round_end(round_state, metrics)
    """

    def __init__(
        self,
        experiment_name: str = "federated_learning",
        log_dir: str = "logs",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        enable_file_logging: bool = True,
    ):
        """
        Initialize server logger.

        Args:
            experiment_name: Name of experiment (used for log filenames)
            log_dir: Directory for log files
            console_level: Logging level for console output
            file_level: Logging level for file output
            enable_file_logging: Whether to write logs to file
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.enable_file_logging = enable_file_logging

        # Create log directory
        if enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup Python logger
        self.logger = logging.getLogger(f"aflf.server.{experiment_name}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()  # Remove existing handlers

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        if enable_file_logging:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(file_level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            self.logger.info(f"Logging to file: {log_file}")

    def log_server_init(
        self,
        num_clients: int,
        model_name: str,
        selection_strategy: str,
        config: Optional[Dict] = None,
    ) -> None:
        """
        Log server initialization.

        Args:
            num_clients: Total number of clients
            model_name: Model architecture name
            selection_strategy: Client selection strategy
            config: Additional configuration parameters
        """
        self.logger.info("=" * 70)
        self.logger.info("FEDERATED SERVER INITIALIZED")
        self.logger.info("=" * 70)
        self.logger.info(f"Experiment:          {self.experiment_name}")
        self.logger.info(f"Model:               {model_name}")
        self.logger.info(f"Total clients:       {num_clients}")
        self.logger.info(f"Selection strategy:  {selection_strategy}")

        if config:
            self.logger.info("\nConfiguration:")
            for key, value in config.items():
                self.logger.info(f"  {key}: {value}")

        self.logger.info("=" * 70)

    def log_round_start(
        self, round_num: int, num_selected: int, selected_clients: Optional[List[int]] = None
    ) -> None:
        """
        Log round start.

        Args:
            round_num: Round number
            num_selected: Number of selected clients
            selected_clients: List of selected client IDs (optional)
        """
        self.logger.info("\n" + "-" * 70)
        self.logger.info(f"ROUND {round_num} STARTED")
        self.logger.info("-" * 70)
        self.logger.info(f"Selected {num_selected} clients for training")

        if selected_clients is not None:
            self.logger.debug(f"Selected client IDs: {selected_clients}")

    def log_client_start(self, round_num: int, client_id: int) -> None:
        """
        Log client training start.

        Args:
            round_num: Round number
            client_id: Client ID
        """
        self.logger.debug(f"Round {round_num}: Client {client_id} started training")

    def log_client_success(
        self, round_num: int, client_id: int, train_loss: float, train_acc: float, duration: float
    ) -> None:
        """
        Log successful client training.

        Args:
            round_num: Round number
            client_id: Client ID
            train_loss: Training loss
            train_acc: Training accuracy
            duration: Training duration (seconds)
        """
        self.logger.debug(
            f"Round {round_num}: Client {client_id} completed "
            f"(loss={train_loss:.4f}, acc={train_acc:.4f}, time={duration:.2f}s)"
        )

    def log_client_failure(
        self, round_num: int, client_id: int, reason: str
    ) -> None:
        """
        Log client failure.

        Args:
            round_num: Round number
            client_id: Client ID
            reason: Failure reason
        """
        self.logger.warning(
            f"Round {round_num}: Client {client_id} failed - {reason}"
        )

    def log_round_end(
        self, round_state: RoundState, metrics: Dict
    ) -> None:
        """
        Log round completion with metrics.

        Args:
            round_state: Round state object
            metrics: Dictionary of metrics
        """
        self.logger.info("\n" + "-" * 70)
        self.logger.info(f"ROUND {round_state.round_num} COMPLETED")
        self.logger.info("-" * 70)

        # Participation statistics
        self.logger.info(
            f"Participation:  {len(round_state.participating_clients)}/{len(round_state.selected_clients)} "
            f"clients ({round_state.participation_rate:.1%})"
        )

        if round_state.dropped_clients:
            self.logger.info(f"Failed clients: {len(round_state.dropped_clients)}")
            self.logger.debug(f"Failed client IDs: {round_state.dropped_clients}")

        # Timing
        self.logger.info(f"Duration:       {round_state.duration:.2f}s")

        # Metrics
        if metrics:
            self.logger.info("\nMetrics:")
            if 'avg_train_loss' in metrics:
                self.logger.info(f"  Train Loss:     {metrics['avg_train_loss']:.4f}")
            if 'avg_train_accuracy' in metrics:
                self.logger.info(f"  Train Accuracy: {metrics['avg_train_accuracy']:.4f}")
            if 'avg_val_loss' in metrics and metrics['avg_val_loss'] is not None:
                self.logger.info(f"  Val Loss:       {metrics['avg_val_loss']:.4f}")
            if 'avg_val_accuracy' in metrics and metrics['avg_val_accuracy'] is not None:
                self.logger.info(f"  Val Accuracy:   {metrics['avg_val_accuracy']:.4f}")
            if 'total_samples' in metrics:
                self.logger.info(f"  Total Samples:  {metrics['total_samples']}")

        self.logger.info("-" * 70)

    def log_training_complete(
        self,
        total_rounds: int,
        total_time: float,
        final_metrics: Optional[Dict] = None,
    ) -> None:
        """
        Log training completion summary.

        Args:
            total_rounds: Total number of rounds completed
            total_time: Total training time (seconds)
            final_metrics: Final model metrics (optional)
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("FEDERATED TRAINING COMPLETED")
        self.logger.info("=" * 70)
        self.logger.info(f"Total rounds:   {total_rounds}")
        self.logger.info(f"Total time:     {total_time:.2f}s ({total_time / 60:.2f}m)")
        self.logger.info(f"Avg time/round: {total_time / total_rounds:.2f}s")

        if final_metrics:
            self.logger.info("\nFinal Metrics:")
            for key, value in final_metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"  {key}: {value:.4f}")
                else:
                    self.logger.info(f"  {key}: {value}")

        self.logger.info("=" * 70)

    def log_checkpoint_save(self, round_num: int, checkpoint_path: str) -> None:
        """
        Log checkpoint save.

        Args:
            round_num: Round number
            checkpoint_path: Path to checkpoint file
        """
        self.logger.info(f"Saved checkpoint for round {round_num}: {checkpoint_path}")

    def log_warning(self, message: str) -> None:
        """
        Log warning message.

        Args:
            message: Warning message
        """
        self.logger.warning(message)

    def log_error(self, message: str) -> None:
        """
        Log error message.

        Args:
            message: Error message
        """
        self.logger.error(message)

    def log_info(self, message: str) -> None:
        """
        Log info message.

        Args:
            message: Info message
        """
        self.logger.info(message)

    def log_debug(self, message: str) -> None:
        """
        Log debug message.

        Args:
            message: Debug message
        """
        self.logger.debug(message)


class ConsoleProgressLogger:
    """
    Simple console progress logger for federated rounds.

    Displays progress bars and real-time statistics without
    external dependencies (no tqdm).

    Example:
        >>> progress = ConsoleProgressLogger(total_rounds=10)
        >>> for round_num in range(10):
        ...     progress.update_round(round_num, metrics={'loss': 0.5})
        >>> progress.finish()
    """

    def __init__(self, total_rounds: int, bar_width: int = 50):
        """
        Initialize progress logger.

        Args:
            total_rounds: Total number of rounds
            bar_width: Width of progress bar in characters
        """
        self.total_rounds = total_rounds
        self.bar_width = bar_width
        self.start_time = time.time()
        self.round_times = []

    def update_round(
        self, round_num: int, metrics: Dict, num_participating: int, num_selected: int
    ) -> None:
        """
        Update progress for current round.

        Args:
            round_num: Current round number (0-indexed)
            metrics: Round metrics dictionary
            num_participating: Number of participating clients
            num_selected: Number of selected clients
        """
        # Calculate progress
        progress = (round_num + 1) / self.total_rounds
        filled = int(self.bar_width * progress)
        bar = "█" * filled + "░" * (self.bar_width - filled)

        # Calculate timing
        elapsed = time.time() - self.start_time
        rounds_completed = round_num + 1
        avg_time_per_round = elapsed / rounds_completed
        eta = avg_time_per_round * (self.total_rounds - rounds_completed)

        # Format metrics
        metric_str = ""
        if 'avg_train_loss' in metrics:
            metric_str += f"Loss: {metrics['avg_train_loss']:.4f} "
        if 'avg_train_accuracy' in metrics:
            metric_str += f"Acc: {metrics['avg_train_accuracy']:.4f}"

        # Print progress
        print(
            f"\rRound {rounds_completed}/{self.total_rounds} "
            f"|{bar}| "
            f"{progress:.0%} "
            f"[{num_participating}/{num_selected} clients] "
            f"{metric_str} "
            f"ETA: {eta:.0f}s",
            end="",
            flush=True,
        )

    def finish(self) -> None:
        """Print final newline."""
        print()  # New line after progress bar
