"""
Logging utilities for the DLSA package.

Provides consistent logging configuration across the package.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


_loggers: dict = {}


def setup_logger(
    name: str = "dlsa",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional path to log file
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # Default format
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def get_logger(name: str = "dlsa") -> logging.Logger:
    """
    Get an existing logger or create a new one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]


class TrainingLogger:
    """
    Specialized logger for training progress.

    Provides methods for logging training metrics with
    consistent formatting.
    """

    def __init__(self, name: str = "dlsa.training", log_file: Optional[Path] = None):
        """
        Initialize training logger.

        Args:
            name: Logger name
            log_file: Optional path for log file
        """
        self.logger = setup_logger(name, log_file=log_file)
        self.start_time: Optional[datetime] = None

    def start_training(self, num_epochs: int, model_name: str) -> None:
        """
        Log training start.

        Args:
            num_epochs: Total epochs to train
            model_name: Name of model being trained
        """
        self.start_time = datetime.now()
        self.logger.info(f"Starting training: {model_name} for {num_epochs} epochs")

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_sharpe: Optional[float] = None,
        val_sharpe: Optional[float] = None
    ) -> None:
        """
        Log epoch results.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Optional validation loss
            train_sharpe: Optional training Sharpe ratio
            val_sharpe: Optional validation Sharpe ratio
        """
        msg = f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f}"

        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:.6f}"
        if train_sharpe is not None:
            msg += f" | Train Sharpe: {train_sharpe:.4f}"
        if val_sharpe is not None:
            msg += f" | Val Sharpe: {val_sharpe:.4f}"

        self.logger.info(msg)

    def log_subperiod(
        self,
        subperiod: int,
        total_subperiods: int,
        sharpe: float,
        mean_return: float
    ) -> None:
        """
        Log subperiod results during backtesting.

        Args:
            subperiod: Current subperiod index
            total_subperiods: Total number of subperiods
            sharpe: Sharpe ratio for subperiod
            mean_return: Mean return for subperiod
        """
        self.logger.info(
            f"Subperiod {subperiod:3d}/{total_subperiods} | "
            f"Sharpe: {sharpe:.4f} | Mean Return: {mean_return:.6f}"
        )

    def end_training(self, final_sharpe: float) -> None:
        """
        Log training completion.

        Args:
            final_sharpe: Final Sharpe ratio
        """
        if self.start_time is not None:
            duration = datetime.now() - self.start_time
            self.logger.info(
                f"Training complete | Duration: {duration} | Final Sharpe: {final_sharpe:.4f}"
            )
        else:
            self.logger.info(f"Training complete | Final Sharpe: {final_sharpe:.4f}")
