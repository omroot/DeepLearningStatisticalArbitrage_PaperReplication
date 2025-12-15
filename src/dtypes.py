"""
Core type definitions and data classes for the DLSA package.

This module defines all the fundamental types, enums, and dataclasses
used throughout the codebase.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path

import numpy as np
import torch


class Device(Enum):
    """Computation device enumeration."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class Objective(Enum):
    """Optimization objective for training."""
    SHARPE = "sharpe"
    MEAN_VARIANCE = "meanvar"


class WeightPolicyType(Enum):
    """Type of weight policy application."""
    RESIDUALS = "residuals"
    ASSETS = "assets"


class WeightPolicyName(Enum):
    """Name of weight policy to apply."""
    NORMAL = "normal"
    MOVING_AVERAGE = "moving_average"
    SPARSE_PERCENT = "sparse_percent"


class PreprocessingMethod(Enum):
    """Preprocessing method for time series."""
    CUMSUM = "cumsum"
    FOURIER = "fourier"
    OU = "ou"


class FactorModelType(Enum):
    """Type of factor model for residual computation."""
    IPCA = "IPCA"
    PCA = "PCA"
    FAMA_FRENCH = "FamaFrench"


@dataclass
class WeightPolicy:
    """
    Configuration for portfolio weight policy.

    Attributes:
        policy_type: Whether to apply policy in residual or asset space
        name: The specific policy to apply
        window: Window size for moving average policy
    """
    policy_type: WeightPolicyType = WeightPolicyType.RESIDUALS
    name: WeightPolicyName = WeightPolicyName.NORMAL
    window: int = 6

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WeightPolicy":
        """Create WeightPolicy from configuration dictionary."""
        return cls(
            policy_type=WeightPolicyType(d.get("type", "residuals")),
            name=WeightPolicyName(d.get("name", "normal")),
            window=d.get("window", 6)
        )


@dataclass
class TrainingConfig:
    """
    Configuration for model training.

    Attributes:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        optimizer_name: Name of optimizer to use
        validation_freq: Frequency of validation checks
        objective: Optimization objective
        random_seed: Random seed for reproducibility
    """
    num_epochs: int = 100
    batch_size: int = 125
    learning_rate: float = 0.001
    optimizer_name: str = "Adam"
    validation_freq: int = 10
    objective: Objective = Objective.SHARPE
    random_seed: int = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        """Create TrainingConfig from configuration dictionary."""
        return cls(
            num_epochs=d.get("num_epochs", 100),
            batch_size=d.get("batch_size", 125),
            learning_rate=d.get("optimizer_opts", {}).get("lr", 0.001),
            optimizer_name=d.get("optimizer_name", "Adam"),
            validation_freq=d.get("validation_freq", 10),
            objective=Objective(d.get("objective", "sharpe")),
            random_seed=d.get("random_seed", 0)
        )


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting.

    Attributes:
        length_training: Number of days in training window
        stride: Number of days to roll forward
        initial_t: Starting subperiod index
        rolling_retrain: Whether to retrain at each roll
        holding_days: Number of days to hold positions
        transaction_cost: Transaction cost in basis points
        holding_cost: Shorting cost in basis points
    """
    length_training: int = 1000
    stride: int = 125
    initial_t: int = 0
    rolling_retrain: bool = True
    holding_days: int = 1
    transaction_cost: float = 0.0
    holding_cost: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BacktestConfig":
        """Create BacktestConfig from configuration dictionary."""
        return cls(
            length_training=d.get("length_training", 1000),
            stride=d.get("stride", 125),
            initial_t=d.get("initial_t", 0),
            rolling_retrain=d.get("rolling_retrain", True),
            holding_days=d.get("holding_days", 1),
            transaction_cost=d.get("trans_cost", 0.0),
            holding_cost=d.get("hold_cost", 0.0)
        )


@dataclass
class ModelConfig:
    """
    Configuration for neural network model.

    Attributes:
        model_class: Name of model class to instantiate
        lookback: Lookback window for features
        filter_numbers: CNN filter sizes
        attention_heads: Number of attention heads
        hidden_units_factor: FFN hidden units multiplier
        dropout: Dropout rate
        filter_size: CNN kernel size
        use_convolution: Whether to use CNN layers
        use_transformer: Whether to use transformer layers
    """
    model_class: str = "CNNTransformer"
    lookback: int = 30
    filter_numbers: List[int] = field(default_factory=lambda: [1, 8])
    attention_heads: int = 4
    hidden_units_factor: int = 2
    dropout: float = 0.25
    filter_size: int = 2
    use_convolution: bool = True
    use_transformer: bool = True
    hidden_units: List[int] = field(default_factory=lambda: [30, 16, 8, 4])

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from configuration dictionary."""
        return cls(
            model_class=d.get("class", "CNNTransformer"),
            lookback=d.get("lookback", 30),
            filter_numbers=d.get("filter_numbers", [1, 8]),
            attention_heads=d.get("attention_heads", 4),
            hidden_units_factor=d.get("hidden_units_factor", 2),
            dropout=d.get("dropout", 0.25),
            filter_size=d.get("filter_size", 2),
            use_convolution=d.get("use_convolution", True),
            use_transformer=d.get("use_transformer", True),
            hidden_units=d.get("hidden_units", [30, 16, 8, 4])
        )


@dataclass
class DataPaths:
    """
    Paths to data directories.

    Attributes:
        data_dir: Root data directory
        residuals_dir: Directory containing residual files
        results_dir: Directory for output results
        models_dir: Directory for saved models
    """
    data_dir: Path
    residuals_dir: Path
    results_dir: Path = field(default_factory=lambda: Path("./results"))
    models_dir: Path = field(default_factory=lambda: Path("./models"))

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.residuals_dir, str):
            self.residuals_dir = Path(self.residuals_dir)
        if isinstance(self.results_dir, str):
            self.results_dir = Path(self.results_dir)
        if isinstance(self.models_dir, str):
            self.models_dir = Path(self.models_dir)


@dataclass
class PerformanceMetrics:
    """
    Portfolio performance metrics.

    Attributes:
        returns: Array of period returns
        cumulative_returns: Cumulative return series
        sharpe_ratio: Annualized Sharpe ratio
        mean_return: Annualized mean return
        volatility: Annualized volatility
        max_drawdown: Maximum drawdown
        turnover: Average portfolio turnover
        short_proportion: Average short position proportion
    """
    returns: np.ndarray
    cumulative_returns: np.ndarray
    sharpe_ratio: float
    mean_return: float
    volatility: float
    max_drawdown: float
    turnover: float
    short_proportion: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "mean_return": self.mean_return,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
            "short_proportion": self.short_proportion
        }


@dataclass
class TrainingResult:
    """
    Result from a training run.

    Attributes:
        model_state: Trained model state dict
        optimizer_state: Optimizer state dict
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        best_epoch: Epoch with best validation loss
        final_sharpe: Final Sharpe ratio on validation
    """
    model_state: Dict[str, torch.Tensor]
    optimizer_state: Dict[str, Any]
    train_losses: List[float]
    val_losses: List[float]
    best_epoch: int
    final_sharpe: float


@dataclass
class BacktestResult:
    """
    Result from a backtest.

    Attributes:
        returns: Period-by-period returns
        weights: Portfolio weights over time
        turnovers: Turnover at each rebalance
        short_proportions: Short positions over time
        metrics: Computed performance metrics
        subperiod_results: Results for each rolling window
    """
    returns: np.ndarray
    weights: np.ndarray
    turnovers: np.ndarray
    short_proportions: np.ndarray
    metrics: PerformanceMetrics
    subperiod_results: List[Dict[str, Any]] = field(default_factory=list)


# Type aliases for common tensor types
TensorLike = Union[np.ndarray, torch.Tensor]
ArrayLike = Union[List[float], np.ndarray]
PathLike = Union[str, Path]
