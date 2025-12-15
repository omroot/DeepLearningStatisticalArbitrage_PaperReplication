"""
Configuration management for DLSA.

Handles loading, saving, and managing configuration from YAML files.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import yaml

from dtypes import (
    TrainingConfig, BacktestConfig, ModelConfig,
    WeightPolicy, Objective, PreprocessingMethod
)


@dataclass
class Config:
    """
    Complete configuration for a DLSA experiment.

    Combines all configuration aspects into a single structure.
    """
    # Paths
    data_dir: str = "./data"
    residuals_dir: str = "./residuals"
    results_dir: str = "./results"
    models_dir: str = "./models"

    # Factor model
    factor_model: str = "IPCA"
    num_factors: int = 5

    # Preprocessing
    preprocessing: str = "cumsum"

    # Training
    num_epochs: int = 100
    batch_size: int = 125
    learning_rate: float = 0.001
    optimizer_name: str = "Adam"
    validation_freq: int = 10
    objective: str = "sharpe"
    random_seed: int = 0

    # Backtest
    length_training: int = 1000
    stride: int = 125
    initial_t: int = 0
    rolling_retrain: bool = True
    holding_days: int = 1
    transaction_cost: float = 0.0
    holding_cost: float = 0.0

    # Model
    model_class: str = "CNNTransformer"
    lookback: int = 30
    filter_numbers: List[int] = field(default_factory=lambda: [1, 8])
    attention_heads: int = 4
    hidden_units_factor: int = 2
    dropout: float = 0.25
    filter_size: int = 2

    # Weight policy
    weight_policy_name: str = "normal"
    weight_policy_type: str = "residuals"
    weight_policy_window: int = 6

    # Device
    device: str = "auto"

    # Experiment
    experiment_name: str = "default"
    debug: bool = False

    def get_training_config(self) -> TrainingConfig:
        """Get TrainingConfig from this config."""
        return TrainingConfig(
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            optimizer_name=self.optimizer_name,
            validation_freq=self.validation_freq,
            objective=Objective(self.objective),
            random_seed=self.random_seed
        )

    def get_backtest_config(self) -> BacktestConfig:
        """Get BacktestConfig from this config."""
        return BacktestConfig(
            length_training=self.length_training,
            stride=self.stride,
            initial_t=self.initial_t,
            rolling_retrain=self.rolling_retrain,
            holding_days=self.holding_days,
            transaction_cost=self.transaction_cost,
            holding_cost=self.holding_cost
        )

    def get_model_config(self) -> ModelConfig:
        """Get ModelConfig from this config."""
        return ModelConfig(
            model_class=self.model_class,
            lookback=self.lookback,
            filter_numbers=self.filter_numbers,
            attention_heads=self.attention_heads,
            hidden_units_factor=self.hidden_units_factor,
            dropout=self.dropout,
            filter_size=self.filter_size
        )

    def get_weight_policy(self) -> WeightPolicy:
        """Get WeightPolicy from this config."""
        from dtypes import WeightPolicyName, WeightPolicyType
        return WeightPolicy(
            name=WeightPolicyName(self.weight_policy_name),
            policy_type=WeightPolicyType(self.weight_policy_type),
            window=self.weight_policy_window
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """Create from dictionary."""
        # Handle nested configs from original format
        if "model" in d:
            model = d.pop("model")
            d["model_class"] = model.get("class", "CNNTransformer")
            d["lookback"] = model.get("lookback", 30)
            d["filter_numbers"] = model.get("filter_numbers", [1, 8])
            d["attention_heads"] = model.get("attention_heads", 4)
            d["hidden_units_factor"] = model.get("hidden_units_factor", 2)
            d["dropout"] = model.get("dropout", 0.25)
            d["filter_size"] = model.get("filter_size", 2)

        if "weight_policy" in d:
            wp = d.pop("weight_policy")
            d["weight_policy_name"] = wp.get("name", "normal")
            d["weight_policy_type"] = wp.get("type", "residuals")
            d["weight_policy_window"] = wp.get("window", 6)

        if "optimizer_opts" in d:
            opts = d.pop("optimizer_opts")
            d["learning_rate"] = opts.get("lr", 0.001)

        # Map old names to new
        name_map = {
            "trans_cost": "transaction_cost",
            "hold_cost": "holding_cost",
        }
        for old, new in name_map.items():
            if old in d and new not in d:
                d[new] = d.pop(old)

        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}

        return cls(**filtered)

    def save(self, path: Path) -> None:
        """Save config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load config from YAML file."""
        path = Path(path)

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)


def load_config(path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML file.

    Args:
        path: Path to config file

    Returns:
        Config object
    """
    return Config.load(Path(path))


def save_config(config: Config, path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Config object
        path: Path to save file
    """
    config.save(Path(path))


def create_default_config(
    experiment_name: str = "default",
    **overrides
) -> Config:
    """
    Create a default configuration with optional overrides.

    Args:
        experiment_name: Name for the experiment
        **overrides: Configuration values to override

    Returns:
        Config object
    """
    config = Config(experiment_name=experiment_name)

    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def merge_configs(
    base: Config,
    override: Dict[str, Any]
) -> Config:
    """
    Merge override values into a base config.

    Args:
        base: Base configuration
        override: Dictionary of values to override

    Returns:
        New Config with merged values
    """
    base_dict = base.to_dict()
    base_dict.update(override)
    return Config.from_dict(base_dict)
