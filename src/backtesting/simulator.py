"""
Rolling window backtesting simulator.

Implements the core simulation loop for training and testing
trading strategies across rolling time windows.
"""

from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import torch
import numpy as np

from models.base import BaseModel
from models.factory import create_model
from data.dataset import ResidualDataset, RollingWindowDataset
from data.loader import DataLoader
from preprocessing import preprocess_data
from portfolio.weights import normalize_weights, apply_weight_policy, compute_portfolio_returns
from training.trainer import Trainer
from training.loss import compute_stats
from backtesting.results import BacktestResults, SubperiodResult
from dtypes import (
    BacktestConfig, TrainingConfig, ModelConfig, Objective,
    PreprocessingMethod, WeightPolicy
)
from utils.logging import get_logger
from utils.device import get_device, set_random_seeds

logger = get_logger(__name__)


class Simulator:
    """
    Rolling window backtesting simulator.

    Orchestrates the complete backtesting process:
    1. Rolling window data splitting
    2. Model training (or loading) for each period
    3. Weight computation and transformation
    4. Return calculation with transaction costs
    5. Results aggregation
    """

    def __init__(
        self,
        backtest_config: BacktestConfig,
        training_config: TrainingConfig,
        model_config: ModelConfig,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the simulator.

        Args:
            backtest_config: Backtesting parameters
            training_config: Training parameters
            model_config: Model architecture parameters
            device: Compute device (auto-detected if None)
        """
        self.backtest_config = backtest_config
        self.training_config = training_config
        self.model_config = model_config
        self.device = device or get_device()

        # Weight policy
        self.weight_policy = WeightPolicy()

        # Results
        self.results = BacktestResults()

        logger.info(f"Simulator initialized on device: {self.device}")

    def run(
        self,
        residuals: torch.Tensor,
        composition_matrix: Optional[torch.Tensor] = None,
        preprocessing: PreprocessingMethod = PreprocessingMethod.CUMSUM,
        models_dir: Optional[Path] = None,
        test_only: bool = False
    ) -> BacktestResults:
        """
        Run the full backtesting simulation.

        Args:
            residuals: Residual returns (T, N)
            composition_matrix: Optional composition matrix (T, N_res, N_assets)
            preprocessing: Preprocessing method
            models_dir: Directory for saving/loading models
            test_only: If True, load pre-trained models instead of training

        Returns:
            BacktestResults with all subperiod results
        """
        # Create rolling window dataset
        rolling_dataset = RollingWindowDataset(
            residuals=residuals,
            lookback=self.model_config.lookback,
            train_length=self.backtest_config.length_training,
            stride=self.backtest_config.stride,
            preprocessing=preprocessing,
            composition_matrix=composition_matrix,
            initial_t=self.backtest_config.initial_t
        )

        num_subperiods = len(rolling_dataset)
        logger.info(f"Starting simulation with {num_subperiods} subperiods")

        # Store config in results
        self.results.config = {
            "train_length": self.backtest_config.length_training,
            "stride": self.backtest_config.stride,
            "lookback": self.model_config.lookback,
            "model_class": self.model_config.model_class,
            "num_subperiods": num_subperiods,
        }

        # Run each subperiod
        for t, (train_dataset, test_dataset) in rolling_dataset:
            logger.info(f"Processing subperiod {t + 1}/{num_subperiods}")

            result = self._run_subperiod(
                t,
                train_dataset,
                test_dataset,
                models_dir,
                test_only
            )

            self.results.add_subperiod(result)

            # Log progress
            logger.info(
                f"Subperiod {t + 1} complete: "
                f"Sharpe={result.sharpe_ratio:.4f}, "
                f"Return={result.mean_return:.4f}"
            )

        # Compute and log final stats
        final_stats = self.results.compute_aggregate_stats()
        logger.info(f"Simulation complete: Sharpe={final_stats['sharpe_ratio']:.4f}")

        return self.results

    def _run_subperiod(
        self,
        subperiod_idx: int,
        train_dataset: ResidualDataset,
        test_dataset: ResidualDataset,
        models_dir: Optional[Path],
        test_only: bool
    ) -> SubperiodResult:
        """
        Run training and testing for a single subperiod.

        Args:
            subperiod_idx: Subperiod index
            train_dataset: Training data
            test_dataset: Test data
            models_dir: Directory for model checkpoints
            test_only: Whether to skip training

        Returns:
            SubperiodResult for this period
        """
        # Create model
        model = create_model(
            self.model_config.model_class,
            lookback=self.model_config.lookback,
            filter_numbers=self.model_config.filter_numbers,
            attention_heads=self.model_config.attention_heads,
            hidden_units_factor=self.model_config.hidden_units_factor,
            dropout=self.model_config.dropout,
            filter_size=self.model_config.filter_size,
            random_seed=self.training_config.random_seed,
        )

        model_path = None
        if models_dir is not None:
            model_path = models_dir / f"model_subperiod_{subperiod_idx}.pt"

        train_loss = 0.0

        if test_only and model_path and model_path.exists():
            # Load pre-trained model
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(self.device)
            logger.info(f"Loaded model from {model_path}")

        elif model.is_trainable:
            # Train model
            trainer = Trainer(
                model=model,
                config=self.training_config,
                device=self.device
            )

            train_result = trainer.train(
                train_dataset,
                val_dataset=test_dataset
            )

            train_loss = train_result["train_losses"][-1] if train_result["train_losses"] else 0.0

            # Save model if requested
            if model_path is not None:
                trainer.save_checkpoint(model_path)

        else:
            # Non-trainable model (e.g., OUThreshold)
            model = model.to(self.device)

        # Generate predictions on test set
        weights = self._predict(model, test_dataset)

        # Apply weight policy and normalize
        norm_weights, asset_weights = apply_weight_policy(
            weights,
            policy_name=self.weight_policy.name.value,
            policy_type=self.weight_policy.policy_type.value,
            composition_matrix=test_dataset.composition_matrix[test_dataset.lookback:]
            if test_dataset.composition_matrix is not None else None,
            window=self.weight_policy.window
        )

        # Compute returns
        test_residuals = test_dataset.residuals[test_dataset.lookback:]
        returns, turnovers, short_props = compute_portfolio_returns(
            norm_weights,
            test_residuals,
            transaction_cost=self.backtest_config.transaction_cost,
            holding_cost=self.backtest_config.holding_cost,
            holding_days=self.backtest_config.holding_days
        )

        # Convert to numpy
        returns_np = returns.cpu().numpy()
        weights_np = norm_weights.cpu().numpy()
        turnovers_np = turnovers.cpu().numpy()
        short_props_np = short_props.cpu().numpy()

        # Compute stats
        stats = compute_stats(returns, self.backtest_config.holding_days)

        return SubperiodResult(
            subperiod_idx=subperiod_idx,
            returns=returns_np,
            weights=weights_np,
            turnovers=turnovers_np,
            short_proportions=short_props_np,
            sharpe_ratio=stats["sharpe_ratio"],
            mean_return=stats["mean_return"],
            volatility=stats["volatility"],
            train_loss=train_loss
        )

    def _predict(
        self,
        model: BaseModel,
        dataset: ResidualDataset
    ) -> torch.Tensor:
        """
        Generate weight predictions from model.

        Args:
            model: Trained model
            dataset: Dataset to predict on

        Returns:
            Predicted weights (T, N)
        """
        model.eval()

        with torch.no_grad():
            features = dataset.features.to(self.device)
            valid_mask = dataset.valid_mask.to(self.device)

            T, N, _ = features.shape
            all_weights = torch.zeros(T, N, device=self.device)

            old_weights = None

            for t in range(T):
                valid_idx = valid_mask[t]
                if valid_idx.sum() == 0:
                    continue

                x = features[t, valid_idx]

                if model.is_frictions_model:
                    old_w = old_weights[valid_idx] if old_weights is not None else None
                    w = model(x, old_w)
                else:
                    w = model(x)

                all_weights[t, valid_idx] = w

                if model.is_frictions_model:
                    old_weights = all_weights[t].clone()

        return all_weights

    def set_weight_policy(
        self,
        policy_name: str = "normal",
        policy_type: str = "residuals",
        window: int = 6
    ) -> None:
        """
        Set the weight transformation policy.

        Args:
            policy_name: Policy name ('normal', 'moving_average', 'sparse_percent')
            policy_type: Apply in 'residuals' or 'assets' space
            window: Window size for moving average
        """
        from dtypes import WeightPolicyName, WeightPolicyType

        self.weight_policy = WeightPolicy(
            name=WeightPolicyName(policy_name),
            policy_type=WeightPolicyType(policy_type),
            window=window
        )


def run_simulation(
    data_dir: Path,
    residuals_dir: Path,
    factor_model: str,
    num_factors: int,
    config: Dict[str, Any],
    output_dir: Optional[Path] = None
) -> BacktestResults:
    """
    Convenience function to run a complete simulation.

    Args:
        data_dir: Path to data directory
        residuals_dir: Path to residuals directory
        factor_model: Factor model name
        num_factors: Number of factors
        config: Configuration dictionary
        output_dir: Optional output directory for results

    Returns:
        BacktestResults
    """
    # Load data
    data_loader = DataLoader(data_dir, residuals_dir)
    residuals = data_loader.get_residuals(factor_model, num_factors)
    comp_mtx = data_loader.get_composition_matrix(factor_model, num_factors)

    # Create configs
    backtest_config = BacktestConfig.from_dict(config)
    training_config = TrainingConfig.from_dict(config)
    model_config = ModelConfig.from_dict(config.get("model", {}))

    # Create and run simulator
    simulator = Simulator(
        backtest_config=backtest_config,
        training_config=training_config,
        model_config=model_config
    )

    # Set weight policy if specified
    if "weight_policy" in config:
        wp = config["weight_policy"]
        simulator.set_weight_policy(
            policy_name=wp.get("name", "normal"),
            policy_type=wp.get("type", "residuals"),
            window=wp.get("window", 6)
        )

    # Run
    results = simulator.run(
        residuals=residuals,
        composition_matrix=comp_mtx,
        preprocessing=PreprocessingMethod(config.get("preprocessing", "cumsum"))
    )

    # Save results if output dir specified
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results.save(output_dir / f"results_{factor_model}_{num_factors}")

    return results
