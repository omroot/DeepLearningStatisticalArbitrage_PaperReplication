#!/usr/bin/env python3
"""
Main entry point for running DLSA simulations.

Usage:
    python run_simulation.py --config path/to/config.yaml
    python run_simulation.py --data-dir ./data --residuals-dir ./residuals
"""

import argparse
from pathlib import Path
import sys

import torch

from config import Config, load_config
from data.loader import DataLoader
from backtesting.simulator import Simulator
from dtypes import PreprocessingMethod
from utils.logging import setup_logger
from utils.device import get_device


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Deep Learning Statistical Arbitrage simulation"
    )

    # Config file
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML configuration file"
    )

    # Data paths (can override config)
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to data directory"
    )
    parser.add_argument(
        "--residuals-dir",
        type=str,
        help="Path to residuals directory"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Path to output results directory"
    )

    # Factor model
    parser.add_argument(
        "--factor-model",
        type=str,
        choices=["IPCA", "PCA", "FamaFrench"],
        help="Factor model type"
    )
    parser.add_argument(
        "--num-factors",
        type=int,
        help="Number of factors"
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        choices=["CNNTransformer", "FourierFFN", "OUThreshold"],
        help="Model architecture"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        help="Lookback window size"
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate"
    )

    # Backtest
    parser.add_argument(
        "--train-length",
        type=int,
        help="Training window length (days)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        help="Rolling window stride (days)"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        default="auto",
        help="Compute device"
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Load pre-trained models instead of training"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    logger = setup_logger("dlsa", log_file=Path("./cache/logs/simulation.log"))

    # Load or create config
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = Config()

    # Override config with command line arguments
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.residuals_dir:
        config.residuals_dir = args.residuals_dir
    if args.results_dir:
        config.results_dir = args.results_dir
    if args.factor_model:
        config.factor_model = args.factor_model
    if args.num_factors:
        config.num_factors = args.num_factors
    if args.model:
        config.model_class = args.model
    if args.lookback:
        config.lookback = args.lookback
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.train_length:
        config.length_training = args.train_length
    if args.stride:
        config.stride = args.stride
    if args.seed:
        config.random_seed = args.seed
    if args.debug:
        config.debug = True

    # Set device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")
    logger.info(f"Factor model: {config.factor_model} with {config.num_factors} factors")
    logger.info(f"Model: {config.model_class}")

    # Load data
    logger.info("Loading data...")
    data_loader = DataLoader(
        data_dir=Path(config.data_dir),
        residuals_dir=Path(config.residuals_dir),
        device=device
    )

    residuals = data_loader.get_residuals(config.factor_model, config.num_factors)
    logger.info(f"Loaded residuals with shape {residuals.shape}")

    try:
        comp_mtx = data_loader.get_composition_matrix(config.factor_model, config.num_factors)
        logger.info(f"Loaded composition matrix with shape {comp_mtx.shape}")
    except (FileNotFoundError, RuntimeError) as e:
        comp_mtx = None
        logger.warning(f"Composition matrix not loaded ({e}), using identity mapping")

    # Create simulator
    simulator = Simulator(
        backtest_config=config.get_backtest_config(),
        training_config=config.get_training_config(),
        model_config=config.get_model_config(),
        device=device
    )

    # Set weight policy
    simulator.set_weight_policy(
        policy_name=config.weight_policy_name,
        policy_type=config.weight_policy_type,
        window=config.weight_policy_window
    )

    # Run simulation
    logger.info("Starting simulation...")
    results = simulator.run(
        residuals=residuals,
        composition_matrix=comp_mtx,
        preprocessing=PreprocessingMethod(config.preprocessing),
        models_dir=Path(config.models_dir) if not args.test_only else None,
        test_only=args.test_only
    )

    # Save results
    results_path = Path(config.results_dir) / f"results_{config.experiment_name}"
    results.save(results_path)
    logger.info(f"Results saved to {results_path}")

    # Print summary
    stats = results.compute_aggregate_stats()
    print("\n" + "=" * 50)
    print("SIMULATION RESULTS")
    print("=" * 50)
    print(f"Sharpe Ratio:     {stats['sharpe_ratio']:.4f}")
    print(f"Annual Return:    {stats['mean_return']:.2%}")
    print(f"Annual Volatility:{stats['volatility']:.2%}")
    print(f"Max Drawdown:     {stats['max_drawdown']:.2%}")
    print(f"Total Return:     {stats['total_return']:.2%}")
    print(f"Avg Turnover:     {stats['avg_turnover']:.4f}")
    print(f"Num Periods:      {stats['num_periods']}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
