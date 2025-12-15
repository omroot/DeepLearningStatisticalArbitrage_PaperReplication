#!/usr/bin/env python3
"""
Compute residuals using factor models.

Usage:
    python run_factor_model.py --data-dir ./data --output-dir ./residuals --model IPCA --factors 5
"""

import argparse
from pathlib import Path
import sys

import numpy as np

from factor_models import create_factor_model
from utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute residuals using factor models"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to data directory with returns.npy"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for residuals"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["IPCA", "PCA", "FamaFrench"],
        default="IPCA",
        help="Factor model type"
    )
    parser.add_argument(
        "--factors",
        type=int,
        default=5,
        help="Number of factors"
    )
    parser.add_argument(
        "--characteristics",
        type=str,
        help="Path to characteristics file (required for IPCA)"
    )
    parser.add_argument(
        "--ff-factors",
        type=str,
        help="Path to Fama-French factors file (required for FamaFrench)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    logger = setup_logger("dlsa.factor_model")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load returns
    returns_path = data_dir / "returns.npy"
    if not returns_path.exists():
        logger.error(f"Returns file not found: {returns_path}")
        return 1

    logger.info(f"Loading returns from {returns_path}")
    returns = np.load(returns_path)
    logger.info(f"Returns shape: {returns.shape}")

    # Load characteristics if needed
    characteristics = None
    if args.model == "IPCA":
        if args.characteristics:
            char_path = Path(args.characteristics)
        else:
            char_path = data_dir / "characteristics.npy"

        if not char_path.exists():
            logger.error(f"Characteristics file required for IPCA: {char_path}")
            return 1

        logger.info(f"Loading characteristics from {char_path}")
        characteristics = np.load(char_path)
        logger.info(f"Characteristics shape: {characteristics.shape}")

    # Load FF factors if needed
    ff_factors = None
    if args.model == "FamaFrench":
        if args.ff_factors:
            ff_path = Path(args.ff_factors)
        else:
            ff_path = data_dir / "ff_factors.npy"

        if not ff_path.exists():
            logger.error(f"Fama-French factors file required: {ff_path}")
            return 1

        logger.info(f"Loading FF factors from {ff_path}")
        ff_factors = np.load(ff_path)

    # Create factor model
    logger.info(f"Creating {args.model} factor model with {args.factors} factors")
    model = create_factor_model(args.model, args.factors)

    # Fit model
    logger.info("Fitting factor model...")
    if args.model == "IPCA":
        model.fit(returns, characteristics)
    elif args.model == "FamaFrench":
        model.fit(returns, factor_returns=ff_factors)
    else:
        model.fit(returns)

    # Compute residuals
    logger.info("Computing residuals...")
    if args.model == "IPCA":
        residuals = model.compute_residuals(returns, characteristics)
    elif args.model == "FamaFrench":
        residuals = model.compute_residuals(returns, factor_returns=ff_factors)
    else:
        residuals = model.compute_residuals(returns)

    # Save residuals
    residuals_path = output_dir / f"residuals-{args.model}-{args.factors}.npy"
    logger.info(f"Saving residuals to {residuals_path}")
    np.save(residuals_path, residuals)

    # Compute and save composition matrix
    logger.info("Computing composition matrix...")
    comp_mtx = model.compute_composition_matrix(returns, characteristics)
    comp_mtx_path = output_dir / f"comp-mtx-{args.model}-{args.factors}.npy"
    np.save(comp_mtx_path, comp_mtx)
    logger.info(f"Saved composition matrix to {comp_mtx_path}")

    # Save model
    model_path = output_dir / f"factor-model-{args.model}-{args.factors}.npz"
    model.save(model_path)
    logger.info(f"Saved model to {model_path}")

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
