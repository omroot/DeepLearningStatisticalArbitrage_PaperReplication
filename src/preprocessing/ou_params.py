"""
Ornstein-Uhlenbeck process parameter estimation.

This module provides utilities for estimating OU process parameters
from time series data.
"""

from typing import Tuple, Optional
import torch
import numpy as np


def compute_ou_parameters(
    cumulative_returns: torch.Tensor,
    use_robust_estimators: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Ornstein-Uhlenbeck process parameters from cumulative returns.

    The OU process is modeled as:
        dX_t = θ(μ - X_t)dt + σdW_t

    In discrete form:
        Y_t = β * X_{t-1} + α + ε_t

    Where:
        β = exp(-θΔt) ≈ 1 - θΔt for small Δt
        α = μ(1 - β)
        σ_ε = σ√((1 - β²)/(2θ))

    Args:
        cumulative_returns: Tensor of shape (T, N) with cumulative returns
        use_robust_estimators: If True, use median/MAD instead of mean/std

    Returns:
        Tuple of (beta, alpha, mu, sigma, r_squared) each of shape (N,)
    """
    T, N = cumulative_returns.shape
    eps = 1e-8

    # Lag-1 regression: Y_t on X_{t-1}
    X = cumulative_returns[:-1]  # (T-1, N)
    Y = cumulative_returns[1:]   # (T-1, N)

    if use_robust_estimators:
        # Use median and MAD (median absolute deviation)
        X_center = torch.median(X, dim=0).values
        Y_center = torch.median(Y, dim=0).values

        X_centered = X - X_center
        Y_centered = Y - Y_center

        # MAD for variance estimation
        mad_X = torch.median(torch.abs(X_centered), dim=0).values
        mad_Y = torch.median(torch.abs(Y_centered), dim=0).values

        # Robust correlation via median of products
        # This is a simplified approach
        var_X = (1.4826 * mad_X) ** 2  # Scale MAD to std
        var_Y = (1.4826 * mad_Y) ** 2
        cov_XY = torch.median(X_centered * Y_centered, dim=0).values

    else:
        # Standard OLS estimators
        X_mean = X.mean(dim=0)
        Y_mean = Y.mean(dim=0)

        X_centered = X - X_mean
        Y_centered = Y - Y_mean

        var_X = (X_centered ** 2).mean(dim=0)
        var_Y = (Y_centered ** 2).mean(dim=0)
        cov_XY = (X_centered * Y_centered).mean(dim=0)

        X_center = X_mean
        Y_center = Y_mean

    # OLS estimates
    beta = cov_XY / (var_X + eps)
    alpha = Y_center - beta * X_center

    # OU parameters
    # μ = α / (1 - β)
    valid_beta = (beta > eps) & (beta < 1 - eps)
    mu = torch.where(valid_beta, alpha / (1 - beta + eps), torch.zeros_like(alpha))

    # σ = sqrt(Var(residuals) / |1 - β²|)
    residuals = Y - beta * X - alpha
    var_residuals = (residuals ** 2).mean(dim=0)
    sigma = torch.sqrt(var_residuals / (torch.abs(1 - beta ** 2) + eps) + eps)

    # R² = Cov(X,Y)² / (Var(X) * Var(Y))
    r_squared = (cov_XY ** 2) / ((var_X * var_Y) + eps)
    r_squared = torch.clamp(r_squared, 0, 1)

    # Set invalid beta to indicate non-mean-reverting
    beta = torch.where(valid_beta, beta, torch.zeros_like(beta))

    return beta, alpha, mu, sigma, r_squared


def compute_ou_signal(
    cumulative_returns: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    r_squared: torch.Tensor,
    signal_threshold: float = 1.25,
    r2_threshold: float = 0.25
) -> torch.Tensor:
    """
    Compute trading signal based on OU process.

    Signal = (μ - Y_t) / σ

    The signal indicates how many standard deviations the current
    value is from the long-term mean.

    Args:
        cumulative_returns: Current cumulative returns (N,)
        mu: Long-term mean estimates (N,)
        sigma: Volatility estimates (N,)
        r_squared: R² values from regression (N,)
        signal_threshold: Threshold for trading (in std devs)
        r2_threshold: Minimum R² for valid signal

    Returns:
        Trading signal tensor (N,) with values in {-1, 0, 1}
    """
    eps = 1e-8

    # Compute raw signal
    signal = (mu - cumulative_returns) / (sigma + eps)

    # Apply R² filter
    valid_r2 = r_squared >= r2_threshold

    # Generate trading positions
    positions = torch.zeros_like(signal)
    positions = torch.where(
        valid_r2 & (signal > signal_threshold),
        torch.ones_like(signal),
        positions
    )
    positions = torch.where(
        valid_r2 & (signal < -signal_threshold),
        -torch.ones_like(signal),
        positions
    )

    return positions


def compute_ou_half_life(beta: torch.Tensor) -> torch.Tensor:
    """
    Compute half-life of mean reversion from beta.

    Half-life = -ln(2) / ln(β)

    Args:
        beta: AR(1) coefficient from OU regression

    Returns:
        Half-life in time units (same as data frequency)
    """
    eps = 1e-8
    # Ensure beta is in valid range
    valid_beta = (beta > eps) & (beta < 1 - eps)

    half_life = -np.log(2) / torch.log(beta + eps)

    # Set invalid to inf
    half_life = torch.where(valid_beta, half_life, torch.full_like(half_life, float('inf')))

    return half_life


def estimate_ou_from_prices(
    prices: torch.Tensor,
    lookback: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Estimate OU parameters from price series.

    First converts to log prices, then estimates parameters.

    Args:
        prices: Price tensor of shape (T, N)
        lookback: Window for estimation

    Returns:
        Tuple of (mu, sigma, half_life, r_squared)
    """
    # Convert to log prices
    log_prices = torch.log(prices + 1e-8)

    # Use the last lookback observations
    window = log_prices[-lookback:]

    # Estimate parameters
    beta, alpha, mu, sigma, r_squared = compute_ou_parameters(window)

    # Compute half-life
    half_life = compute_ou_half_life(beta)

    return mu, sigma, half_life, r_squared
