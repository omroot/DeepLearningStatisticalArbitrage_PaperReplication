"""
Preprocessing methods for time series data.

This module implements three preprocessing approaches:
1. Cumulative sum windows (cumsum)
2. Fourier coefficients (fourier)
3. Ornstein-Uhlenbeck parameters (ou)
"""

from typing import Union
import torch
import numpy as np

from dtypes import PreprocessingMethod


def preprocess_data(
    data: torch.Tensor,
    method: Union[PreprocessingMethod, str],
    lookback: int
) -> torch.Tensor:
    """
    Preprocess time series data using specified method.

    Args:
        data: Input data tensor of shape (T, N)
        method: Preprocessing method to apply
        lookback: Lookback window size

    Returns:
        Preprocessed tensor of shape:
            - cumsum/fourier: (T-lookback, N, lookback)
            - ou: (T-lookback, N, 4)

    Raises:
        ValueError: If method is not recognized
    """
    if isinstance(method, str):
        method = PreprocessingMethod(method)

    if method == PreprocessingMethod.CUMSUM:
        return preprocess_cumsum(data, lookback)
    elif method == PreprocessingMethod.FOURIER:
        return preprocess_fourier(data, lookback)
    elif method == PreprocessingMethod.OU:
        return preprocess_ou(data, lookback)
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")


def preprocess_cumsum(
    data: torch.Tensor,
    lookback: int
) -> torch.Tensor:
    """
    Compute cumulative sum windows.

    For each asset and time t, computes the cumulative returns
    over the lookback window [t-lookback+1, t].

    Args:
        data: Returns tensor of shape (T, N)
        lookback: Window size

    Returns:
        Windows tensor of shape (T-lookback, N, lookback)
    """
    T, N = data.shape
    num_windows = T - lookback

    # Compute cumulative sum along time dimension
    cumsum = torch.cumsum(data, dim=0)

    # Pre-allocate output tensor
    windows = torch.zeros(num_windows, N, lookback, dtype=data.dtype, device=data.device)

    # For each window position
    for t in range(num_windows):
        # Window covers [t, t+lookback)
        # Cumulative return in window = cumsum[t+lookback-1] - cumsum[t-1]
        if t == 0:
            # First window: just use cumsum directly
            windows[t] = cumsum[t:t + lookback].T
        else:
            # Subtract baseline
            baseline = cumsum[t - 1].unsqueeze(1)  # (N, 1)
            windows[t] = (cumsum[t:t + lookback].T - baseline)

    return windows


def preprocess_cumsum_vectorized(
    data: torch.Tensor,
    lookback: int
) -> torch.Tensor:
    """
    Vectorized version of cumsum preprocessing.

    More memory efficient for large datasets.

    Args:
        data: Returns tensor of shape (T, N)
        lookback: Window size

    Returns:
        Windows tensor of shape (T-lookback, N, lookback)
    """
    T, N = data.shape
    num_windows = T - lookback

    # Create indices for extracting windows
    # Shape: (num_windows, lookback)
    indices = torch.arange(lookback, device=data.device).unsqueeze(0) + \
              torch.arange(num_windows, device=data.device).unsqueeze(1)

    # Extract windows: (num_windows, lookback, N) -> (num_windows, N, lookback)
    windows = data[indices].permute(0, 2, 1)

    # Compute cumulative sum within each window
    windows = torch.cumsum(windows, dim=2)

    return windows


def preprocess_fourier(
    data: torch.Tensor,
    lookback: int
) -> torch.Tensor:
    """
    Compute Fourier coefficients of cumulative return windows.

    Applies FFT to cumsum windows and returns real/imaginary parts.

    Args:
        data: Returns tensor of shape (T, N)
        lookback: Window size

    Returns:
        Fourier features tensor of shape (T-lookback, N, lookback)
    """
    # First get cumsum windows
    windows = preprocess_cumsum(data, lookback)

    # Apply real FFT along last dimension
    # Output shape: (T-lookback, N, lookback//2 + 1) complex
    fft_result = torch.fft.rfft(windows, dim=-1)

    # Extract real and imaginary parts
    real_part = fft_result.real
    imag_part = fft_result.imag

    # Concatenate real and imaginary (excluding DC component's imaginary which is 0)
    # and Nyquist's imaginary (also 0 for real input)
    num_freqs = fft_result.shape[-1]

    # Output structure: [real_0, ..., real_n, imag_1, ..., imag_{n-1}]
    features = torch.zeros_like(windows)
    features[..., :num_freqs] = real_part
    features[..., num_freqs:num_freqs + num_freqs - 2] = imag_part[..., 1:-1]

    return features


def preprocess_ou(
    data: torch.Tensor,
    lookback: int
) -> torch.Tensor:
    """
    Compute Ornstein-Uhlenbeck parameters for each window.

    For each asset and time window, estimates OU process parameters:
    - Y_t (current cumulative return)
    - μ (long-term mean)
    - σ (volatility)
    - R² (regression fit quality)

    Args:
        data: Returns tensor of shape (T, N)
        lookback: Window size

    Returns:
        OU parameters tensor of shape (T-lookback, N, 4)
        Features: [Y_t, μ, σ, R²]
    """
    T, N = data.shape
    num_windows = T - lookback

    # Compute cumulative sum
    cumsum = torch.cumsum(data, dim=0)

    # Pre-allocate output
    ou_params = torch.zeros(num_windows, N, 4, dtype=data.dtype, device=data.device)

    for t in range(num_windows):
        # Get window of cumulative returns
        if t == 0:
            window = cumsum[t:t + lookback]
        else:
            window = cumsum[t:t + lookback] - cumsum[t - 1]

        # window shape: (lookback, N)
        # OU regression: Y_t = β * X_{t-1} + α + ε
        X = window[:-1]  # (lookback-1, N)
        Y = window[1:]   # (lookback-1, N)

        # Compute statistics for OLS regression
        X_mean = X.mean(dim=0, keepdim=True)
        Y_mean = Y.mean(dim=0, keepdim=True)

        X_centered = X - X_mean
        Y_centered = Y - Y_mean

        # β = Cov(X, Y) / Var(X)
        var_X = (X_centered ** 2).sum(dim=0)
        cov_XY = (X_centered * Y_centered).sum(dim=0)

        # Avoid division by zero
        eps = 1e-8
        beta = cov_XY / (var_X + eps)

        # α = E[Y] - β * E[X]
        alpha = Y_mean.squeeze(0) - beta * X_mean.squeeze(0)

        # OU parameters
        # μ = α / (1 - β) - long-term mean
        # Only valid when 0 < β < 1 (mean-reverting)
        valid_beta = (beta > 0) & (beta < 1)
        mu = torch.where(valid_beta, alpha / (1 - beta + eps), torch.zeros_like(alpha))

        # σ² = Var(ε) / |1 - β²|
        residuals = Y - beta * X - alpha
        var_residuals = (residuals ** 2).mean(dim=0)
        sigma_sq = var_residuals / (torch.abs(1 - beta ** 2) + eps)
        sigma = torch.sqrt(sigma_sq + eps)

        # R² = Cov(X,Y)² / (Var(X) * Var(Y))
        var_Y = (Y_centered ** 2).sum(dim=0)
        r_squared = (cov_XY ** 2) / ((var_X * var_Y) + eps)

        # Current cumulative return (last value in window)
        Y_current = window[-1]

        # Store parameters: [Y_t, μ, σ, R²]
        ou_params[t, :, 0] = Y_current
        ou_params[t, :, 1] = mu
        ou_params[t, :, 2] = sigma
        ou_params[t, :, 3] = r_squared

        # Set invalid (non-mean-reverting) assets to zero
        ou_params[t, ~valid_beta, :] = 0

    return ou_params
