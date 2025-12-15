"""
Tensor and array operations utilities.

This module provides helper functions for converting between
NumPy arrays and PyTorch tensors, and common tensor operations.
"""

from typing import Union, Optional
import numpy as np
import torch


def to_tensor(
    data: Union[np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Convert data to PyTorch tensor.

    Args:
        data: Input data as numpy array or tensor
        device: Target device (None for default)
        dtype: Target data type

    Returns:
        PyTorch tensor on specified device
    """
    if isinstance(data, torch.Tensor):
        tensor = data.to(dtype=dtype)
    else:
        tensor = torch.tensor(data, dtype=dtype)

    if device is not None:
        tensor = tensor.to(device)

    return tensor


def to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert data to NumPy array.

    Args:
        data: Input data as tensor or array

    Returns:
        NumPy array
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def ensure_2d(data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Ensure data is 2-dimensional by adding dimensions if needed.

    Args:
        data: Input data

    Returns:
        2D array or tensor
    """
    if data.ndim == 1:
        if isinstance(data, torch.Tensor):
            return data.unsqueeze(0)
        return data.reshape(1, -1)
    return data


def safe_divide(
    numerator: Union[np.ndarray, torch.Tensor],
    denominator: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8
) -> Union[np.ndarray, torch.Tensor]:
    """
    Safely divide, avoiding division by zero.

    Args:
        numerator: Numerator values
        denominator: Denominator values
        eps: Small value added to denominator

    Returns:
        Result of division
    """
    if isinstance(numerator, torch.Tensor):
        return numerator / (denominator + eps)
    return numerator / (denominator + eps)


def moving_average(
    data: torch.Tensor,
    window: int,
    dim: int = 0
) -> torch.Tensor:
    """
    Compute moving average over a tensor dimension.

    Uses cumulative sum for efficient computation.

    Args:
        data: Input tensor
        window: Window size for averaging
        dim: Dimension to average over

    Returns:
        Tensor with moving average applied
    """
    if window <= 1:
        return data

    # Pad with zeros at the beginning
    pad_shape = list(data.shape)
    pad_shape[dim] = window - 1
    padded = torch.cat([torch.zeros(pad_shape, device=data.device, dtype=data.dtype), data], dim=dim)

    # Compute cumulative sum
    cumsum = torch.cumsum(padded, dim=dim)

    # Compute moving average using difference of cumsums
    if dim == 0:
        result = (cumsum[window:] - cumsum[:-window]) / window
    else:
        # Handle other dimensions
        slices_end = [slice(None)] * data.ndim
        slices_start = [slice(None)] * data.ndim
        slices_end[dim] = slice(window, None)
        slices_start[dim] = slice(None, -window)
        result = (cumsum[tuple(slices_end)] - cumsum[tuple(slices_start)]) / window

    return result


def normalize_weights(
    weights: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Normalize weights to sum to unit absolute value (L1 normalization).

    Args:
        weights: Input weights tensor
        dim: Dimension to normalize over
        eps: Small value to avoid division by zero

    Returns:
        Normalized weights
    """
    abs_sum = torch.sum(torch.abs(weights), dim=dim, keepdim=True)
    return weights / (abs_sum + eps)


def compute_turnover(
    weights: torch.Tensor,
    previous_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute portfolio turnover as sum of absolute weight changes.

    Args:
        weights: Current weights (T x N)
        previous_weights: Previous weights (T x N), or None to use shifted weights

    Returns:
        Turnover values (T,)
    """
    if previous_weights is None:
        # Shift weights by one period
        previous_weights = torch.cat([
            torch.zeros(1, weights.shape[1], device=weights.device, dtype=weights.dtype),
            weights[:-1]
        ], dim=0)

    return torch.sum(torch.abs(weights - previous_weights), dim=1)


def compute_short_proportion(weights: torch.Tensor) -> torch.Tensor:
    """
    Compute proportion of short positions.

    Args:
        weights: Portfolio weights (T x N)

    Returns:
        Short proportion (T,)
    """
    short_positions = torch.clamp(weights, max=0.0)
    return torch.sum(torch.abs(short_positions), dim=1)


def create_mask(
    data: torch.Tensor,
    missing_value: float = 0.0
) -> torch.Tensor:
    """
    Create boolean mask for non-missing values.

    Args:
        data: Input data where missing_value indicates missing
        missing_value: Value that indicates missing data

    Returns:
        Boolean mask (True where data is not missing)
    """
    return data != missing_value
