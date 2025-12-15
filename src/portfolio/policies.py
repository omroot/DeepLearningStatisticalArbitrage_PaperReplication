"""
Weight policies for portfolio construction.

Implements various weight transformation policies:
- Moving average smoothing
- Sparse portfolio selection
"""

from typing import Optional
from dataclasses import dataclass
import torch


@dataclass
class WeightPolicy:
    """
    Configuration for weight transformation policy.

    Attributes:
        name: Policy name ('normal', 'moving_average', 'sparse_percent')
        policy_type: Apply in 'residuals' or 'assets' space
        window: Window size for moving average
        percent: Percentage for sparse selection
    """
    name: str = "normal"
    policy_type: str = "residuals"
    window: int = 6
    percent: float = 0.1

    @classmethod
    def from_dict(cls, d: dict) -> "WeightPolicy":
        """Create from dictionary."""
        return cls(
            name=d.get("name", "normal"),
            policy_type=d.get("type", "residuals"),
            window=d.get("window", 6),
            percent=d.get("percent", 0.1)
        )


def apply_moving_average(
    weights: torch.Tensor,
    window: int
) -> torch.Tensor:
    """
    Apply moving average smoothing to weights.

    Uses a simple trailing moving average to smooth
    weight transitions over time.

    Args:
        weights: Portfolio weights (T, N)
        window: Moving average window size

    Returns:
        Smoothed weights (T, N)
    """
    if window <= 1:
        return weights

    T, N = weights.shape
    smoothed = torch.zeros_like(weights)

    for t in range(T):
        start_idx = max(0, t - window + 1)
        smoothed[t] = weights[start_idx:t+1].mean(dim=0)

    return smoothed


def apply_moving_average_efficient(
    weights: torch.Tensor,
    window: int
) -> torch.Tensor:
    """
    Efficient moving average using cumulative sum.

    More efficient for large tensors.

    Args:
        weights: Portfolio weights (T, N)
        window: Moving average window size

    Returns:
        Smoothed weights (T, N)
    """
    if window <= 1:
        return weights

    T, N = weights.shape

    # Pad with zeros
    padded = torch.cat([
        torch.zeros(window - 1, N, device=weights.device, dtype=weights.dtype),
        weights
    ], dim=0)

    # Cumulative sum
    cumsum = torch.cumsum(padded, dim=0)

    # Moving average via difference of cumsums
    smoothed = (cumsum[window:] - cumsum[:-window]) / window

    # Handle edge cases for first few values
    for t in range(min(window - 1, T)):
        smoothed[t] = weights[:t+1].mean(dim=0)

    return smoothed


def apply_sparse_percent(
    weights: torch.Tensor,
    percent: float
) -> torch.Tensor:
    """
    Keep only top percent of weights by absolute value.

    For each time step, zeros out all but the top `percent`
    fraction of weights (by absolute value).

    Args:
        weights: Portfolio weights (T, N)
        percent: Fraction of weights to keep (0-1)

    Returns:
        Sparse weights (T, N)
    """
    T, N = weights.shape
    k = max(1, int(N * percent))

    sparse_weights = torch.zeros_like(weights)

    for t in range(T):
        # Get indices of top k absolute weights
        abs_weights = torch.abs(weights[t])
        _, top_indices = torch.topk(abs_weights, k)

        # Keep only top weights
        sparse_weights[t, top_indices] = weights[t, top_indices]

    return sparse_weights


def apply_long_only(
    weights: torch.Tensor
) -> torch.Tensor:
    """
    Convert to long-only portfolio.

    Sets negative weights to zero and renormalizes.

    Args:
        weights: Portfolio weights (T, N)

    Returns:
        Long-only weights (T, N)
    """
    # Zero out negative weights
    long_weights = torch.clamp(weights, min=0)

    # Renormalize
    total = torch.sum(long_weights, dim=1, keepdim=True)
    long_weights = long_weights / (total + 1e-8)

    return long_weights


def apply_weight_bounds(
    weights: torch.Tensor,
    min_weight: float = -1.0,
    max_weight: float = 1.0
) -> torch.Tensor:
    """
    Clip weights to specified bounds.

    Args:
        weights: Portfolio weights (T, N)
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset

    Returns:
        Bounded weights (T, N)
    """
    return torch.clamp(weights, min=min_weight, max=max_weight)


def apply_turnover_constraint(
    weights: torch.Tensor,
    previous_weights: torch.Tensor,
    max_turnover: float
) -> torch.Tensor:
    """
    Apply turnover constraint by limiting weight changes.

    If proposed turnover exceeds max_turnover, scales down
    the weight changes proportionally.

    Args:
        weights: Proposed new weights (T, N) or (N,)
        previous_weights: Previous weights (T, N) or (N,)
        max_turnover: Maximum allowed turnover

    Returns:
        Constrained weights
    """
    # Compute proposed changes
    changes = weights - previous_weights

    # Compute proposed turnover
    turnover = torch.sum(torch.abs(changes), dim=-1, keepdim=True)

    # Scale down if needed
    scale = torch.where(
        turnover > max_turnover,
        max_turnover / (turnover + 1e-8),
        torch.ones_like(turnover)
    )

    constrained = previous_weights + changes * scale

    return constrained
