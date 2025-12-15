"""
Portfolio weight computation and transformation.

This module handles:
- Weight normalization (L1 for leverage constraint)
- Transformation from residual to asset weights via composition matrix
- Portfolio return computation with transaction costs
"""

from typing import Optional, Tuple
import torch

from utils.tensor_ops import safe_divide


def normalize_weights(
    weights: torch.Tensor,
    composition_matrix: Optional[torch.Tensor] = None,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Normalize weights to enforce leverage constraint (L1 = 1).

    When composition_matrix is provided, normalization is done in
    asset space but applied to residual weights.

    Args:
        weights: Residual weights of shape (T, N_res) or (N_res,)
        composition_matrix: Optional composition matrix (T, N_res, N_assets)
        eps: Small value to avoid division by zero

    Returns:
        Tuple of (normalized_weights, asset_weights)
        - normalized_weights: (T, N_res) or (N_res,)
        - asset_weights: (T, N_assets) or (N_assets,) if comp_mtx provided
    """
    is_1d = weights.dim() == 1
    if is_1d:
        weights = weights.unsqueeze(0)

    if composition_matrix is not None:
        if composition_matrix.dim() == 2:
            composition_matrix = composition_matrix.unsqueeze(0)

        # Compute asset weights: (T, N_res) @ (T, N_res, N_assets) -> (T, N_assets)
        # Using batch matrix multiplication
        asset_weights = torch.bmm(
            weights.unsqueeze(1),  # (T, 1, N_res)
            composition_matrix     # (T, N_res, N_assets)
        ).squeeze(1)               # (T, N_assets)

        # Compute L1 norm in asset space
        l1_norm = torch.sum(torch.abs(asset_weights), dim=1, keepdim=True)  # (T, 1)

        # Normalize both residual and asset weights
        normalized_weights = weights / (l1_norm + eps)
        normalized_asset_weights = asset_weights / (l1_norm + eps)

    else:
        # Normalize in residual space
        l1_norm = torch.sum(torch.abs(weights), dim=1, keepdim=True)
        normalized_weights = weights / (l1_norm + eps)
        normalized_asset_weights = None

    if is_1d:
        normalized_weights = normalized_weights.squeeze(0)
        if normalized_asset_weights is not None:
            normalized_asset_weights = normalized_asset_weights.squeeze(0)

    return normalized_weights, normalized_asset_weights


def transform_to_asset_weights(
    residual_weights: torch.Tensor,
    composition_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Transform residual weights to asset weights using composition matrix.

    Args:
        residual_weights: Weights of shape (T, N_res) or (N_res,)
        composition_matrix: Matrix of shape (T, N_res, N_assets) or (N_res, N_assets)

    Returns:
        Asset weights of shape (T, N_assets) or (N_assets,)
    """
    is_1d = residual_weights.dim() == 1
    if is_1d:
        residual_weights = residual_weights.unsqueeze(0)
        composition_matrix = composition_matrix.unsqueeze(0)

    # Batch matrix multiply: (T, 1, N_res) @ (T, N_res, N_assets) -> (T, 1, N_assets)
    asset_weights = torch.bmm(
        residual_weights.unsqueeze(1),
        composition_matrix
    ).squeeze(1)

    if is_1d:
        asset_weights = asset_weights.squeeze(0)

    return asset_weights


def apply_weight_policy(
    weights: torch.Tensor,
    policy_name: str,
    policy_type: str = "residuals",
    composition_matrix: Optional[torch.Tensor] = None,
    **policy_kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Apply a weight policy to transform portfolio weights.

    Args:
        weights: Input weights (T, N)
        policy_name: Name of policy ('normal', 'moving_average', 'sparse_percent')
        policy_type: Apply in 'residuals' or 'assets' space
        composition_matrix: Required if policy_type='assets'
        **policy_kwargs: Policy-specific parameters

    Returns:
        Tuple of (transformed_weights, asset_weights)
    """
    from portfolio.policies import apply_moving_average, apply_sparse_percent

    if policy_name == "normal":
        # No transformation, just normalize
        return normalize_weights(weights, composition_matrix)

    elif policy_name == "moving_average":
        window = policy_kwargs.get("window", 6)

        if policy_type == "residuals":
            # Apply MA in residual space
            weights = apply_moving_average(weights, window)
            return normalize_weights(weights, composition_matrix)

        else:  # assets
            if composition_matrix is None:
                raise ValueError("composition_matrix required for assets policy")

            # First normalize in residual space
            norm_weights, _ = normalize_weights(weights)

            # Transform to assets
            asset_weights = transform_to_asset_weights(norm_weights, composition_matrix)

            # Apply MA in asset space
            asset_weights = apply_moving_average(asset_weights, window)

            # Re-normalize in asset space
            l1_norm = torch.sum(torch.abs(asset_weights), dim=1, keepdim=True)
            asset_weights = asset_weights / (l1_norm + 1e-8)

            return norm_weights, asset_weights

    elif policy_name == "sparse_percent":
        percent = policy_kwargs.get("percent", 0.1)
        weights = apply_sparse_percent(weights, percent)
        return normalize_weights(weights, composition_matrix)

    else:
        raise ValueError(f"Unknown policy: {policy_name}")


def compute_portfolio_returns(
    weights: torch.Tensor,
    returns: torch.Tensor,
    transaction_cost: float = 0.0,
    holding_cost: float = 0.0,
    holding_days: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute portfolio returns with transaction costs.

    Args:
        weights: Portfolio weights (T, N)
        returns: Asset returns (T, N)
        transaction_cost: Transaction cost per unit turnover
        holding_cost: Cost for short positions per period
        holding_days: Number of days between rebalances

    Returns:
        Tuple of (portfolio_returns, turnovers, short_proportions)
    """
    T, N = weights.shape

    if holding_days == 1:
        # Simple case: daily returns
        gross_returns = torch.sum(weights * returns, dim=1)

        # Turnover: sum of absolute weight changes
        prev_weights = torch.cat([
            torch.zeros(1, N, device=weights.device, dtype=weights.dtype),
            weights[:-1]
        ], dim=0)
        turnovers = torch.sum(torch.abs(weights - prev_weights), dim=1)

        # Short positions
        short_proportions = torch.sum(torch.clamp(-weights, min=0), dim=1)

        # Net returns after costs
        net_returns = (
            gross_returns
            - transaction_cost * turnovers
            - holding_cost * short_proportions
        )

        return net_returns, turnovers, short_proportions

    else:
        # Multi-day holding
        num_periods = T // holding_days
        net_returns = torch.zeros(T, device=weights.device, dtype=weights.dtype)
        turnovers = torch.zeros(T, device=weights.device, dtype=weights.dtype)
        short_proportions = torch.zeros(T, device=weights.device, dtype=weights.dtype)

        for t in range(holding_days, T):
            # Use weights from B days ago
            w_idx = t - holding_days + 1

            # Compute cumulative return over holding period
            period_returns = returns[w_idx:t+1]
            cum_return = torch.prod(1 + torch.sum(weights[w_idx] * period_returns, dim=1)) - 1

            # Average daily return
            net_returns[t] = cum_return / holding_days

            # Turnover only at rebalance points
            if (t - holding_days) % holding_days == 0:
                if w_idx > 0:
                    turnovers[t] = torch.sum(torch.abs(weights[w_idx] - weights[w_idx - holding_days]))

            short_proportions[t] = torch.sum(torch.clamp(-weights[w_idx], min=0))

        # Apply costs
        net_returns = (
            net_returns
            - transaction_cost * turnovers / holding_days
            - holding_cost * short_proportions
        )

        return net_returns, turnovers, short_proportions
