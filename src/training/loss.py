"""
Loss functions for portfolio optimization.

Implements Sharpe ratio and mean-variance objectives.
"""

from typing import Literal
import torch

from dtypes import Objective


def compute_loss(
    returns: torch.Tensor,
    objective: Objective,
    holding_days: int = 1
) -> torch.Tensor:
    """
    Compute loss for optimization objective.

    Args:
        returns: Portfolio returns (T,)
        objective: Optimization objective
        holding_days: Holding period for annualization

    Returns:
        Scalar loss value (negative for maximization)
    """
    if objective == Objective.SHARPE:
        return sharpe_loss(returns, holding_days)
    elif objective == Objective.MEAN_VARIANCE:
        return mean_variance_loss(returns, holding_days)
    else:
        raise ValueError(f"Unknown objective: {objective}")


def sharpe_loss(
    returns: torch.Tensor,
    holding_days: int = 1,
    annualization_factor: float = 252.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute negative Sharpe ratio as loss.

    Sharpe = mean(returns) / std(returns) * sqrt(annualization)

    Args:
        returns: Portfolio returns (T,)
        holding_days: Holding period
        annualization_factor: Days per year
        eps: Small value for numerical stability

    Returns:
        Negative Sharpe ratio (for minimization)
    """
    # Handle holding days
    effective_periods = annualization_factor / holding_days

    mean_return = torch.mean(returns)
    std_return = torch.std(returns, unbiased=False)

    # Annualized Sharpe ratio
    sharpe = mean_return / (std_return + eps) * torch.sqrt(torch.tensor(effective_periods))

    # Return negative for minimization
    return -sharpe


def mean_variance_loss(
    returns: torch.Tensor,
    holding_days: int = 1,
    annualization_factor: float = 252.0,
    risk_aversion: float = 1.0
) -> torch.Tensor:
    """
    Compute mean-variance loss.

    Loss = -(mean - risk_aversion * variance)

    Args:
        returns: Portfolio returns (T,)
        holding_days: Holding period
        annualization_factor: Days per year
        risk_aversion: Risk aversion coefficient

    Returns:
        Negative mean-variance utility
    """
    effective_periods = annualization_factor / holding_days

    mean_return = torch.mean(returns) * effective_periods
    variance = torch.var(returns, unbiased=False) * effective_periods

    utility = mean_return - risk_aversion * variance

    return -utility


def compute_sharpe_ratio(
    returns: torch.Tensor,
    holding_days: int = 1,
    annualization_factor: float = 252.0
) -> float:
    """
    Compute annualized Sharpe ratio.

    Args:
        returns: Portfolio returns
        holding_days: Holding period
        annualization_factor: Days per year

    Returns:
        Annualized Sharpe ratio
    """
    with torch.no_grad():
        loss = sharpe_loss(returns, holding_days, annualization_factor)
        return -loss.item()


def compute_stats(
    returns: torch.Tensor,
    holding_days: int = 1,
    annualization_factor: float = 252.0
) -> dict:
    """
    Compute comprehensive return statistics.

    Args:
        returns: Portfolio returns
        holding_days: Holding period
        annualization_factor: Days per year

    Returns:
        Dictionary of statistics
    """
    effective_periods = annualization_factor / holding_days

    with torch.no_grad():
        mean_return = torch.mean(returns).item() * effective_periods
        std_return = torch.std(returns, unbiased=False).item() * (effective_periods ** 0.5)
        sharpe = mean_return / (std_return + 1e-8)

        # Compute max drawdown
        cum_returns = torch.cumprod(1 + returns, dim=0)
        running_max = torch.cummax(cum_returns, dim=0)[0]
        drawdown = (running_max - cum_returns) / running_max
        max_drawdown = torch.max(drawdown).item()

        return {
            "mean_return": mean_return,
            "volatility": std_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "num_periods": len(returns),
        }
