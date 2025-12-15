"""
Performance metrics for portfolio evaluation.

Implements standard risk-adjusted return metrics.
"""

from typing import Union, Dict, Any
import numpy as np
import torch


def sharpe_ratio(
    returns: Union[np.ndarray, torch.Tensor],
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252.0
) -> float:
    """
    Compute annualized Sharpe ratio.

    Sharpe = (mean(returns) - rf) / std(returns) * sqrt(annualization)

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate (default 0)
        annualization_factor: Trading days per year

    Returns:
        Annualized Sharpe ratio
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()

    returns = np.asarray(returns)

    if len(returns) == 0:
        return 0.0

    daily_rf = risk_free_rate / annualization_factor
    excess_returns = returns - daily_rf

    mean_excess = np.mean(excess_returns)
    std_returns = np.std(returns, ddof=1)

    if std_returns == 0:
        return 0.0

    return mean_excess / std_returns * np.sqrt(annualization_factor)


def sortino_ratio(
    returns: Union[np.ndarray, torch.Tensor],
    target_return: float = 0.0,
    annualization_factor: float = 252.0
) -> float:
    """
    Compute annualized Sortino ratio.

    Similar to Sharpe but only penalizes downside volatility.

    Args:
        returns: Array of returns
        target_return: Target return (annual)
        annualization_factor: Trading days per year

    Returns:
        Annualized Sortino ratio
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()

    returns = np.asarray(returns)

    if len(returns) == 0:
        return 0.0

    daily_target = target_return / annualization_factor
    excess_returns = returns - daily_target

    # Downside returns (negative excess)
    downside_returns = np.minimum(excess_returns, 0)
    downside_std = np.std(downside_returns, ddof=1)

    if downside_std == 0:
        return 0.0

    mean_excess = np.mean(excess_returns)
    return mean_excess / downside_std * np.sqrt(annualization_factor)


def max_drawdown(
    returns: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute maximum drawdown.

    Args:
        returns: Array of returns

    Returns:
        Maximum drawdown as a positive fraction
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()

    returns = np.asarray(returns)

    if len(returns) == 0:
        return 0.0

    # Compute cumulative returns
    cum_returns = np.cumprod(1 + returns)

    # Running maximum
    running_max = np.maximum.accumulate(cum_returns)

    # Drawdown series
    drawdown = (running_max - cum_returns) / running_max

    return np.max(drawdown)


def calmar_ratio(
    returns: Union[np.ndarray, torch.Tensor],
    annualization_factor: float = 252.0
) -> float:
    """
    Compute Calmar ratio (return / max drawdown).

    Args:
        returns: Array of returns
        annualization_factor: Trading days per year

    Returns:
        Calmar ratio
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()

    returns = np.asarray(returns)

    mdd = max_drawdown(returns)
    annual_return = annualized_return(returns, annualization_factor)

    if mdd == 0:
        return 0.0

    return annual_return / mdd


def annualized_return(
    returns: Union[np.ndarray, torch.Tensor],
    annualization_factor: float = 252.0
) -> float:
    """
    Compute annualized return.

    Args:
        returns: Array of returns
        annualization_factor: Trading days per year

    Returns:
        Annualized return
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()

    returns = np.asarray(returns)

    if len(returns) == 0:
        return 0.0

    # Compound return
    total_return = np.prod(1 + returns) - 1
    num_years = len(returns) / annualization_factor

    if num_years == 0:
        return 0.0

    # Annualized (geometric mean)
    return (1 + total_return) ** (1 / num_years) - 1


def annualized_volatility(
    returns: Union[np.ndarray, torch.Tensor],
    annualization_factor: float = 252.0
) -> float:
    """
    Compute annualized volatility.

    Args:
        returns: Array of returns
        annualization_factor: Trading days per year

    Returns:
        Annualized volatility
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()

    returns = np.asarray(returns)

    if len(returns) == 0:
        return 0.0

    return np.std(returns, ddof=1) * np.sqrt(annualization_factor)


def compute_all_metrics(
    returns: Union[np.ndarray, torch.Tensor],
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252.0
) -> Dict[str, float]:
    """
    Compute all performance metrics.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        annualization_factor: Trading days per year

    Returns:
        Dictionary of all metrics
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()

    returns = np.asarray(returns)

    # Cumulative returns
    cum_returns = np.cumprod(1 + returns)
    total_return = cum_returns[-1] - 1 if len(cum_returns) > 0 else 0.0

    return {
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, annualization_factor),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, annualization_factor),
        "calmar_ratio": calmar_ratio(returns, annualization_factor),
        "max_drawdown": max_drawdown(returns),
        "annualized_return": annualized_return(returns, annualization_factor),
        "annualized_volatility": annualized_volatility(returns, annualization_factor),
        "total_return": total_return,
        "num_periods": len(returns),
        "mean_daily_return": np.mean(returns) if len(returns) > 0 else 0.0,
        "std_daily_return": np.std(returns, ddof=1) if len(returns) > 1 else 0.0,
        "skewness": float(np.nan_to_num(
            np.mean((returns - np.mean(returns)) ** 3) / (np.std(returns) ** 3 + 1e-8)
        )) if len(returns) > 0 else 0.0,
        "kurtosis": float(np.nan_to_num(
            np.mean((returns - np.mean(returns)) ** 4) / (np.std(returns) ** 4 + 1e-8) - 3
        )) if len(returns) > 0 else 0.0,
    }
