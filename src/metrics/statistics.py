"""
Statistical tests and measures for portfolio evaluation.
"""

from typing import Union, Tuple
import numpy as np
import torch
from scipy import stats


def t_statistic(
    returns: Union[np.ndarray, torch.Tensor],
    null_mean: float = 0.0
) -> Tuple[float, float]:
    """
    Compute t-statistic for testing if mean return differs from null.

    Args:
        returns: Array of returns
        null_mean: Null hypothesis mean

    Returns:
        Tuple of (t-statistic, p-value)
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()

    returns = np.asarray(returns)

    if len(returns) < 2:
        return 0.0, 1.0

    t_stat, p_value = stats.ttest_1samp(returns, null_mean)
    return float(t_stat), float(p_value)


def information_ratio(
    returns: Union[np.ndarray, torch.Tensor],
    benchmark_returns: Union[np.ndarray, torch.Tensor],
    annualization_factor: float = 252.0
) -> float:
    """
    Compute information ratio (excess return / tracking error).

    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        annualization_factor: Trading days per year

    Returns:
        Annualized information ratio
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()
    if isinstance(benchmark_returns, torch.Tensor):
        benchmark_returns = benchmark_returns.detach().cpu().numpy()

    returns = np.asarray(returns)
    benchmark_returns = np.asarray(benchmark_returns)

    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark must have same length")

    if len(returns) == 0:
        return 0.0

    excess_returns = returns - benchmark_returns
    tracking_error = np.std(excess_returns, ddof=1)

    if tracking_error == 0:
        return 0.0

    return np.mean(excess_returns) / tracking_error * np.sqrt(annualization_factor)


def hit_rate(
    returns: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute hit rate (fraction of positive returns).

    Args:
        returns: Array of returns

    Returns:
        Hit rate (0 to 1)
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()

    returns = np.asarray(returns)

    if len(returns) == 0:
        return 0.0

    return np.mean(returns > 0)


def profit_factor(
    returns: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute profit factor (gross profit / gross loss).

    Args:
        returns: Array of returns

    Returns:
        Profit factor (>1 is profitable)
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()

    returns = np.asarray(returns)

    gains = returns[returns > 0]
    losses = returns[returns < 0]

    gross_profit = np.sum(gains) if len(gains) > 0 else 0.0
    gross_loss = -np.sum(losses) if len(losses) > 0 else 0.0

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def avg_win_loss_ratio(
    returns: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute average win / average loss ratio.

    Args:
        returns: Array of returns

    Returns:
        Win/loss ratio
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()

    returns = np.asarray(returns)

    gains = returns[returns > 0]
    losses = returns[returns < 0]

    avg_win = np.mean(gains) if len(gains) > 0 else 0.0
    avg_loss = -np.mean(losses) if len(losses) > 0 else 0.0

    if avg_loss == 0:
        return float('inf') if avg_win > 0 else 0.0

    return avg_win / avg_loss


def newey_west_t_stat(
    returns: Union[np.ndarray, torch.Tensor],
    lags: int = None
) -> Tuple[float, float]:
    """
    Compute Newey-West adjusted t-statistic for autocorrelated returns.

    Args:
        returns: Array of returns
        lags: Number of lags (default: floor(4 * (T/100)^(2/9)))

    Returns:
        Tuple of (t-statistic, p-value)
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()

    returns = np.asarray(returns)
    T = len(returns)

    if T < 2:
        return 0.0, 1.0

    if lags is None:
        lags = int(np.floor(4 * (T / 100) ** (2 / 9)))

    mean_return = np.mean(returns)
    demeaned = returns - mean_return

    # Variance with Newey-West correction
    gamma_0 = np.sum(demeaned ** 2) / T

    nw_var = gamma_0
    for j in range(1, lags + 1):
        gamma_j = np.sum(demeaned[j:] * demeaned[:-j]) / T
        weight = 1 - j / (lags + 1)  # Bartlett kernel
        nw_var += 2 * weight * gamma_j

    se = np.sqrt(nw_var / T)

    if se == 0:
        return 0.0, 1.0

    t_stat = mean_return / se

    # Two-tailed p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=T - 1))

    return float(t_stat), float(p_value)


def correlation_with_benchmark(
    returns: Union[np.ndarray, torch.Tensor],
    benchmark_returns: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute correlation with benchmark.

    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Correlation coefficient
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()
    if isinstance(benchmark_returns, torch.Tensor):
        benchmark_returns = benchmark_returns.detach().cpu().numpy()

    returns = np.asarray(returns)
    benchmark_returns = np.asarray(benchmark_returns)

    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        return 0.0

    corr = np.corrcoef(returns, benchmark_returns)[0, 1]
    return float(np.nan_to_num(corr))
