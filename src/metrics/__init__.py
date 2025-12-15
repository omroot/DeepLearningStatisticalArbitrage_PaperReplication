"""
Metrics and evaluation module.
"""

from metrics.performance import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    annualized_return,
    annualized_volatility,
    compute_all_metrics,
)
from metrics.statistics import (
    t_statistic,
    information_ratio,
    hit_rate,
)

__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "annualized_return",
    "annualized_volatility",
    "compute_all_metrics",
    "t_statistic",
    "information_ratio",
    "hit_rate",
]
