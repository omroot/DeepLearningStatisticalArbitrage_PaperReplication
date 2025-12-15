"""
Backtest results storage and analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

import numpy as np
import torch


@dataclass
class SubperiodResult:
    """
    Results for a single backtesting subperiod.

    Attributes:
        subperiod_idx: Index of the subperiod
        returns: Portfolio returns for this period
        weights: Portfolio weights
        turnovers: Turnover values
        short_proportions: Short position proportions
        sharpe_ratio: Sharpe ratio for this period
        mean_return: Mean return
        volatility: Volatility
        train_loss: Final training loss
    """
    subperiod_idx: int
    returns: np.ndarray
    weights: np.ndarray
    turnovers: np.ndarray
    short_proportions: np.ndarray
    sharpe_ratio: float
    mean_return: float
    volatility: float
    train_loss: float = 0.0


@dataclass
class BacktestResults:
    """
    Complete backtest results.

    Stores all subperiod results and computes aggregate statistics.
    """
    subperiod_results: List[SubperiodResult] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def add_subperiod(self, result: SubperiodResult) -> None:
        """
        Add a subperiod result.

        Args:
            result: SubperiodResult to add
        """
        self.subperiod_results.append(result)

    @property
    def all_returns(self) -> np.ndarray:
        """Get concatenated returns from all subperiods."""
        if not self.subperiod_results:
            return np.array([])
        return np.concatenate([r.returns for r in self.subperiod_results])

    @property
    def all_turnovers(self) -> np.ndarray:
        """Get concatenated turnovers from all subperiods."""
        if not self.subperiod_results:
            return np.array([])
        return np.concatenate([r.turnovers for r in self.subperiod_results])

    @property
    def all_short_proportions(self) -> np.ndarray:
        """Get concatenated short proportions from all subperiods."""
        if not self.subperiod_results:
            return np.array([])
        return np.concatenate([r.short_proportions for r in self.subperiod_results])

    def compute_aggregate_stats(
        self,
        annualization_factor: float = 252.0
    ) -> Dict[str, float]:
        """
        Compute aggregate statistics across all subperiods.

        Args:
            annualization_factor: Days per year for annualization

        Returns:
            Dictionary of aggregate statistics
        """
        returns = self.all_returns

        if len(returns) == 0:
            return {}

        mean_daily = np.mean(returns)
        std_daily = np.std(returns)

        mean_annual = mean_daily * annualization_factor
        std_annual = std_daily * np.sqrt(annualization_factor)
        sharpe = mean_annual / (std_annual + 1e-8)

        # Cumulative returns and drawdown
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (running_max - cum_returns) / (running_max + 1e-8)
        max_drawdown = np.max(drawdown)

        # Average turnover
        avg_turnover = np.mean(self.all_turnovers)

        # Average short proportion
        avg_short = np.mean(self.all_short_proportions)

        return {
            "mean_return": float(mean_annual),
            "volatility": float(std_annual),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "avg_turnover": float(avg_turnover),
            "avg_short_proportion": float(avg_short),
            "total_return": float(cum_returns[-1] - 1),
            "num_periods": len(returns),
            "num_subperiods": len(self.subperiod_results),
        }

    def compute_subperiod_stats(self) -> Dict[str, Any]:
        """
        Compute statistics across subperiods.

        Returns:
            Dictionary with mean/std of subperiod Sharpe ratios
        """
        if not self.subperiod_results:
            return {}

        sharpes = [r.sharpe_ratio for r in self.subperiod_results]

        return {
            "mean_sharpe": float(np.mean(sharpes)),
            "std_sharpe": float(np.std(sharpes)),
            "min_sharpe": float(np.min(sharpes)),
            "max_sharpe": float(np.max(sharpes)),
            "median_sharpe": float(np.median(sharpes)),
        }

    def get_cumulative_returns(self) -> np.ndarray:
        """
        Get cumulative return series.

        Returns:
            Cumulative returns array
        """
        returns = self.all_returns
        if len(returns) == 0:
            return np.array([1.0])
        return np.cumprod(1 + returns)

    def save(self, path: Path) -> None:
        """
        Save results to disk.

        Args:
            path: Path to save results
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        data = {
            "config": self.config,
            "aggregate_stats": self.compute_aggregate_stats(),
            "subperiod_stats": self.compute_subperiod_stats(),
            "subperiods": [
                {
                    "idx": r.subperiod_idx,
                    "sharpe_ratio": float(r.sharpe_ratio),
                    "mean_return": float(r.mean_return),
                    "volatility": float(r.volatility),
                    "train_loss": float(r.train_loss) if r.train_loss is not None else None,
                }
                for r in self.subperiod_results
            ]
        }

        # Save summary as JSON
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(data, f, indent=2)

        # Save detailed arrays
        np.savez(
            path.with_suffix(".npz"),
            returns=self.all_returns,
            turnovers=self.all_turnovers,
            short_proportions=self.all_short_proportions,
        )

    @classmethod
    def load(cls, path: Path) -> "BacktestResults":
        """
        Load results from disk.

        Args:
            path: Path to saved results

        Returns:
            BacktestResults instance
        """
        path = Path(path)

        results = cls()

        # Load JSON summary
        with open(path.with_suffix(".json"), "r") as f:
            data = json.load(f)

        results.config = data.get("config", {})

        # Load detailed arrays
        arrays = np.load(path.with_suffix(".npz"))

        # Reconstruct subperiod results (simplified)
        for sp_data in data.get("subperiods", []):
            result = SubperiodResult(
                subperiod_idx=sp_data["idx"],
                returns=np.array([]),  # Not fully reconstructed
                weights=np.array([]),
                turnovers=np.array([]),
                short_proportions=np.array([]),
                sharpe_ratio=sp_data["sharpe_ratio"],
                mean_return=sp_data["mean_return"],
                volatility=sp_data["volatility"],
                train_loss=sp_data.get("train_loss", 0.0),
            )
            results.subperiod_results.append(result)

        return results

    def __repr__(self) -> str:
        stats = self.compute_aggregate_stats()
        return (
            f"BacktestResults("
            f"subperiods={len(self.subperiod_results)}, "
            f"sharpe={stats.get('sharpe_ratio', 0):.4f}, "
            f"return={stats.get('total_return', 0):.2%})"
        )
