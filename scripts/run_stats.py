#!/usr/bin/env python3
"""
Compute and display statistics from backtest results.

Usage:
    python run_stats.py --results-dir ./results
"""

import argparse
from pathlib import Path
import sys
import json

import numpy as np

from backtesting.results import BacktestResults
from metrics import compute_all_metrics, t_statistic, hit_rate
from utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute statistics from backtest results"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing backtest results"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for statistics (JSON)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "latex"],
        default="text",
        help="Output format"
    )

    return parser.parse_args()


def format_stats_text(stats: dict) -> str:
    """Format statistics as text."""
    lines = []
    lines.append("=" * 60)
    lines.append("BACKTEST STATISTICS")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Performance Metrics:")
    lines.append("-" * 40)
    lines.append(f"  Sharpe Ratio:       {stats['sharpe_ratio']:>10.4f}")
    lines.append(f"  Sortino Ratio:      {stats['sortino_ratio']:>10.4f}")
    lines.append(f"  Calmar Ratio:       {stats['calmar_ratio']:>10.4f}")
    lines.append(f"  Annual Return:      {stats['annualized_return']:>10.2%}")
    lines.append(f"  Annual Volatility:  {stats['annualized_volatility']:>10.2%}")
    lines.append(f"  Max Drawdown:       {stats['max_drawdown']:>10.2%}")
    lines.append(f"  Total Return:       {stats['total_return']:>10.2%}")
    lines.append("")
    lines.append("Return Statistics:")
    lines.append("-" * 40)
    lines.append(f"  Mean Daily Return:  {stats['mean_daily_return']:>10.6f}")
    lines.append(f"  Std Daily Return:   {stats['std_daily_return']:>10.6f}")
    lines.append(f"  Skewness:           {stats['skewness']:>10.4f}")
    lines.append(f"  Kurtosis:           {stats['kurtosis']:>10.4f}")
    lines.append(f"  Hit Rate:           {stats['hit_rate']:>10.2%}")
    lines.append("")
    lines.append("Statistical Tests:")
    lines.append("-" * 40)
    lines.append(f"  T-statistic:        {stats['t_stat']:>10.4f}")
    lines.append(f"  P-value:            {stats['p_value']:>10.6f}")
    lines.append("")
    lines.append("Trade Statistics:")
    lines.append("-" * 40)
    lines.append(f"  Num Periods:        {stats['num_periods']:>10d}")
    lines.append(f"  Avg Turnover:       {stats.get('avg_turnover', 0):>10.4f}")
    lines.append(f"  Avg Short Prop:     {stats.get('avg_short_proportion', 0):>10.4f}")
    lines.append("=" * 60)

    return "\n".join(lines)


def format_stats_latex(stats: dict) -> str:
    """Format statistics as LaTeX table."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Backtest Statistics}")
    lines.append(r"\begin{tabular}{lr}")
    lines.append(r"\toprule")
    lines.append(r"Metric & Value \\")
    lines.append(r"\midrule")
    lines.append(f"Sharpe Ratio & {stats['sharpe_ratio']:.4f} \\\\")
    lines.append(f"Sortino Ratio & {stats['sortino_ratio']:.4f} \\\\")
    lines.append(f"Annual Return & {stats['annualized_return']:.2%} \\\\")
    lines.append(f"Annual Volatility & {stats['annualized_volatility']:.2%} \\\\")
    lines.append(f"Max Drawdown & {stats['max_drawdown']:.2%} \\\\")
    lines.append(f"T-statistic & {stats['t_stat']:.4f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    """Main entry point."""
    args = parse_args()

    logger = setup_logger("dlsa.stats")

    results_dir = Path(args.results_dir)

    # Find all result files
    result_files = list(results_dir.glob("*.npz"))
    if not result_files:
        logger.error(f"No result files found in {results_dir}")
        return 1

    all_stats = {}

    for result_file in result_files:
        name = result_file.stem

        # Load returns
        data = np.load(result_file)
        returns = data['returns']

        logger.info(f"Processing {name}: {len(returns)} periods")

        # Compute metrics
        metrics = compute_all_metrics(returns)

        # Add additional stats
        t_stat, p_value = t_statistic(returns)
        metrics['t_stat'] = t_stat
        metrics['p_value'] = p_value
        metrics['hit_rate'] = hit_rate(returns)

        # Add turnover if available
        if 'turnovers' in data:
            metrics['avg_turnover'] = float(np.mean(data['turnovers']))
        if 'short_proportions' in data:
            metrics['avg_short_proportion'] = float(np.mean(data['short_proportions']))

        all_stats[name] = metrics

    # Output
    for name, stats in all_stats.items():
        print(f"\n{name}")

        if args.format == "text":
            print(format_stats_text(stats))
        elif args.format == "latex":
            print(format_stats_latex(stats))
        elif args.format == "json":
            print(json.dumps(stats, indent=2))

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        logger.info(f"Saved statistics to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
