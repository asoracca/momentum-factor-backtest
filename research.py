"""Statistical evaluation helpers for the momentum study."""

from __future__ import annotations

import numpy as np
import pandas as pd


def chronological_split(details, development_fraction=0.60):
    """Create one development period and one untouched evaluation period.

    The momentum rule has no fitted parameters, so the first segment is not
    described as training data.
    """
    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must be between zero and one")
    split = int(len(details) * development_fraction)
    if split < 1 or split >= len(details):
        raise ValueError("not enough observations for the requested split")
    return details.iloc[:split].copy(), details.iloc[split:].copy()


def cost_sensitivity(prices, backtest, costs=(0, 5, 10, 25), mode="long_short"):
    """Evaluate annual return and Sharpe across cost assumptions."""
    rows = []
    for cost_bps in costs:
        returns = backtest(
            prices,
            mode=mode,
            transaction_cost_bps=cost_bps,
        )
        annual_return = returns.mean() * 12
        annual_volatility = returns.std() * np.sqrt(12)
        rows.append(
            {
                "cost_bps": cost_bps,
                "annual_return": annual_return,
                "sharpe": (
                    annual_return / annual_volatility
                    if annual_volatility > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).set_index("cost_bps")


def block_bootstrap_significance(returns, n_sim=5_000, block_size=3, seed=17):
    """Test positive Sharpe with a centered circular-block bootstrap.

    Centering imposes the null of zero expected return. Sampling contiguous
    blocks preserves short-run dependence that an ordinary shuffle destroys.
    The returned p-value is one-sided because the research hypothesis is a
    positive momentum premium.
    """
    clean = pd.Series(returns).dropna().to_numpy(dtype=float)
    if len(clean) < 6:
        raise ValueError("at least six returns are required")
    if n_sim < 1:
        raise ValueError("n_sim must be positive")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    volatility = clean.std(ddof=1)
    if volatility <= 0:
        raise ValueError("returns must have positive volatility")
    actual_sharpe = clean.mean() / volatility * np.sqrt(12)
    null_returns = clean - clean.mean()
    rng = np.random.default_rng(seed)
    bootstrap_sharpes = np.empty(n_sim)
    blocks_needed = int(np.ceil(len(clean) / block_size))

    for simulation in range(n_sim):
        starts = rng.integers(0, len(clean), size=blocks_needed)
        indices = np.concatenate(
            [(np.arange(start, start + block_size) % len(clean)) for start in starts]
        )[: len(clean)]
        sample = null_returns[indices]
        sample_volatility = sample.std(ddof=1)
        bootstrap_sharpes[simulation] = (
            sample.mean() / sample_volatility * np.sqrt(12)
            if sample_volatility > 0
            else 0.0
        )

    p_value = (1 + np.sum(bootstrap_sharpes >= actual_sharpe)) / (n_sim + 1)
    return {
        "actual_sharpe": float(actual_sharpe),
        "p_value": float(p_value),
        "null_median_sharpe": float(np.median(bootstrap_sharpes)),
        "simulations": n_sim,
        "block_size": block_size,
    }


def format_bootstrap(result):
    conclusion = (
        "Evidence of positive mean return."
        if result["p_value"] < 0.05 and result["actual_sharpe"] > 0
        else "Insufficient evidence of positive mean return."
    )
    return (
        f"Actual Sharpe: {result['actual_sharpe']:.2f}\n"
        f"Null median Sharpe: {result['null_median_sharpe']:.2f}\n"
        f"One-sided p-value: {result['p_value']:.3f}\n"
        f"{conclusion}"
    )
