"""Monthly data, eligibility, formation and execution interfaces (no network I/O)."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd


class PortfolioResult(NamedTuple):
    prices: pd.DataFrame
    membership: pd.DataFrame
    eligible: pd.DataFrame
    signals: pd.DataFrame
    holdings: pd.DataFrame
    details: pd.DataFrame
    missing_held: pd.Series
    asset_returns: pd.DataFrame


@dataclass(frozen=True)
class ExperimentConfig:
    evaluation_start: str
    min_assets: int = 10
    top_pct: float = 0.2
    bottom_pct: float = 0.2
    costs: tuple = (0, 5, 10, 25)
    blocks: tuple = (1, 3, 6)
    seed: int = 17
    simulations: int = 1000
    holdout_status: str = "previously_inspected"

    def __post_init__(self):
        boundary = pd.Timestamp(self.evaluation_start)
        if pd.isna(boundary):
            raise ValueError("evaluation_start must be a date")
        object.__setattr__(self, "evaluation_start", boundary.strftime("%Y-%m-%d"))
        object.__setattr__(self, "costs", tuple(self.costs))
        object.__setattr__(self, "blocks", tuple(self.blocks))
        if (
            self.min_assets < 2
            or not 0 < self.top_pct <= 0.5
            or not 0 < self.bottom_pct <= 0.5
        ):
            raise ValueError("invalid portfolio configuration")
        if not self.costs or any(c < 0 for c in self.costs):
            raise ValueError("costs must be nonnegative")
        if not self.blocks or any(b < 1 for b in self.blocks) or self.simulations < 1:
            raise ValueError("invalid bootstrap configuration")
        if self.holdout_status not in {
            "synthetic_validation",
            "previously_inspected",
            "untouched_declared",
        }:
            raise ValueError("unknown holdout status")


def validate_prices(prices):
    """Require a complete monthly calendar; missing observations remain NaN."""
    prices = prices.copy()
    if prices.index.has_duplicates or prices.columns.has_duplicates or prices.empty:
        raise ValueError("prices must have unique dates/tickers and be nonempty")
    months = pd.DatetimeIndex(prices.index).to_period("M")
    if months.has_duplicates or not months.is_monotonic_increasing:
        raise ValueError("prices must contain ordered unique months")
    prices.index = months.to_timestamp("M")
    prices = prices.reindex(
        pd.period_range(months.min(), months.max(), freq="M").to_timestamp("M")
    )
    prices = prices.astype(float)
    if np.isinf(prices.to_numpy()).any() or (prices <= 0).any().any():
        raise ValueError("observed adjusted prices must be positive and finite")
    return prices.sort_index(axis=1)


def membership_mask(prices, membership=None):
    """Inclusive start, exclusive end; intervals must be known at their start.

    Input columns: ticker, start, end (blank = open). Rows describe membership
    effective at formation month-end, not membership during the return month.
    """
    if membership is None:
        return pd.DataFrame(True, index=prices.index, columns=prices.columns)
    mask = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    for row in membership.itertuples(index=False):
        if row.ticker not in prices.columns:
            raise ValueError(f"membership ticker missing from prices: {row.ticker}")
        start = pd.Timestamp(row.start)
        end = pd.Timestamp(row.end) if pd.notna(row.end) else pd.Timestamp.max
        if pd.isna(start) or end <= start:
            raise ValueError("invalid membership interval")
        mask.loc[(mask.index >= start) & (mask.index < end), row.ticker] = True
    return mask


def eligibility(prices, membership=None):
    member = membership_mask(prices, membership)
    # All thirteen observations through t are required; no full-sample screen.
    observed = prices.notna().rolling(13, min_periods=13).sum().eq(13)
    return member, member & observed


def form_holdings(
    scores, eligible, mode="long_short", top_pct=0.2, bottom_pct=0.2, min_assets=10
):
    if mode not in {"long_short", "long_only", "equal_weight"}:
        raise ValueError("mode must be long_short, long_only, or equal_weight")
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for date in scores.index:
        # Stable sorting after alphabetic ticker ordering makes ties deterministic.
        ranked = (
            scores.loc[date, eligible.loc[date]]
            .dropna()
            .sort_index()
            .sort_values(kind="stable")
        )
        if len(ranked) < min_assets:
            continue  # explicitly cash, including subsequent liquidation turnover
        n_top, n_bottom = (
            max(1, int(len(ranked) * top_pct)),
            max(1, int(len(ranked) * bottom_pct)),
        )
        if mode == "equal_weight":
            weights.loc[date, ranked.index] = 1 / len(ranked)
        else:
            weights.loc[date, ranked.index[-n_top:]] = 1 / n_top
            if mode == "long_short":
                weights.loc[date, ranked.index[:n_bottom]] = -1 / n_bottom
    return weights


def execute(prices, holdings, cost_bps=10):
    """Target-to-target absolute weight turnover; drift/borrow/impact unmodeled.

    Missing held returns invalidate that month's P&L, never become zero.
    Decisions at close t earn close t -> close t+1 returns.
    """
    asset_returns = prices.pct_change(fill_method=None)
    held = holdings.shift(1).iloc[13:]
    returns = asset_returns.reindex(held.index)
    missing = (held.ne(0) & returns.isna()).sum(axis=1)
    gross = (held * returns).sum(axis=1).mask(missing.gt(0))
    turnover = (
        holdings.diff().fillna(holdings).abs().sum(axis=1).shift(1).reindex(held.index)
    )
    cost = turnover * cost_bps / 10000
    details = pd.DataFrame(
        {
            "gross_return": gross,
            "turnover": turnover,
            "cost": cost,
            "net_return": gross - cost,
        }
    )
    return details, missing, asset_returns


def build_portfolio(
    prices,
    membership=None,
    mode="long_short",
    top_pct=0.2,
    bottom_pct=0.2,
    min_assets=10,
    cost_bps=10,
):
    from momentum_backtest import compute_momentum_signal

    prices = validate_prices(prices)
    member, eligible = eligibility(prices, membership)
    scores = compute_momentum_signal(prices)
    holdings = form_holdings(scores, eligible, mode, top_pct, bottom_pct, min_assets)
    details, missing, returns = execute(prices, holdings, cost_bps)
    return PortfolioResult(
        prices, member, eligible, scores, holdings, details, missing, returns
    )
