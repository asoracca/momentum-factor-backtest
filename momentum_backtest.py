"""
momentum_backtest.py
--------------------
Cross-sectional momentum factor backtest.
Strategy: Jegadeesh & Titman (1993)
  - Rank stocks by 12-1 month return (last 12 months, skip most recent month)
  - Long top 20% (winners), short bottom 20% (losers)
  - Rebalance monthly
  - Chronological 60% development / 40% untouched evaluation split
  - Long-only and equal-weight alternatives
  - Turnover-based transaction costs and cost sensitivity
  - Centered circular-block bootstrap significance test

Universe: 200 S&P 500 stocks across all 11 GICS sectors
Period:   10 years (for sufficient OOS periods and statistical power)

Run:
    python momentum_backtest.py
"""

import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Universe — 200 S&P 500 stocks across all 11 GICS sectors ────────────────
UNIVERSE = [
    # Information Technology (30)
    "AAPL",
    "MSFT",
    "NVDA",
    "AVGO",
    "AMD",
    "QCOM",
    "TXN",
    "AMAT",
    "LRCX",
    "KLAC",
    "MU",
    "INTC",
    "ADI",
    "MRVL",
    "NXPI",
    "ON",
    "STX",
    "WDC",
    "HPQ",
    "KEYS",
    "FTNT",
    "PANW",
    "CRWD",
    "ZS",
    "SNPS",
    "CDNS",
    "ANSS",
    "PTC",
    "NTAP",
    "FSLR",
    # Communication Services (15)
    "GOOGL",
    "META",
    "NFLX",
    "DIS",
    "CMCSA",
    "T",
    "VZ",
    "TMUS",
    "CHTR",
    "EA",
    "TTWO",
    "MTCH",
    "LYV",
    "IPG",
    "OMC",
    # Consumer Discretionary (20)
    "AMZN",
    "TSLA",
    "HD",
    "MCD",
    "NKE",
    "SBUX",
    "LOW",
    "TJX",
    "BKNG",
    "MAR",
    "HLT",
    "RCL",
    "CCL",
    "YUM",
    "DRI",
    "EXPE",
    "EBAY",
    "ETSY",
    "ROST",
    "ULTA",
    # Consumer Staples (15)
    "WMT",
    "COST",
    "PG",
    "KO",
    "PEP",
    "PM",
    "MO",
    "CL",
    "KMB",
    "GIS",
    "CPB",
    "SJM",
    "HRL",
    "CAG",
    "MKC",
    # Financials (25)
    "JPM",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "C",
    "AXP",
    "BLK",
    "SCHW",
    "CB",
    "MMC",
    "AON",
    "TRV",
    "AFL",
    "MET",
    "PRU",
    "ALL",
    "PGR",
    "ICE",
    "CME",
    "MCO",
    "SPGI",
    "FDS",
    "BR",
    "AMP",
    # Healthcare (25)
    "UNH",
    "LLY",
    "JNJ",
    "ABBV",
    "MRK",
    "PFE",
    "TMO",
    "ABT",
    "DHR",
    "BMY",
    "AMGN",
    "GILD",
    "ISRG",
    "SYK",
    "BSX",
    "MDT",
    "EW",
    "ZBH",
    "BAX",
    "BDX",
    "IQV",
    "CRL",
    "IDXX",
    "MTD",
    "PODD",
    # Industrials (20)
    "CAT",
    "DE",
    "BA",
    "GE",
    "HON",
    "RTX",
    "LMT",
    "NOC",
    "GD",
    "LHX",
    "EMR",
    "ITW",
    "PH",
    "ROK",
    "ETN",
    "IR",
    "XYL",
    "ROP",
    "CTAS",
    "FAST",
    # Energy (15)
    "XOM",
    "CVX",
    "COP",
    "EOG",
    "SLB",
    "PSX",
    "VLO",
    "MPC",
    "OXY",
    "HES",
    "DVN",
    "FANG",
    "HAL",
    "BKR",
    "APA",
    # Materials (10)
    "LIN",
    "APD",
    "SHW",
    "ECL",
    "DD",
    "PPG",
    "NEM",
    "FCX",
    "NUE",
    "VMC",
    # Real Estate (10)
    "AMT",
    "PLD",
    "CCI",
    "EQIX",
    "PSA",
    "DLR",
    "O",
    "WELL",
    "SPG",
    "EQR",
    # Utilities (10)
    "NEE",
    "DUK",
    "SO",
    "D",
    "AEP",
    "EXC",
    "SRE",
    "XEL",
    "PCG",
    "ED",
    # CRM / Cloud / Software (5 extra)
    "CRM",
    "NOW",
    "ADBE",
    "INTU",
    "ORCL",
]


# ── Data ────────────────────────────────────────────────────────────────────
def fetch_prices(period="10y"):
    """Download monthly adjusted close prices for the universe."""
    import yfinance as yf

    print(f"Fetching prices for {len(UNIVERSE)} stocks ({period})...")
    raw = yf.download(
        UNIVERSE, period=period, interval="1mo", progress=False, auto_adjust=True
    )

    # Flatten MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = ["_".join(c).strip("_") for c in raw.columns]

    # Extract Close columns
    close_cols = [c for c in raw.columns if c.startswith("Close")]
    prices = raw[close_cols].copy()
    prices.columns = [c.replace("Close_", "") for c in prices.columns]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.dropna(how="all")

    # Drop any ticker that has >20% missing data
    keep = prices.columns[prices.isna().mean() < 0.2]
    prices = prices[keep].ffill()

    print(f"  Got {len(prices)} months × {len(prices.columns)} stocks")
    return prices


# ── Signal ──────────────────────────────────────────────────────────────────
def compute_momentum_signal(prices, formation_months=12, skip_months=1):
    """
    12-1 momentum: return over past 12 months, skipping the most recent month.
    Skipping the last month avoids short-term reversal noise.

    Returns DataFrame of momentum scores (same shape as prices).
    """
    # Total return over formation window, skip most recent month
    lagged = prices.shift(skip_months)  # price 1 month ago
    past_price = prices.shift(formation_months)  # price 12 months ago
    momentum = (lagged - past_price) / past_price  # % return 12-1 months ago
    return momentum


# ── Backtest ────────────────────────────────────────────────────────────────
def run_backtest(
    prices,
    top_pct=0.20,
    bottom_pct=0.20,
    mode="long_short",
    transaction_cost_bps=10.0,
    return_details=False,
):
    """
    Monthly rebalance: long top quintile, short bottom quintile.
    Equal-weight within each leg.
    ``mode`` may be ``long_short``, ``long_only``, or ``equal_weight``.
    Costs are charged on one-way turnover. Returns are net of costs by default.
    """
    momentum = compute_momentum_signal(prices)
    monthly_returns = prices.pct_change()

    rows = []
    dates = []
    previous_weights = pd.Series(dtype=float)

    for i in range(12, len(prices) - 1):
        scores = momentum.iloc[i].dropna()
        if len(scores) < 10:
            continue

        n_top = max(1, int(len(scores) * top_pct))
        winners = scores.nlargest(n_top).index.tolist()
        weights = pd.Series(0.0, index=scores.index)
        if mode == "long_short":
            n_bottom = max(1, int(len(scores) * bottom_pct))
            losers = scores.nsmallest(n_bottom).index.tolist()
            weights.loc[winners] = 1.0 / len(winners)
            weights.loc[losers] = -1.0 / len(losers)
        elif mode == "long_only":
            weights.loc[winners] = 1.0 / len(winners)
        elif mode == "equal_weight":
            weights.loc[:] = 1.0 / len(weights)
        else:
            raise ValueError("mode must be long_short, long_only, or equal_weight")

        # Next month returns (forward-looking, correct: signal at t, returns at t+1)
        next_ret = monthly_returns.iloc[i + 1]

        all_assets = weights.index.union(previous_weights.index)
        current_aligned = weights.reindex(all_assets, fill_value=0.0)
        previous_aligned = previous_weights.reindex(all_assets, fill_value=0.0)
        turnover = (current_aligned - previous_aligned).abs().sum()
        gross_return = (weights * next_ret.reindex(weights.index).fillna(0.0)).sum()
        cost = turnover * transaction_cost_bps / 10_000
        rows.append(
            {
                "gross_return": gross_return,
                "turnover": turnover,
                "cost": cost,
                "net_return": gross_return - cost,
            }
        )
        dates.append(prices.index[i + 1])
        previous_weights = weights

    details = pd.DataFrame(rows, index=dates)
    details.index.name = "date"
    if return_details:
        return details
    return details["net_return"].rename(f"{mode}_returns")


# ── Stats ────────────────────────────────────────────────────────────────────
def compute_stats(returns, label="Momentum"):
    """Print Sharpe, Sortino, max drawdown, win rate, annualized return."""
    if returns.empty or len(returns) < 6:
        print(f"  Not enough data for {label}")
        return {}

    ann_return = returns.mean() * 12
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(12)
    sortino = ann_return / downside if downside > 0 else 0

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    win_rate = (returns > 0).mean()
    total_ret = (1 + returns).prod() - 1

    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}")
    print(f"  Months of data:       {len(returns)}")
    print(f"  Win rate:             {win_rate:.1%}")
    print(f"  Annualized return:    {ann_return:.1%}")
    print(f"  Annualized vol:       {ann_vol:.1%}")
    print(f"  Sharpe ratio:         {sharpe:.2f}")
    print(f"  Sortino ratio:        {sortino:.2f}")
    print(f"  Max drawdown:         {max_dd:.1%}")
    print(f"  Total return:         {total_ret:.1%}")
    print(f"{'=' * 55}")

    print("  Descriptive metrics only; use the holdout bootstrap for inference.")

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "win_rate": win_rate,
    }


# ── Plot ─────────────────────────────────────────────────────────────────────
def plot_results(full_returns, oos_returns):
    """3-panel chart: cumulative return, monthly returns, drawdown."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=False)
    fig.suptitle(
        "Momentum Factor Backtest — L/S Cross-Sectional Momentum",
        fontsize=14,
        fontweight="bold",
    )

    # Panel 1: Cumulative returns
    cum_full = (1 + full_returns).cumprod()
    cum_oos = (1 + oos_returns).cumprod()

    axes[0].plot(
        cum_full.index,
        cum_full.values,
        color="#1D9E75",
        linewidth=1.8,
        label="Full sample",
    )
    axes[0].plot(
        oos_returns.index,
        cum_oos.values,
        color="#A32D2D",
        linewidth=1.8,
        linestyle="--",
        label="OOS only",
    )
    axes[0].axhline(1, color="black", linewidth=0.8, alpha=0.4)
    axes[0].set_ylabel("Growth of $1")
    axes[0].set_title("Cumulative Return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    # Panel 2: Monthly returns bar
    colors = ["#1D9E75" if r >= 0 else "#A32D2D" for r in full_returns]
    axes[1].bar(
        full_returns.index, full_returns.values, color=colors, alpha=0.7, width=20
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Monthly Return")
    axes[1].set_title("Monthly P&L")
    axes[1].grid(True, alpha=0.25)

    # Panel 3: Drawdown
    cum = (1 + full_returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    axes[2].fill_between(dd.index, dd.values, 0, color="#A32D2D", alpha=0.4)
    axes[2].set_ylabel("Drawdown")
    axes[2].set_title("Underwater Curve (Drawdown from Peak)")
    axes[2].grid(True, alpha=0.25)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    Path("data").mkdir(exist_ok=True)
    plt.savefig("data/momentum_backtest.png", dpi=150, bbox_inches="tight")
    print("\nSaved: data/momentum_backtest.png")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from research import (
        block_bootstrap_significance,
        chronological_split,
        cost_sensitivity,
        format_bootstrap,
    )

    print("\n" + "=" * 55)
    print("  MOMENTUM FACTOR BACKTEST")
    print("  Strategy: Jegadeesh & Titman (1993)")
    print("  Long top 20% | Short bottom 20% | Monthly rebalance")
    print("=" * 55)

    prices = fetch_prices(period="10y")

    # Full-sample backtest
    full_details = run_backtest(prices, return_details=True)
    development, evaluation = chronological_split(full_details)
    full_returns = full_details["net_return"]
    oos_returns = evaluation["net_return"]
    compute_stats(full_returns, label="FULL SAMPLE, NET OF 10 BPS COSTS")
    compute_stats(oos_returns, label="UNTOUCHED 40% EVALUATION PERIOD")

    print("\n── Untouched evaluation and simple alternatives ──────────")
    long_only = run_backtest(prices, mode="long_only")
    equal_weight = run_backtest(prices, mode="equal_weight")
    _, long_only_oos = chronological_split(long_only.to_frame("net_return"))
    _, equal_weight_oos = chronological_split(equal_weight.to_frame("net_return"))
    compute_stats(long_only_oos["net_return"], label="LONG-ONLY MOMENTUM OOS")
    compute_stats(equal_weight_oos["net_return"], label="EQUAL-WEIGHT OOS")

    # Statistical test and cost sensitivity on the untouched period.
    bootstrap = block_bootstrap_significance(oos_returns)
    print("\nCENTERED BLOCK BOOTSTRAP")
    print(format_bootstrap(bootstrap))
    sensitivity = cost_sensitivity(prices, run_backtest)
    print("\nTRANSACTION-COST SENSITIVITY")
    print(sensitivity.round(3))

    Path("data").mkdir(exist_ok=True)
    full_details.to_csv("data/momentum_returns.csv")
    pd.concat(
        [
            oos_returns.rename("long_short"),
            long_only_oos["net_return"].rename("long_only"),
            equal_weight_oos["net_return"].rename("equal_weight"),
        ],
        axis=1,
    ).to_csv("data/oos_comparison.csv")
    sensitivity.to_csv("data/cost_sensitivity.csv")

    # Plot
    plot_results(full_returns, oos_returns)

    print("\nDone. Check data/momentum_backtest.png for charts.")
