"""
momentum_backtest.py
--------------------
Cross-sectional momentum factor backtest.
Strategy: Jegadeesh & Titman (1993)
  - Rank stocks by 12-1 month return (last 12 months, skip most recent month)
  - Long top 20% (winners), short bottom 20% (losers)
  - Rebalance monthly
  - Walk-forward test: train 24mo, test 6mo rolling

Universe: 200 S&P 500 stocks across all 11 GICS sectors
Period:   10 years (for sufficient OOS periods and statistical power)

Run:
    python momentum_backtest.py
"""

from pathlib import Path
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

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
def run_backtest(prices, top_pct=0.20, bottom_pct=0.20):
    """
    Monthly rebalance: long top quintile, short bottom quintile.
    Equal-weight within each leg.
    Returns a Series of monthly portfolio returns.
    """
    momentum = compute_momentum_signal(prices)
    monthly_returns = prices.pct_change()

    portfolio_returns = []
    dates = []

    for i in range(13, len(prices) - 1):
        scores = momentum.iloc[i].dropna()
        if len(scores) < 10:
            continue

        n_top = max(1, int(len(scores) * top_pct))
        n_bottom = max(1, int(len(scores) * bottom_pct))

        winners = scores.nlargest(n_top).index.tolist()
        losers = scores.nsmallest(n_bottom).index.tolist()

        # Next month returns (forward-looking, correct: signal at t, returns at t+1)
        next_ret = monthly_returns.iloc[i + 1]

        long_ret = next_ret[winners].mean()
        short_ret = next_ret[losers].mean()
        port_ret = long_ret - short_ret  # long-short portfolio

        portfolio_returns.append(port_ret)
        dates.append(prices.index[i + 1])

    return pd.Series(portfolio_returns, index=dates, name="momentum_returns")


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

    if sharpe >= 0.5:
        print("  ✅ Solid edge — strategy has positive risk-adj return")
    elif sharpe >= 0.2:
        print("  ⚠️  Weak edge — works but barely")
    else:
        print("  ❌ No edge detected in this period")

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "win_rate": win_rate,
    }


# ── Walk-forward ─────────────────────────────────────────────────────────────
def walk_forward_test(
    prices, train_months=24, test_months=6, top_pct=0.20, bottom_pct=0.20
):
    """
    Rolling walk-forward: train on 24 months, test on next 6, shift by 6, repeat.
    Only out-of-sample (test) returns are kept.
    """
    momentum = compute_momentum_signal(prices)
    monthly_returns = prices.pct_change()

    all_oos_returns = []
    n = len(prices)

    period = 0
    start = 13  # need 12 months for signal to warm up

    while start + train_months + test_months <= n:
        train_end = start + train_months
        test_end = train_end + test_months

        test_returns = []
        test_dates = []

        for i in range(train_end, test_end - 1):
            scores = momentum.iloc[i].dropna()
            if len(scores) < 10:
                continue

            n_top = max(1, int(len(scores) * top_pct))
            n_bottom = max(1, int(len(scores) * bottom_pct))

            winners = scores.nlargest(n_top).index.tolist()
            losers = scores.nsmallest(n_bottom).index.tolist()

            next_ret = monthly_returns.iloc[i + 1]
            long_ret = next_ret[winners].mean()
            short_ret = next_ret[losers].mean()
            port_ret = long_ret - short_ret

            test_returns.append(port_ret)
            test_dates.append(prices.index[i + 1])

        if test_returns:
            period_series = pd.Series(test_returns, index=test_dates)
            period_series.name = f"Period_{period}"
            all_oos_returns.extend(test_returns)

            w = (period_series > 0).sum()
            print(
                f"  OOS Period {period}: {len(period_series)} months | "
                f"win {w / len(period_series):.0%} | "
                f"return {period_series.mean() * 12:.1%} ann"
            )

        period += 1
        start += test_months  # roll forward

    return pd.Series(all_oos_returns, name="oos_returns")


# ── Monte Carlo ──────────────────────────────────────────────────────────────
def monte_carlo_significance(returns, n_sim=5000):
    """Shuffle monthly returns 5000x. What % of random Sharpes beat yours?"""
    if len(returns) < 6:
        return

    actual_sharpe = returns.mean() / returns.std() * np.sqrt(12)
    shuffle_sharpes = []

    for _ in range(n_sim):
        shuffled = returns.sample(frac=1).values
        s = shuffled.mean() / shuffled.std() * np.sqrt(12) if shuffled.std() > 0 else 0
        shuffle_sharpes.append(s)

    p_val = (np.array(shuffle_sharpes) >= actual_sharpe).mean()
    print(f"\n── Monte Carlo ({n_sim:,} simulations) ──────────────────")
    print(f"  Actual Sharpe:     {actual_sharpe:.2f}")
    print(f"  Median random:     {np.median(shuffle_sharpes):.2f}")
    print(f"  p-value:           {p_val:.3f}")
    if actual_sharpe <= 0:
        print("  ❌ Negative Sharpe — strategy LOSES money OOS")
        print(f"     p={p_val:.3f} means {p_val:.0%} of random shuffles are even worse")
        print("     This strategy is significantly bad, not significantly good")
    elif p_val < 0.05:
        print("  ✅ Statistically significant — edge is real (p < 0.05)")
    elif p_val < 0.10:
        print("  ⚠️  Marginal significance (p < 0.10)")
    else:
        print(f"  ❌ Not significant — {p_val:.0%} of random strategies match this")


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
    print("\n" + "=" * 55)
    print("  MOMENTUM FACTOR BACKTEST")
    print("  Strategy: Jegadeesh & Titman (1993)")
    print("  Long top 20% | Short bottom 20% | Monthly rebalance")
    print("=" * 55)

    prices = fetch_prices(period="10y")

    # Full-sample backtest
    full_returns = run_backtest(prices)
    compute_stats(full_returns, label="FULL SAMPLE BACKTEST")

    # Walk-forward OOS
    print("\n── Walk-Forward OOS (train 24mo → test 6mo) ──────────")
    oos_returns = walk_forward_test(prices, train_months=24, test_months=6)
    compute_stats(oos_returns, label="WALK-FORWARD OOS RESULTS")

    # Monte Carlo on OOS
    monte_carlo_significance(oos_returns)

    # Plot
    plot_results(full_returns, oos_returns)

    print("\nDone. Check data/momentum_backtest.png for charts.")
