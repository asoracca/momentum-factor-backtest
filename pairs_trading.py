"""
pairs_trading.py — Cointegration-based statistical arbitrage (market-neutral)
-----------------------------------------------------------------------------
Correlation = moves together short-term. COINTEGRATION = a stable long-run
relationship, so the SPREAD between two prices mean-reverts. We:

  1. Scan a universe for the most cointegrated pair (Engle-Granger test).
  2. Estimate the hedge ratio (OLS) and build the spread + its z-score.
  3. Backtest: when |z| > 2 (spread stretched), short the rich leg / long the
     cheap leg; exit when |z| < 0.5. Long one + short the other = market-neutral.

    python pairs_trading.py
"""

import os
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    tickers = ["KO", "PEP", "XOM", "CVX", "JPM", "BAC", "HD", "LOW", "V", "MA", "GOOGL", "MSFT"]
    px = yf.download(tickers, start="2019-01-01", auto_adjust=True, progress=False)["Close"].dropna()
    print(f"Loaded {len(px)} days for {len(tickers)} tickers.\n")

    # 1. find the most cointegrated pair
    best = None
    for a, b in itertools.combinations(tickers, 2):
        try:
            _, pval, _ = coint(px[a], px[b])
        except Exception:
            continue
        if best is None or pval < best[2]:
            best = (a, b, pval)
    a, b, pval = best
    print(f"Most cointegrated pair: {a} & {b}   Engle-Granger p = {pval:.4f}")
    print("  (p < 0.05 = statistically cointegrated; spread should mean-revert.)" if pval < 0.05
          else "  (p > 0.05 = weak cointegration; treat results as illustrative.)")

    # 2. hedge ratio + spread z-score
    model = sm.OLS(px[a], sm.add_constant(px[b])).fit()
    beta = model.params[b]
    spread = px[a] - beta * px[b]
    z = (spread - spread.mean()) / spread.std()
    print(f"  Hedge ratio (beta): {beta:.3f}")

    # 3. backtest (use yesterday's z -> no look-ahead; dollar-neutral legs)
    ret_a, ret_b = px[a].pct_change(), px[b].pct_change()
    pos, daily, positions = 0, [], []
    for i in range(1, len(px)):
        zi = z.iloc[i - 1]
        if pos == 0:
            if zi > 2: pos = -1
            elif zi < -2: pos = 1
        elif pos == 1 and zi >= -0.5: pos = 0
        elif pos == -1 and zi <= 0.5: pos = 0
        daily.append(pos * (ret_a.iloc[i] - ret_b.iloc[i]))
        positions.append(pos)
    s = pd.Series(daily, index=px.index[1:]).fillna(0.0)

    sharpe = np.sqrt(252) * s.mean() / s.std() if s.std() > 0 else 0.0
    total = (1 + s).prod() - 1
    ntrades = sum(1 for j in range(1, len(positions)) if positions[j] != positions[j - 1] and positions[j] != 0)
    print("=" * 56)
    print(f"  PAIRS BACKTEST: long/short {a} vs {b}")
    print(f"  Entries: {ntrades} | Sharpe: {sharpe:.2f} | Total return: {total*100:+.1f}%")
    print("=" * 56)
    print("  Market-neutral Sharpe > 1 is promising — validate with trading costs and" if sharpe > 1
          else "  Modest result — and per your Fama-French lesson, check it's not just factor noise.")
    print("  out-of-sample data before believing it." if sharpe > 1 else "")

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(z.index, z); ax[0].axhline(2, ls="--", c="r", alpha=.5); ax[0].axhline(-2, ls="--", c="g", alpha=.5); ax[0].axhline(0, c="gray", alpha=.4)
    ax[0].set_title(f"{a} - {b} spread z-score (enter at +/-2)")
    ax[1].plot(s.index, (1 + s).cumprod()); ax[1].set_title("Strategy cumulative return (market-neutral)")
    plt.tight_layout(); os.makedirs("data", exist_ok=True); plt.savefig("data/pairs_trading.png", dpi=110)
    print("  Saved: data/pairs_trading.png")


if __name__ == "__main__":
    main()
