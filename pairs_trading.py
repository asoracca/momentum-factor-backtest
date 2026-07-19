"""
pairs_trading.py — Rigorous cointegration stat-arb
--------------------------------------------------
Upgrades the naive version with the three things that separate a real study
from an overfit backtest:

  1. OUT-OF-SAMPLE split: select the pair on the TRAIN half, trade it only on
     the unseen TEST half (z-score uses train-derived mean/std/beta — no peeking).
  2. MULTIPLE-TESTING honesty: report the Bonferroni-adjusted p (raw p x #pairs).
  3. TRANSACTION COSTS: charge ~10 bps per position change and report net Sharpe.

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

COST = 0.0010  # ~10 bps charged on each unit of position change (both legs)


def sharpe(s):
    return np.sqrt(252) * s.mean() / s.std() if s.std() > 0 else 0.0


def backtest(a, b, prices, beta, mu, sd):
    """Trade the spread on `prices` using TRAIN-derived beta/mu/sd. Returns gross & net."""
    spread = prices[a] - beta * prices[b]
    z = (spread - mu) / sd
    ret_a, ret_b = prices[a].pct_change(), prices[b].pct_change()
    pos, gross, net, positions = 0, [], [], []
    for i in range(1, len(prices)):
        zi = z.iloc[i - 1]
        prev = pos
        if pos == 0:
            if zi > 2:
                pos = -1
            elif zi < -2:
                pos = 1
        elif pos == 1 and zi >= -0.5:
            pos = 0
        elif pos == -1 and zi <= 0.5:
            pos = 0
        g = pos * (ret_a.iloc[i] - ret_b.iloc[i])
        c = abs(pos - prev) * COST
        gross.append(g)
        net.append(g - c)
        positions.append(pos)
    idx = prices.index[1:]
    return (
        pd.Series(gross, index=idx).fillna(0.0),
        pd.Series(net, index=idx).fillna(0.0),
        positions,
        z,
    )


def main():
    tickers = [
        "KO",
        "PEP",
        "XOM",
        "CVX",
        "JPM",
        "BAC",
        "HD",
        "LOW",
        "V",
        "MA",
        "GOOGL",
        "MSFT",
    ]
    px = yf.download(tickers, start="2017-01-01", auto_adjust=True, progress=False)[
        "Close"
    ].dropna()
    n = len(px)
    split = int(n * 0.6)
    train, test = px.iloc[:split], px.iloc[split:]
    pairs = list(itertools.combinations(tickers, 2))
    print(
        f"Loaded {n} days. Train {len(train)} / Test {len(test)}. Scanning {len(pairs)} pairs.\n"
    )

    best = None
    for a, b in pairs:
        try:
            _, p, _ = coint(train[a], train[b])
        except Exception:
            continue
        if best is None or p < best[2]:
            best = (a, b, p)
    a, b, p = best
    p_bonf = min(1.0, p * len(pairs))
    print(f"Best pair on TRAIN: {a} & {b}")
    print(f"  raw cointegration p   = {p:.4f}")
    print(f"  Bonferroni-adjusted p = {p_bonf:.4f}   (x{len(pairs)} pairs tested)")
    print(
        "  -> survives multiple testing."
        if p_bonf < 0.05
        else "  -> does NOT survive multiple testing (likely data-mined)."
    )

    model = sm.OLS(train[a], sm.add_constant(train[b])).fit()
    beta = model.params[b]
    spread_tr = train[a] - beta * train[b]
    mu, sd = spread_tr.mean(), spread_tr.std()
    print(f"  hedge ratio (train): {beta:.3f}\n")

    _, n_tr, _, _ = backtest(a, b, train, beta, mu, sd)
    g_te, n_te, pos_te, z_te = backtest(a, b, test, beta, mu, sd)
    ntr = sum(
        1
        for j in range(1, len(pos_te))
        if pos_te[j] != pos_te[j - 1] and pos_te[j] != 0
    )

    print("=" * 60)
    print(f"  IN-SAMPLE (train):                  Sharpe {sharpe(n_tr):+.2f}")
    print(f"  OUT-OF-SAMPLE (test, net of costs): Sharpe {sharpe(n_te):+.2f}")
    print(
        f"     gross Sharpe {sharpe(g_te):+.2f} | trades {ntr} | net return {((1 + n_te).prod() - 1) * 100:+.1f}%"
    )
    print("=" * 60)
    if sharpe(n_te) > 0.5 and p_bonf < 0.05 and ntr >= 10:
        print(
            "  Holds up OOS, after costs, and survives multiple testing — genuinely interesting."
        )
    else:
        print(
            "  Verdict: collapses once you demand OOS + costs + multiple-testing honesty."
        )
        print(
            "  That is the realistic outcome for most 'discovered' pairs — and the correct finding."
        )

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(z_te.index, z_te)
    for lvl, c in [(2, "r"), (-2, "g"), (0, "gray")]:
        ax[0].axhline(lvl, ls="--", c=c, alpha=0.5)
    ax[0].set_title(f"{a}-{b} spread z-score — OUT-OF-SAMPLE (traded on unseen data)")
    ax[1].plot(n_te.index, (1 + n_te).cumprod(), label="net of costs")
    ax[1].plot(g_te.index, (1 + g_te).cumprod(), ls="--", alpha=0.6, label="gross")
    ax[1].legend()
    ax[1].set_title("Out-of-sample cumulative return")
    plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/pairs_oos.png", dpi=110)
    print("  Saved: data/pairs_oos.png")


if __name__ == "__main__":
    main()
