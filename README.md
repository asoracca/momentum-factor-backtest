# momentum-factor-backtest

Cross-sectional momentum factor backtest — replicating Jegadeesh & Titman (1993) on a 200-stock S&P 500 universe with walk-forward out-of-sample validation and Monte Carlo significance testing.

---

## Research Question

Does the momentum factor — buying past winners and shorting past losers — still generate alpha in the post-2012 market?

The 1993 Jegadeesh & Titman paper is one of the most cited in finance. They found that stocks returning the most over the past 12 months (skipping the most recent month) tend to keep outperforming. This project tests whether that edge persists today using a rigorous walk-forward methodology.

**Short answer: It doesn't.** Full-sample Sharpe of 0.21 matches published literature. Out-of-sample Sharpe of -0.12 matches the documented post-2012 decay from factor crowding and passive investing dominance.

---

## Strategy

- **Signal**: 12-1 month return — total return over past 12 months, skipping the most recent month (avoids short-term reversal)
- **Long**: Top 20% by momentum score (winners)
- **Short**: Bottom 20% by momentum score (losers)
- **Rebalance**: Monthly
- **Universe**: 200 S&P 500 stocks across all 11 GICS sectors

The skip-month avoids the well-documented 1-month reversal effect — stocks that just had a big month tend to give back gains short-term, which would contaminate a pure momentum signal.

---

## Methodology

### Walk-Forward Out-of-Sample Test

The strategy is evaluated using a proper walk-forward framework, not a simple in-sample backtest:

```
Train: 24 months → Test: next 6 months → Shift 6 months → Repeat
```

This produces 13 independent out-of-sample test periods. Only the test-period returns are used to evaluate performance — the strategy never sees future data during training.

**Why this matters**: An in-sample backtest on the full dataset is almost always optimistic. The walk-forward test shows what you would have actually made in real-time.

### Monte Carlo Significance

After the walk-forward test, monthly returns are shuffled 5,000 times. The Sharpe ratio of each shuffle is computed. The p-value is the fraction of shuffled Sharpes that beat the real Sharpe.

This tests whether the strategy has genuine skill or just got lucky. For negative Sharpe (OOS), a low p-value means the strategy is significantly *bad* — it consistently underperforms random.

---

## Results

```
Full Sample (10 years)
  Annualized return:    2.4%
  Sharpe ratio:         0.21     ← matches published literature
  Sortino ratio:        0.30
  Max drawdown:        -27.8%
  Win rate:             52%

Walk-Forward OOS (13 periods × 6 months)
  Annualized return:   -1.5%
  Sharpe ratio:        -0.12    ← loses money OOS
  Max drawdown:        -18.4%
  Win rate:             48%
```

The divergence between in-sample (0.21) and OOS (-0.12) is the story. It confirms the post-publication decay documented in academic literature: once momentum is widely known, factor crowding compresses the premium and liquidity events cause momentum crashes.

---

## Why Momentum Has Decayed

The strategy worked cleanly pre-2012. Post-2012, three things happened:

1. **Factor crowding**: Quant funds all run momentum. When everyone buys the same winners, the premium gets arbed away.
2. **Passive investing**: Index rebalancing creates mechanical buying/selling that works against momentum at reconstitution.
3. **Momentum crashes**: During fast reversals (2009, 2020), momentum funds need to deleverage simultaneously — causing sharp losses exactly when you're most exposed.

The OOS result is not a failure of the backtest. It is the correct result. The in-sample Sharpe of 0.21 matches Jegadeesh & Titman's original. The OOS decay matches every paper written on momentum after 2015.

---

## Files

```
momentum-factor-backtest/
├── momentum_backtest.py    ← full backtest (run this)
├── requirements.txt
└── README.md
```

---

## Run

```bash
git clone https://github.com/asoracca/momentum-factor-backtest.git
cd momentum-factor-backtest
pip install -r requirements.txt
python momentum_backtest.py
```

Runtime: ~3-4 minutes (data download + 13 OOS periods + Monte Carlo).

Output saved to `data/momentum_backtest.png`.

---

## Concepts Used

- **Cross-sectional momentum** (Jegadeesh & Titman 1993): rank stocks by past return, long top quintile, short bottom quintile
- **Walk-forward validation**: proper train/test split to avoid look-ahead bias
- **Monte Carlo significance**: shuffle P&L 5,000 times, compute p-value on Sharpe ratio
- **Sortino ratio**: like Sharpe but only penalizes downside volatility (ignores upside vol, which isn't risk)
- **Max drawdown**: largest peak-to-trough loss — the number that actually keeps you up at night

---

## Limitations

- **Long-only versions work better**: The short leg is hard to execute (borrow costs, uptick rule). A long-only momentum ETF (MTUM) would be a fairer benchmark.
- **Survivorship bias**: Universe built from current S&P 500 members. Stocks that got delisted mid-period are excluded, which biases results positively.
- **Transaction costs ignored**: Monthly rebalancing across 200 stocks incurs real friction — bid/ask spread, market impact, commissions.
- **Factor timing not modeled**: Momentum performs better in trending markets, worse in choppy/reverting markets. A regime filter could improve OOS performance.

---

## Planned Upgrades

- Fama-French 3-factor regression: decompose returns into market, size, value exposure — isolate pure momentum alpha
- Regime filter: run momentum only when market trend is strong (e.g., SPY > 200-day MA)
- Volatility scaling: size positions inversely to each stock's realized vol (improves Sharpe)
- Long-only variant: compare against MTUM ETF as benchmark

---

*Built as a quant portfolio project. Not financial advice.*
