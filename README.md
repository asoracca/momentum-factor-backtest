# Momentum Factor Research

A reproducible study of cross-sectional equity momentum, factor exposure, and
cointegration pairs. The emphasis is honest evaluation: next-period execution,
an untouched chronological test period, transaction costs, simple benchmarks,
and conclusions that match the statistical evidence.

## Main research question

Does a fixed 12-1 momentum rule add value in a recent large-cap US equity
sample after costs and against simple alternatives?

The study does **not** claim to reproduce the original Jegadeesh and Titman
sample. It applies the same broad signal idea to a hand-selected, current-stock
universe with materially different data and implementation assumptions.

## Momentum experiment

`momentum_backtest.py`:

- computes return from month `t-12` through `t-1`;
- ranks stocks using information available at month `t`;
- holds the selected portfolio during month `t+1`;
- evaluates long-short, long-only, and equal-weight portfolios;
- charges turnover-based costs at 0, 5, 10, and 25 basis points;
- reserves the final 40% of observations as one chronological evaluation set;
- tests positive mean return using a centered circular-block bootstrap.

The first 60% is called the **development period**, not training data. The rule
has no fitted parameters, so calling this a rolling training exercise would be
misleading.

### Why the bootstrap changed

An earlier version shuffled the same monthly returns and recalculated Sharpe.
That is invalid because permutation preserves the sample mean and volatility,
so it cannot create a meaningful Sharpe null distribution.

The current test centers returns to impose a zero-mean null, then resamples
short circular blocks. This retains some local dependence while asking whether
the observed positive Sharpe is unusually large under a zero-return null.

## Run

```bash
git clone https://github.com/asoracca/momentum-factor-backtest.git
cd momentum-factor-backtest
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

python -m pytest -q
python momentum_backtest.py
```

The main script writes the current return series, benchmark comparison, cost
sensitivity table, and chart under `data/`.

## Interpreting results

Use the untouched-period numbers as the headline result. Full-sample metrics
are descriptive and should not be presented as independent validation.

A positive return or Sharpe is not automatically an edge. The stronger claim
requires all of the following:

1. positive performance in the untouched period;
2. resilience to reasonable costs;
3. improvement over long-only and equal-weight alternatives;
4. a small block-bootstrap p-value;
5. enough observations to make the estimate stable.

If these checks fail, the appropriate conclusion is **insufficient evidence of
an implementable premium in this sample**. It is not proof that momentum never
works, and it does not identify the economic cause of weak performance.

## Other studies

### Factor regression

`fama_french.py` estimates market, size, value, and momentum exposure. Its
historical demo found a large momentum loading and an alpha estimate that was
not statistically distinguishable from zero. Failing to reject zero alpha is
not the same as proving alpha is exactly zero.

When `data/strategy_returns.csv` is absent, this script constructs a separate
daily demo portfolio. Its result must not be described as a regression of the
main 200-stock monthly experiment.

### Cointegration pairs

`pairs_trading.py` selects a pair on the first 60% of observations and evaluates
it on the final 40%, using train-derived spread parameters, a Bonferroni
multiple-testing adjustment, and transaction costs. The saved V/MA snapshot had
few out-of-sample trades and did not survive the adjusted significance check,
so it is an example of a hypothesis that remained inconclusive.

## Important limitations

- The universe uses current constituents and therefore has survivorship and
  selection bias.
- Yahoo Finance data is convenient research data, not institutional-quality
  point-in-time data.
- Forward-filling and incomplete histories can distort eligibility.
- The long-short test does not model borrow availability or stock-specific
  borrow fees.
- Monthly adjusted-close returns simplify execution and market impact.
- One chronological holdout is honest but noisy; it does not establish that a
  result generalizes across every regime.
- Repeatedly modifying the strategy after viewing the holdout would make that
  period part of development data.

## Repository structure

```text
momentum_backtest.py  momentum construction, costs, alternatives, and plots
research.py           chronological split, bootstrap, and cost sensitivity
fama_french.py        separate factor-regression study
pairs_trading.py      separate cointegration study
tests/                deterministic, network-free methodology tests
```

Educational research only; not financial advice.
