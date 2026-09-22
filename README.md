# Momentum Factor Research

An auditable monthly 12-1 momentum experiment with offline fixtures, point-in-time
membership input, explicit missing coverage, costs, and cautious inference.
The real default universe is **survivorship-biased current stocks**, not historical
index constituents. Synthetic membership validates software; it does not fix that sample.

## Reproduce offline

Python 3.12; no credentials or network needed after dependency installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
python -m pytest -q
python experiment.py offline
```

The command writes a small comparison table, `timing.png`, return CSVs with metadata,
a frozen manifest, coverage and attribution tables, and `experiments.sqlite` under
`outputs/`. Every run has a separate directory. Seed 17; 1,000 bootstrap replicates.
CI runs tests, formatting, lint, and this same demo. `python demo.py` is a shortcut.

Measured synthetic example (24 evaluation months, January 2023–December 2024,
10 bps per unit of absolute target-weight turnover):

| Portfolio | Annualized arithmetic mean | Mean monthly turnover | Sum of cost fractions |
|---|---:|---:|---:|
| Long-short | 0.2490 | 0.083333 | 0.0020 |
| Long-only | 0.1290 | 0.083333 | 0.0020 |
| Equal-weight | 0.0035 | 0.020833 | 0.0005 |

These intentionally simple toy paths are **not trading performance**. See
[fixture calculations](fixtures/README.md) for expected rankings, holdings and turnover.

## Real-data experiment

This separate command downloads complete monthly Yahoo adjusted closes, preserving
failed tickers and missing observations. Network and provider availability are required.
Freeze dates and parameters before examining output:

```bash
python experiment.py live --start 2015-01-01 --end 2026-09-01 \
  --evaluation-start 2022-01-31 --output outputs/real
```

A measured September 22, 2026 run had 56 evaluation months. At 10 bps,
long-short's annualized arithmetic mean was 3.59% (block-3 95% interval
−12.97% to 21.93%, one-sided p=0.338), versus 19.52% long-only and 14.06%
equal-weight. Long-short's mean fell from 4.71% at zero costs to 1.92% at
25 bps. Yahoo returned no prices for ANSS, IPG, MMC and HES; they were recorded
as missing. This inspected, selected sample provides insufficient evidence of a
positive long-short premium. Equal-weight's zero modeled turnover within this
period reflects the target-weight approximation, which ignores weight drift.

`--end` is exclusive. The default status is **previously inspected**, so the result is
exploratory. Use `--untouched-declared` only for a genuinely uninspected period selected
before tuning; the software records this declaration but cannot verify research history.
An inspected holdout never becomes fresh validation because the code changed.

To use an archived data source and optional historical membership:

```bash
python experiment.py csv --prices /path/prices.csv \
  --membership /path/membership.csv --source-timestamp 2026-09-01T00:00:00Z \
  --adjustment-policy 'provider total-return adjusted closes' \
  --evaluation-start 2022-01-31 --output outputs/historical
```

Prices: `date,TICKER,...`, one row per month, positive observed levels, blanks for
missing data. Membership: `ticker,start,end`, start inclusive/end exclusive, blank
end for open intervals. Membership must be known as of formation; effective dates
alone cannot prove this. Include exited stocks and their terminal returns in the price
source. The live command also accepts `--membership`; no authentic historical
constituent dataset is bundled. Source revision timestamps unavailable from Yahoo
remain explicitly unknown; retrieval timestamps are recorded separately.

## Methods and architecture

- `pipeline.py`: monthly validation, trailing eligibility, stable ranking, holdings,
  and next-period execution; no full-sample availability filter or price filling.
- `momentum_backtest.py`: existing universe and 12-1 signal; its backtest interface
  delegates to the same pipeline. Its CLI delegates to `experiment.py`.
- `research.py`: chronological split, centered circular-block null and evaluation.
- `experiment.py`: frozen configuration, hashes, source metadata, versions, artifacts.
- `store.py`: versioned SQLite tables and parameterized coverage/comparison/audit joins.
- `demo.py`: timing diagram. `fama_french.py`: explicitly selected artifact regression.

At formation close t, rank `P[t-1]/P[t-12]-1`; require all 13 prices through t and
membership at t. Hold during t→t+1. Long-short has +1 long and −1 short exposure.
If fewer than the minimum eligible names exist, the portfolio targets cash. Costs
include initial entry and later target changes, using sum(abs(new−old)); drift,
borrow fees/availability, financing, slippage and market impact are unmodeled.
Monthly adjusted closes are a simplified execution model, and adjustments may be
revised retrospectively. Terminal recoveries are never invented.

Changing every price after a cutoff and adding future membership must leave earlier
signals, eligibility and holdings unchanged; the deterministic leakage test enforces
this. A missing held return makes that portfolio month unknown. Missing periods are
counted, not dropped to create a misleading continuous bootstrap sample.

The fixed calendar evaluation compares all three portfolios on identical dates,
with cost levels 0/5/10/25 bps and circular block lengths 1/3/6. The primary specification
is long-short, 10 bps, block 3: a one-sided positive-mean test under a centered
zero-mean null. Uncentered block resamples give a 95% percentile interval for the
annualized arithmetic mean. Other specifications are exploratory sensitivity checks,
not additional opportunities to select a significant finding. No family-wide
significance claim is made. Short, constant or incomplete samples receive no p-value.
See [SQL and artifact details](docs/artifacts.md).

## Factor regression

No automatic daily fallback. Select a return CSV produced by the experiment and
an independently sourced factor CSV covering all selected dates:

```bash
python fama_french.py --returns outputs/RUN_ID/long_short_returns.csv \
  --factors /path/monthly_factors.csv --start 2023-01-31
```

Factor columns: `date,Mkt-RF,SMB,HML,MOM,RF`, in decimal units. Adjacent JSON must
contain `frequency: "monthly"`, `units: "decimal"`, `source`, and the CSV's SHA-256
in `sha256`. Frequency, hashes, missing dates, nonfinite values and rank-deficient
designs fail clearly. Total-return portfolios subtract RF; zero-investment long-short
spreads do not subtract it again. Estimates use HAC errors (3 monthly or 5 daily lags),
report sample size, alpha interval and p-value, and preserve negative results.
A nonsignificant alpha does not prove alpha is zero: noisy or small samples may
be consistent with both economically meaningful positive and negative values.

`python factor_demo.py` runs the **separate daily network demo**, with simplified
eligibility and no costs; it is not validation of the monthly experiment.
`pairs_trading.py` remains a separate cointegration study. Its historical V/MA
snapshot was inconclusive after multiple-testing adjustment. Existing files in
`data/` are legacy snapshots, not outputs of this revised experiment.

Dependency APIs checked against official [pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pct_change.html),
[statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.fit.html),
[Python SQLite](https://docs.python.org/3.12/library/sqlite3.html), and
[yfinance](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
documentation. Momentum construction follows the existing Jegadeesh–Titman (1993)
inspiration; this different sample does not replicate their study. Educational research.
