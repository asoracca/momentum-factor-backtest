"""
factor_analysis.py  --  Fama-French (+ momentum) factor regression
====================================================================
Drop this into  momentum-factor-backtest/  (e.g. as src/factor_analysis.py).

WHAT IT ANSWERS
---------------
"Do I have real alpha, or am I just being paid for factor exposure?"

It regresses YOUR strategy's excess returns on the Fama-French factors:

    R_strategy - Rf  =  alpha
                        + b_mkt * (Mkt-Rf)     # market beta
                        + b_smb * SMB          # size  (small minus big)
                        + b_hml * HML          # value (high minus low book/mkt)
                        + b_mom * MOM          # momentum (Carhart 4th factor)
                        + e

The intercept `alpha` is the return your strategy earns that the factors
*cannot* explain. If alpha is positive AND statistically significant
(|t| > ~2, after Newey-West correction for autocorrelation), you have
something real. If alpha dies once you add factors, your "edge" was just
factor beta you could have bought cheaply with an index/ETF.

DATA
----
Fama-French factors are pulled automatically from Ken French's data library
via pandas-datareader (no API key). MOM is the separate momentum factor file.

    pip install pandas numpy statsmodels pandas-datareader

USAGE
-----
    from factor_analysis import run_factor_regression, load_ff_factors

    # returns_series: a pandas Series of PERIODIC strategy returns (decimal,
    # not %), indexed by date. Daily or monthly both work.
    result = run_factor_regression(returns_series, freq="daily")
    result.summary_report()        # pretty console verdict
    result.to_frame()              # tidy DataFrame of coefs/t-stats

Run as a script for a self-test on synthetic data:
    python factor_analysis.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

# -----------------------------------------------------------------------------
# 1.  Load factors from Ken French's library
# -----------------------------------------------------------------------------
_FF_DAILY = "F-F_Research_Data_Factors_daily"
_FF_MONTHLY = "F-F_Research_Data_Factors"
_MOM_DAILY = "F-F_Momentum_Factor_daily"
_MOM_MONTHLY = "F-F_Momentum_Factor"


def load_ff_factors(freq: str = "daily", start="2000-01-01", end=None) -> pd.DataFrame:
    """Return a DataFrame with columns Mkt-RF, SMB, HML, MOM, RF (all decimal).

    Falls back gracefully if pandas-datareader / network is unavailable: it
    raises a clear error telling you to drop a local CSV in instead.
    """
    try:
        from pandas_datareader.famafrench import get_available_datasets  # noqa
        import pandas_datareader.data as web
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pandas-datareader is required to auto-download FF factors.\n"
            "  pip install pandas-datareader\n"
            "Or pass your own factor DataFrame to run_factor_regression(...)."
        ) from e

    ff_name = _FF_DAILY if freq == "daily" else _FF_MONTHLY
    mom_name = _MOM_DAILY if freq == "daily" else _MOM_MONTHLY

    ff = web.DataReader(ff_name, "famafrench", start=start, end=end)[0]
    mom = web.DataReader(mom_name, "famafrench", start=start, end=end)[0]

    # French publishes these in PERCENT -> convert to decimal
    ff = ff / 100.0
    mom = mom / 100.0
    mom.columns = ["MOM"]  # the file labels it 'Mom   ' with stray spaces

    factors = ff.join(mom, how="inner")
    factors.index = pd.to_datetime(factors.index.to_timestamp()
                                   if hasattr(factors.index, "to_timestamp")
                                   else factors.index)
    return factors[["Mkt-RF", "SMB", "HML", "MOM", "RF"]]


# -----------------------------------------------------------------------------
# 2.  The regression
# -----------------------------------------------------------------------------
@dataclass
class FactorResult:
    model: sm.regression.linear_model.RegressionResultsWrapper
    freq: str
    n_obs: int
    ann_factor: float

    @property
    def alpha_periodic(self) -> float:
        return self.model.params["const"]

    @property
    def alpha_annualized(self) -> float:
        # geometric annualization of a per-period intercept
        return (1 + self.alpha_periodic) ** self.ann_factor - 1

    @property
    def alpha_tstat(self) -> float:
        return self.model.tvalues["const"]

    @property
    def alpha_pvalue(self) -> float:
        return self.model.pvalues["const"]

    def betas(self) -> pd.Series:
        return self.model.params.drop("const")

    def to_frame(self) -> pd.DataFrame:
        out = pd.DataFrame({
            "coef": self.model.params,
            "t_stat": self.model.tvalues,
            "p_value": self.model.pvalues,
        })
        out.index = out.index.str.replace("const", "alpha (per period)")
        return out

    def verdict(self) -> str:
        sig = abs(self.alpha_tstat) >= 2.0
        pos = self.alpha_periodic > 0
        if pos and sig:
            return ("REAL ALPHA: positive intercept that the factors cannot "
                    "explain (|t| >= 2). Your edge is not just factor exposure.")
        if pos and not sig:
            return ("INCONCLUSIVE: alpha is positive but not statistically "
                    "significant (|t| < 2). Could be skill or luck — need more "
                    "data / breadth before trusting it.")
        return ("NO ALPHA: once you control for market/size/value/momentum, "
                "the strategy adds nothing (or negative). Returns were factor "
                "beta you could buy cheaply.")

    def summary_report(self) -> None:
        print("=" * 64)
        print(f" FAMA-FRENCH + MOMENTUM FACTOR REGRESSION  ({self.freq})")
        print("=" * 64)
        print(f" Observations           : {self.n_obs}")
        print(f" Alpha (per period)     : {self.alpha_periodic:+.5%}")
        print(f" Alpha (annualized)     : {self.alpha_annualized:+.2%}")
        print(f" Alpha t-stat (NW)      : {self.alpha_tstat:+.2f}"
              f"   p={self.alpha_pvalue:.3f}")
        print(f" R-squared              : {self.model.rsquared:.3f}")
        print("-" * 64)
        print(" Factor loadings (betas):")
        for name, b in self.betas().items():
            t = self.model.tvalues[name]
            print(f"   {name:<8} beta={b:+.3f}   t={t:+.2f}")
        print("-" * 64)
        print(" VERDICT:")
        for line in _wrap(self.verdict(), 60):
            print("   " + line)
        print("=" * 64)


def run_factor_regression(
    strategy_returns: pd.Series,
    factors: pd.DataFrame | None = None,
    freq: str = "daily",
    use_momentum: bool = True,
    newey_west_lags: int | None = None,
) -> FactorResult:
    """Regress strategy returns on FF(+MOM) factors.

    Parameters
    ----------
    strategy_returns : pd.Series
        Periodic strategy returns in DECIMAL form (0.01 == 1%), date-indexed.
        These should be TOTAL returns; Rf is subtracted internally.
    factors : optional pre-loaded factor DataFrame (else auto-downloaded).
    freq : "daily" or "monthly".
    use_momentum : include Carhart MOM factor (recommended for a momentum book).
    newey_west_lags : HAC lag length. Default: 5 (daily) / 3 (monthly).
    """
    if not isinstance(strategy_returns.index, pd.DatetimeIndex):
        strategy_returns = strategy_returns.copy()
        strategy_returns.index = pd.to_datetime(strategy_returns.index)

    if factors is None:
        factors = load_ff_factors(freq=freq,
                                  start=str(strategy_returns.index.min().date()))

    df = pd.concat([strategy_returns.rename("R"), factors], axis=1).dropna()
    if len(df) < 30:
        warnings.warn(f"Only {len(df)} overlapping observations — results fragile.")

    cols = ["Mkt-RF", "SMB", "HML"] + (["MOM"] if use_momentum else [])
    y = df["R"] - df["RF"]                    # excess strategy return
    X = sm.add_constant(df[cols])

    if newey_west_lags is None:
        newey_west_lags = 5 if freq == "daily" else 3

    model = sm.OLS(y, X).fit(cov_type="HAC",
                             cov_kwds={"maxlags": newey_west_lags})

    ann = 252.0 if freq == "daily" else 12.0
    return FactorResult(model=model, freq=freq, n_obs=len(df), ann_factor=ann)


def _wrap(text: str, width: int):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out


# -----------------------------------------------------------------------------
# 3.  Self-test on synthetic data
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n = 750
    idx = pd.bdate_range("2022-01-03", periods=n)

    # fake factors
    mkt = rng.normal(0.0003, 0.011, n)
    smb = rng.normal(0.0000, 0.005, n)
    hml = rng.normal(0.0000, 0.006, n)
    mom = rng.normal(0.0002, 0.007, n)
    rf = np.full(n, 0.00008)
    fake_factors = pd.DataFrame(
        {"Mkt-RF": mkt, "SMB": smb, "HML": hml, "MOM": mom, "RF": rf}, index=idx)

    # strategy = momentum-tilted + small TRUE alpha of ~5bp/day
    true_alpha = 0.0005
    strat = (rf + true_alpha + 1.1 * mkt + 0.3 * mom + 0.1 * smb
             + rng.normal(0, 0.004, n))
    strat = pd.Series(strat, index=idx, name="strategy")

    res = run_factor_regression(strat, factors=fake_factors, freq="daily")
    res.summary_report()
    print("\nTidy table:\n", res.to_frame().round(4))