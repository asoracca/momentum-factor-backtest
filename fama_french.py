"""
fama_french.py
--------------
Fama-French (+Momentum) factor regression. Splits a strategy's returns into
exposure to known risk factors and tests for leftover ALPHA.

    excess_ret_t = alpha + b_mkt*(Mkt-RF) + b_smb*SMB + b_hml*HML + b_mom*MOM + e

  Significant POSITIVE alpha -> real edge. Alpha ~ 0 with big betas -> just factors.

    python fama_french.py
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def to_dates(idx):
    """Coerce any index to tz-naive calendar dates so joins line up."""
    if isinstance(idx, pd.PeriodIndex):
        idx = idx.to_timestamp()
    idx = pd.DatetimeIndex(pd.to_datetime(idx))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def load_factors(start="2015-01-01", end=None):
    from pandas_datareader.famafrench import FamaFrenchReader
    ff3 = FamaFrenchReader("F-F_Research_Data_Factors_daily", start=start, end=end).read()[0]
    mom = FamaFrenchReader("F-F_Momentum_Factor_daily", start=start, end=end).read()[0]
    ff3.columns = [c.strip() for c in ff3.columns]
    mom.columns = [c.strip() for c in mom.columns]
    mom = mom.rename(columns={"Mom": "MOM"})
    ff = ff3.join(mom, how="inner") / 100.0
    ff.index = to_dates(ff.index)
    return ff


def demo_strategy_returns(start="2015-01-01"):
    import yfinance as yf
    tickers = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","JPM","XOM","JNJ","PG",
               "KO","PEP","WMT","HD","BAC","DIS","CSCO","INTC","T","VZ"]
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"].dropna(how="all")
    px.index = to_dates(px.index)
    rets = px.pct_change()
    mom = px.shift(21) / px.shift(252) - 1.0
    out = {}
    for d in rets.index:
        sig = mom.loc[d].dropna()
        if len(sig) < 6:
            out[d] = 0.0; continue
        longs = sig.nlargest(5).index
        shorts = sig.nsmallest(5).index
        r = rets.loc[d, longs].mean() - rets.loc[d, shorts].mean()
        out[d] = float(r) if pd.notna(r) else 0.0
    s = pd.Series(out)
    s.index = to_dates(s.index)
    return s.dropna()


def load_strategy_returns():
    path = "data/strategy_returns.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        c = df.columns
        s = df.set_index(c[0])[c[1]].astype(float)
        s.index = to_dates(s.index)
        print(f"Loaded strategy returns from {path} ({len(s)} days).")
        return s
    print("No data/strategy_returns.csv — building a demo momentum long/short portfolio.")
    return demo_strategy_returns()


def run():
    strat = load_strategy_returns()
    ff = load_factors(start=str(strat.index.min().date()))
    df = pd.concat([strat.rename("ret"), ff], axis=1, join="inner").dropna()
    print(f"  strategy days: {len(strat)} | factor days: {len(ff)} | overlapping: {len(df)}")
    if len(df) < 60:
        print("Still not enough overlap — paste this line and I'll fix it."); return

    y = df["ret"] - df["RF"]
    X = sm.add_constant(df[["Mkt-RF", "SMB", "HML", "MOM"]])
    model = sm.OLS(y, X).fit()

    a_d = model.params["const"]
    a_ann = (1 + a_d) ** 252 - 1
    print("=" * 60)
    print("  FAMA-FRENCH (+MOM) FACTOR REGRESSION")
    print("=" * 60)
    print(f"  Observations: {int(model.nobs)} days")
    print(f"  R-squared:    {model.rsquared:.3f}")
    print("-" * 60)
    print(f"  Alpha (daily):      {a_d*100:+.4f}%   t = {model.tvalues['const']:+.2f}   p = {model.pvalues['const']:.3f}")
    print(f"  Alpha (annualized): {a_ann*100:+.2f}%")
    print("-" * 60)
    print("  Factor exposures (beta):")
    for f in ["Mkt-RF", "SMB", "HML", "MOM"]:
        print(f"    {f:8s} beta = {model.params[f]:+.3f}   t = {model.tvalues[f]:+.2f}   p = {model.pvalues[f]:.3f}")
    print("=" * 60)
    if model.pvalues["const"] < 0.05 and a_d > 0:
        print("  Significant POSITIVE alpha — real edge beyond the factors.")
    elif model.pvalues["const"] < 0.05 and a_d < 0:
        print("  Significant NEGATIVE alpha — underperforms its factor exposure.")
    else:
        print("  Alpha not distinguishable from zero — returns are factor exposure, not skill.")
    dom = max(["Mkt-RF", "SMB", "HML", "MOM"], key=lambda f: abs(model.params[f]))
    print(f"  Biggest tilt: {dom} (beta {model.params[dom]:+.2f}).")
    print("=" * 60)

    pred = model.predict(X)
    plt.figure(figsize=(10, 5))
    plt.plot((1 + y).cumprod().index, (1 + y).cumprod(), label="Strategy (excess)", lw=2)
    plt.plot((1 + pred).cumprod().index, (1 + pred).cumprod(), label="Factor-predicted", lw=1.5, ls="--")
    plt.title("Strategy vs Fama-French Factor Model"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/fama_french_fit.png", dpi=110)
    print("  Saved chart: data/fama_french_fit.png")


if __name__ == "__main__":
    run()
