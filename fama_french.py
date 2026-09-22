"""
fama_french.py
--------------
Fama-French (+Momentum) factor regression. Splits a strategy's returns into
exposure to known risk factors and tests for leftover ALPHA.

    excess_ret_t = alpha + b_mkt*(Mkt-RF) + b_smb*SMB + b_hml*HML + b_mom*MOM + e

  Inference is conditional on this sample and model; nonrejection is not equality.

    python fama_french.py
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib

matplotlib.use("Agg")


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

    ff3 = FamaFrenchReader(
        "F-F_Research_Data_Factors_daily", start=start, end=end
    ).read()[0]
    mom = FamaFrenchReader("F-F_Momentum_Factor_daily", start=start, end=end).read()[0]
    ff3.columns = [c.strip() for c in ff3.columns]
    mom.columns = [c.strip() for c in mom.columns]
    mom = mom.rename(columns={"Mom": "MOM"})
    ff = ff3.join(mom, how="inner") / 100.0
    ff.index = to_dates(ff.index)
    return ff


def demo_strategy_returns(start="2015-01-01"):
    import yfinance as yf

    tickers = [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "JPM",
        "XOM",
        "JNJ",
        "PG",
        "KO",
        "PEP",
        "WMT",
        "HD",
        "BAC",
        "DIS",
        "CSCO",
        "INTC",
        "T",
        "VZ",
    ]
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)[
        "Close"
    ].dropna(how="all")
    px.index = to_dates(px.index)
    rets = px.pct_change(fill_method=None)
    mom = px.shift(21) / px.shift(252) - 1.0
    out = {}
    for d in rets.index:
        sig = mom.loc[d].dropna()
        if len(sig) < 6:
            out[d] = 0.0
            continue
        longs = sig.nlargest(5).index
        shorts = sig.nsmallest(5).index
        r = rets.loc[d, longs].mean() - rets.loc[d, shorts].mean()
        out[d] = float(r) if pd.notna(r) else 0.0
    s = pd.Series(out)
    s.index = to_dates(s.index)
    return s.dropna()


def read_artifact(path, expected_frequency=None):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError(f"artifact hash mismatch: {path}")
    frequency = metadata.get("frequency")
    if frequency not in {"monthly", "daily"} or (
        expected_frequency and frequency != expected_frequency
    ):
        raise ValueError("artifact frequency mismatch")
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("artifact dates must be unique and ordered")
    if frequency == "monthly":
        if frame.index.to_period("M").has_duplicates:
            raise ValueError("multiple observations in a monthly artifact")
        frame.index = frame.index.to_period("M").to_timestamp("M")
    return frame, metadata


def fit_artifacts(return_path, factor_path, start=None):
    strategy, metadata = read_artifact(return_path)
    factors, factor_meta = read_artifact(factor_path, metadata["frequency"])
    if factor_meta.get("units") != "decimal":
        raise ValueError("factor units must explicitly be decimal, not percent")
    if not factor_meta.get("source"):
        raise ValueError("factor source metadata is required")
    strategy = strategy.loc[start:] if start else strategy
    column = metadata["return_column"]
    required = ["Mkt-RF", "SMB", "HML", "MOM", "RF"]
    if not set(required).issubset(factors.columns):
        raise ValueError("missing required factor columns")
    if not strategy.index.isin(factors.index).all():
        raise ValueError(
            "factor date coverage does not match the selected return artifact"
        )
    factors = factors.loc[strategy.index, required]
    if (
        len(strategy) < 12
        or not np.isfinite(strategy[column]).all()
        or not np.isfinite(factors).all().all()
    ):
        raise ValueError(
            "need at least 12 finite aligned returns; missing values are not silently dropped"
        )
    if metadata["frequency"] == "monthly" and len(
        pd.period_range(strategy.index.min(), strategy.index.max(), freq="M")
    ) != len(strategy):
        raise ValueError("monthly return artifact has calendar gaps")
    kind = metadata.get("return_kind")
    if kind not in {"zero_investment", "total", "excess"}:
        raise ValueError("return_kind must specify total, excess, or zero_investment")
    y = strategy[column] - factors["RF"] if kind == "total" else strategy[column]
    X = sm.add_constant(factors[required[:-1]], has_constant="add")
    if np.linalg.matrix_rank(X.to_numpy()) != X.shape[1]:
        raise ValueError("factor design is rank deficient")
    lags = 3 if metadata["frequency"] == "monthly" else 5
    model = sm.OLS(y, X, missing="raise").fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    interval = model.conf_int().loc["const"].tolist()
    result = dict(
        return_artifact=str(return_path),
        return_sha256=metadata["sha256"],
        factor_artifact=str(factor_path),
        factor_sha256=factor_meta["sha256"],
        factor_source=factor_meta["source"],
        frequency=metadata["frequency"],
        return_kind=kind,
        n=int(model.nobs),
        start=str(strategy.index.min().date()),
        end=str(strategy.index.max().date()),
        alpha=float(model.params["const"]),
        alpha_ci95=interval,
        p_value=float(model.pvalues["const"]),
        hac_lags=lags,
        betas=model.params.drop("const").to_dict(),
        interpretation="Conditional estimate; nonsignificance does not establish zero alpha or absence of skill; repeated model searches require multiple-testing control.",
    )
    return result


def run():
    parser = argparse.ArgumentParser(
        description="Explicit return-artifact factor regression; no fallback"
    )
    parser.add_argument("--returns", required=True)
    parser.add_argument(
        "--factors",
        required=True,
        help="CSV with date,Mkt-RF,SMB,HML,MOM,RF and JSON metadata",
    )
    parser.add_argument("--start", help="Optional fixed evaluation start")
    parser.add_argument("--output", default="outputs/factor_regression.json")
    args = parser.parse_args()
    result = fit_artifacts(args.returns, args.factors, args.start)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
