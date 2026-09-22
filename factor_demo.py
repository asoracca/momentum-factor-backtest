"""Explicitly separate historical daily demo; never used as a monthly fallback."""

import hashlib
import json
from pathlib import Path

from fama_french import demo_strategy_returns, fit_artifacts, load_factors


def main():
    out = Path("outputs/daily_factor_demo")
    out.mkdir(parents=True, exist_ok=True)
    returns = demo_strategy_returns()
    factors = load_factors(start=str(returns.index.min().date()))
    for name, frame, metadata in [
        (
            "returns",
            returns.rename("net_return").to_frame(),
            dict(return_column="net_return", return_kind="zero_investment"),
        ),
        (
            "factors",
            factors,
            dict(
                units="decimal",
                source="Kenneth French daily factors via pandas-datareader",
            ),
        ),
    ]:
        path = out / f"{name}.csv"
        frame.to_csv(path, index_label="date")
        metadata.update(
            frequency="daily", sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
        path.with_suffix(".json").write_text(json.dumps(metadata))
    print("SEPARATE DAILY DEMO: current stocks, gross returns, simplified eligibility")
    print(json.dumps(fit_artifacts(out / "returns.csv", out / "factors.csv"), indent=2))


if __name__ == "__main__":
    main()
