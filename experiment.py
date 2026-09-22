"""Reproducible offline, CSV, and live monthly experiment commands."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from demo import timing_diagram
from pipeline import ExperimentConfig, build_portfolio, validate_prices
from research import evaluate_portfolios
from store import attribution, comparison, connect, coverage, save_run

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run_experiment(prices, membership, config, output, source):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    run_dir = output / run_id
    run_dir.mkdir()
    prices = validate_prices(prices)
    if (
        len(prices) < 15
        or not prices.index[13]
        < pd.Timestamp(config.evaluation_start)
        <= prices.index[-1]
    ):
        raise ValueError(
            "evaluation_start must leave development and evaluation return months"
        )
    price_bytes = prices.to_csv(index_label="date", float_format="%.17g").encode()
    membership_bytes = (
        membership.to_csv(index=False).encode() if membership is not None else b""
    )
    (run_dir / "prices.csv").write_bytes(price_bytes)
    if membership is not None:
        (run_dir / "membership.csv").write_bytes(membership_bytes)
    code_hashes = {p.name: sha(p.read_bytes()) for p in sorted(ROOT.glob("*.py"))}
    config_dict = asdict(config)
    manifest = dict(
        schema_version=1,
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        command=sys.argv,
        config=config_dict,
        config_sha256=sha(json.dumps(config_dict, sort_keys=True).encode()),
        prices_sha256=sha(price_bytes),
        membership_sha256=sha(membership_bytes),
        code_sha256=code_hashes,
        git_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        git_dirty=bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        source=source,
        date_start=str(prices.index.min().date()),
        date_end=str(prices.index.max().date()),
        tickers=list(prices.columns),
        frequency="monthly",
        versions={
            p: version(p)
            for p in ["numpy", "pandas", "matplotlib", "statsmodels", "yfinance"]
        },
        python=platform.python_version(),
        eligibility_policy="13 observed monthly prices through formation; effective membership at formation",
        missing_policy="no filling; missing held return makes portfolio month unknown; no inference with gaps",
        execution="formation at close t; earn close t to t+1; absolute target-weight turnover, no drift",
        inference="primary L/S 10bps block3; zero-mean centered circular-block one-sided null; other rows exploratory; no specification selection or family-wide significance claim",
        assumptions=[
            "adjusted prices may be retrospectively revised",
            "borrow, financing, impact and drift not modeled",
            "unknown terminal recoveries remain missing",
            "inspected evaluation is not fresh validation",
        ],
    )
    portfolios = {
        mode: build_portfolio(
            prices,
            membership,
            mode,
            config.top_pct,
            config.bottom_pct,
            config.min_assets,
        )
        for mode in ["long_short", "long_only", "equal_weight"]
    }
    summary = evaluate_portfolios(portfolios, config)
    summary.to_csv(run_dir / "evaluation.csv", index=False)
    for mode, p in portfolios.items():
        artifact = p.details.copy()
        artifact.index.name = "date"
        artifact.to_csv(run_dir / f"{mode}_returns.csv")
        metadata = dict(
            run_id=run_id,
            frequency="monthly",
            mode=mode,
            return_column="net_return",
            cost_bps=10,
            return_kind="zero_investment" if mode == "long_short" else "total",
            sha256=sha((run_dir / f"{mode}_returns.csv").read_bytes()),
        )
        (run_dir / f"{mode}_returns.json").write_text(json.dumps(metadata, indent=2))
    manifest["elapsed_seconds_before_storage"] = time.perf_counter() - started
    with connect(output / "experiments.sqlite") as db:
        save_run(db, manifest, portfolios, summary)
        table = comparison(db, run_id, config.evaluation_start, 10)
        table.to_csv(run_dir / "comparison.csv", index=False)
        coverage(db, run_id).to_csv(run_dir / "coverage.csv", index=False)
        attribution(db, run_id, "long_short", 10).to_csv(
            run_dir / "attribution.csv", index=False
        )
    timing_diagram(run_dir / "timing.png")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(source["label"])
    print(
        f"Evaluation status: {config.holdout_status}; start: {config.evaluation_start}"
    )
    print(table.round(6).to_string(index=False))
    print(f"Artifacts: {run_dir}")
    return run_dir


def load_csv(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["offline", "csv", "live"])
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--prices")
    parser.add_argument("--membership")
    parser.add_argument(
        "--source-timestamp",
        help="UTC source timestamp, or unknown when the provider does not supply one",
    )
    parser.add_argument("--adjustment-policy")
    parser.add_argument("--evaluation-start")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument(
        "--end", default="2026-09-01", help="Exclusive, first day of month"
    )
    parser.add_argument(
        "--untouched-declared",
        action="store_true",
        help="Only if this period was never inspected or used to tune",
    )
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).isoformat()
    membership = None
    if args.command == "offline":
        prices = load_csv(ROOT / "fixtures/prices.csv")
        membership = pd.read_csv(ROOT / "fixtures/membership.csv")
        config = ExperimentConfig(
            "2023-01-31", min_assets=2, holdout_status="synthetic_validation"
        )
        source = dict(
            label="SYNTHETIC point-in-time fixture; not market evidence",
            kind="synthetic",
            adjustment_policy="toy compounded price levels",
            source_timestamp=None,
            ingested_at=timestamp,
        )
    else:
        if not args.evaluation_start:
            parser.error(
                "--evaluation-start is required and must be frozen before examining results"
            )
        config = ExperimentConfig(
            args.evaluation_start,
            holdout_status="untouched_declared"
            if args.untouched_declared
            else "previously_inspected",
        )
        if args.membership:
            membership = pd.read_csv(args.membership)
        if args.command == "csv":
            if (
                not args.prices
                or not args.adjustment_policy
                or not args.source_timestamp
            ):
                parser.error(
                    "csv requires --prices, --adjustment-policy, --source-timestamp"
                )
            prices = load_csv(args.prices)
            source = dict(
                kind="historical_file",
                source_timestamp=None
                if args.source_timestamp == "unknown"
                else args.source_timestamp,
                ingested_at=timestamp,
                adjustment_policy=args.adjustment_policy,
                input_sha256=sha(Path(args.prices).read_bytes()),
            )
        else:
            import yfinance as yf

            from momentum_backtest import UNIVERSE

            start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
            current_month = pd.Timestamp.now().to_period("M").start_time
            if start.day != 1 or end.day != 1 or end > current_month or end <= start:
                parser.error(
                    "live dates must bound complete months: first-of-month start/end, end <= current month"
                )
            tickers = (
                sorted(membership.ticker.unique())
                if membership is not None
                else UNIVERSE
            )
            raw = yf.download(
                tickers,
                start=args.start,
                end=args.end,
                interval="1mo",
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                raise ValueError(
                    "live provider returned no data; no fallback was substituted"
                )
            prices = raw["Close"].reindex(columns=tickers)
            prices.index = pd.to_datetime(prices.index).tz_localize(None)
            prices = prices.loc[(prices.index >= start) & (prices.index < end)]
            prices.index = prices.index.to_period("M").to_timestamp("M")
            prices = prices.reindex(
                pd.period_range(
                    start, end - pd.Timedelta(days=1), freq="M"
                ).to_timestamp("M")
            )
            source = dict(
                kind="live_download",
                source_timestamp=None,
                retrieval_started_at=timestamp,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                adjustment_policy="Yahoo auto_adjust=True monthly Close; provider revision timestamp unavailable",
                requested_start=args.start,
                requested_end_exclusive=args.end,
            )
        source["label"] = (
            "User-supplied historical membership; provenance must be independently verified"
            if membership is not None
            else "SURVIVORSHIP-BIASED current-universe study"
        )
        if membership is not None:
            source["membership_input_sha256"] = sha(Path(args.membership).read_bytes())
            source["membership_source_timestamp"] = args.source_timestamp
    run_experiment(prices, membership, config, args.output, source)


if __name__ == "__main__":
    main()
