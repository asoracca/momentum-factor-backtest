import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
from test_pipeline import fixture

from experiment import run_experiment
from pipeline import ExperimentConfig, build_portfolio
from research import evaluate_portfolios
from store import attribution, comparison, connect, coverage


def test_sql_provenance_cost_join_and_injection(tmp_path):
    prices, membership = fixture()
    config = ExperimentConfig(
        "2023-01-31",
        min_assets=2,
        simulations=20,
        holdout_status="synthetic_validation",
    )
    output = run_experiment(
        prices, membership, config, tmp_path, {"kind": "synthetic", "label": "test"}
    )
    manifest = json.loads((output / "manifest.json").read_text())
    with connect(tmp_path / "experiments.sqlite") as db:
        run = manifest["run_id"]
        table = comparison(db, run, "2023-01-31", 10)
        assert len(table) == 3
        assert (table.months == 24).all()
        assert comparison(db, "' OR 1=1 --", "2023-01-31", 10).empty
        cover = coverage(db, run).set_index("ticker")
        assert cover.loc["B", "missing_prices"] == 1
        assert cover.loc["D", "missing_prices"] == 37
        joined = attribution(db, run, "long_short", 10)
        assert joined.eligible.eq(1).all()
        for _, month in joined.groupby("return_date"):
            assert month.contribution.sum() == pytest.approx(month.gross.iloc[0])
            assert month.gross.iloc[0] - month.portfolio_cost.iloc[0] == pytest.approx(
                month.portfolio_net.iloc[0]
            )
        # Missing held observations cannot disappear behind SQL AVG.
        full = comparison(db, run, "2021-02-28", 10).set_index("mode")
        assert pd.isna(full.loc["equal_weight", "annual_mean"])
    assert manifest["config"]["seed"] == 17
    assert (output / "timing.png").exists()
    assert len(manifest["prices_sha256"]) == 64


def test_unsupported_schema_is_rejected(tmp_path):
    path = tmp_path / "bad.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=999")
    with pytest.raises(ValueError, match="unsupported"):
        connect(path)


def test_evaluation_never_compresses_gaps_for_bootstrap():
    prices, membership = fixture()
    portfolio = build_portfolio(prices, membership, "equal_weight", min_assets=2)
    result = evaluate_portfolios(
        {"equal_weight": portfolio},
        ExperimentConfig("2021-02-28", min_assets=2, simulations=10),
    )
    assert (result.status == "incomplete_coverage").all()
    assert result.p_value.isna().all()
    assert result.annual_mean.isna().all()


def test_cost_sensitivity_on_same_fixed_dates_and_constant_series():
    prices, membership = fixture()
    portfolio = build_portfolio(prices, membership, min_assets=2)
    config = ExperimentConfig("2023-01-31", min_assets=2, simulations=20)
    table = evaluate_portfolios({"long_short": portfolio}, config)
    first = table[table.block == 1].set_index("bps")
    assert first.loc[25, "annual_mean"] < first.loc[0, "annual_mean"]
    assert (table.months == 24).all()
    assert table.status.eq("exploratory").all()
    portfolio[5].loc[:, "gross_return"] = 0.01
    portfolio[5].loc[:, "turnover"] = 0
    constant = evaluate_portfolios({"long_short": portfolio}, config)
    assert constant.p_value.isna().all()
    assert constant.status.eq("insufficient_sample_or_variation").all()


def test_signal_excludes_most_recent_month_and_execution_uses_next_month():
    prices, membership = fixture()
    original = build_portfolio(prices, membership, min_assets=2)
    changed = prices.copy()
    changed.iloc[12] *= 2
    revised = build_portfolio(changed, membership, min_assets=2)
    pd.testing.assert_series_equal(original[3].iloc[12], revised[3].iloc[12])
    pd.testing.assert_series_equal(original[4].iloc[12], revised[4].iloc[12])
    assert original[5].iloc[0].gross_return == pytest.approx(0.03)
    assert revised[5].iloc[0].gross_return == pytest.approx(0.015)


def test_equal_weight_hand_checked_entry_turnover():
    prices, membership = fixture()
    p = build_portfolio(prices, membership, "equal_weight", min_assets=2)
    np.testing.assert_allclose(p[4].iloc[12], [0.25, 0.25, 0.25, 0.25, 0])
    assert p[5].iloc[2].turnover == pytest.approx(0.4)
