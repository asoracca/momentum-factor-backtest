from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import build_portfolio, validate_prices

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def fixture():
    return pd.read_csv(
        FIXTURES / "prices.csv", index_col=0, parse_dates=True
    ), pd.read_csv(FIXTURES / "membership.csv")


def test_hand_checked_rank_holdings_turnover_and_entry():
    prices, membership = fixture()
    _, _, _, scores, holdings, details, _, _ = build_portfolio(
        prices, membership, min_assets=2
    )
    np.testing.assert_allclose(
        scores.iloc[12][["A", "B", "C", "D"]], np.array([0.99, 1, 1.01, 1.02]) ** 11 - 1
    )
    assert holdings.iloc[12].to_dict() == {"A": -1, "B": 0, "C": 0, "D": 1, "E": 0}
    assert details.iloc[:3].turnover.tolist() == [2, 0, 2]
    assert holdings.iloc[14]["E"] == 1
    assert holdings.iloc[14]["D"] == 0
    for mode, initial in [("long_only", 1), ("equal_weight", 1)]:
        *_, d, _, _ = build_portfolio(prices, membership, mode=mode, min_assets=2)
        assert d.iloc[0].turnover == initial
        assert d.iloc[1].turnover == 0


def test_future_prices_and_membership_cannot_change_past_decisions():
    prices, membership = fixture()
    baseline = build_portfolio(prices, membership, min_assets=2)
    changed = prices.copy()
    changed.iloc[25:] = changed.iloc[25:] * 100
    changed.loc[changed.index[25] :, "A"] = np.nan
    revised_membership = pd.concat(
        [
            membership,
            pd.DataFrame([{"ticker": "E", "start": "2024-01-01", "end": None}]),
        ]
    )
    revised = build_portfolio(changed, revised_membership, min_assets=2)
    for i in [1, 2, 3, 4]:
        pd.testing.assert_frame_equal(baseline[i].iloc[:25], revised[i].iloc[:25])
    pd.testing.assert_frame_equal(baseline[5].iloc[:12], revised[5].iloc[:12])


def test_missing_held_return_is_unknown_and_termination_is_not_filled():
    prices, membership = fixture()
    _, _, eligible, _, holdings, details, missing, returns = build_portfolio(
        prices, membership, mode="equal_weight", min_assets=2
    )
    assert missing.loc["2021-05-31"] == 1
    assert pd.isna(details.loc["2021-05-31", "net_return"])
    assert not eligible.loc["2021-05-31", "B"]
    assert returns.loc["2021-11-30", "D"] == pytest.approx(1.02 * 0.2 - 1)
    assert pd.isna(returns.loc["2021-12-31", "D"])
    assert holdings.loc["2021-12-31", "D"] == 0


def test_missing_calendar_month_is_restored_not_compressed():
    prices, _ = fixture()
    result = validate_prices(prices.drop(prices.index[15]))
    assert result.iloc[15].isna().all()
    assert len(result) == len(prices)
