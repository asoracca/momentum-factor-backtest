import numpy as np
import pandas as pd
import pytest

from momentum_backtest import compute_momentum_signal, run_backtest
from research import block_bootstrap_significance, chronological_split, cost_sensitivity


def synthetic_prices(months=36, tickers=12):
    index = pd.date_range("2020-01-31", periods=months, freq="ME")
    data = {}
    for ticker in range(tickers):
        monthly_return = -0.01 + ticker * 0.003
        data[f"S{ticker:02d}"] = 100 * (1 + monthly_return) ** np.arange(months)
    return pd.DataFrame(data, index=index)


def test_signal_uses_only_information_available_at_the_signal_date():
    prices = synthetic_prices()
    baseline = compute_momentum_signal(prices).iloc[20]
    changed = prices.copy()
    changed.iloc[21:] *= 100
    revised = compute_momentum_signal(changed).iloc[20]
    pd.testing.assert_series_equal(baseline, revised)


def test_backtest_reports_turnover_costs_and_net_returns():
    details = run_backtest(synthetic_prices(), return_details=True)
    assert list(details.columns) == [
        "gross_return",
        "turnover",
        "cost",
        "net_return",
    ]
    assert (details["turnover"] >= 0).all()
    assert (details["cost"] >= 0).all()
    np.testing.assert_allclose(
        details["net_return"], details["gross_return"] - details["cost"]
    )


def test_higher_costs_cannot_improve_the_same_strategy_returns():
    prices = synthetic_prices()
    free = run_backtest(prices, transaction_cost_bps=0)
    costly = run_backtest(prices, transaction_cost_bps=25)
    assert (costly <= free).all()
    assert costly.mean() < free.mean()


@pytest.mark.parametrize("mode", ["long_short", "long_only", "equal_weight"])
def test_all_supported_modes_produce_finite_returns(mode):
    returns = run_backtest(synthetic_prices(), mode=mode)
    assert len(returns) > 10
    assert np.isfinite(returns).all()


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="mode must be"):
        run_backtest(synthetic_prices(), mode="magic")


def test_chronological_split_has_no_overlap_and_preserves_order():
    details = run_backtest(synthetic_prices(), return_details=True)
    development, evaluation = chronological_split(details, 0.6)
    assert development.index.max() < evaluation.index.min()
    assert len(development) + len(evaluation) == len(details)


def test_block_bootstrap_is_deterministic_and_tests_positive_mean():
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.normal(0.02, 0.01, size=60))
    first = block_bootstrap_significance(returns, n_sim=1_000, seed=17)
    second = block_bootstrap_significance(returns, n_sim=1_000, seed=17)
    assert first == second
    assert first["actual_sharpe"] > 0
    assert first["p_value"] < 0.05


def test_cost_sensitivity_uses_the_requested_cost_grid():
    result = cost_sensitivity(
        synthetic_prices(), run_backtest, costs=(0, 10, 25), mode="long_only"
    )
    assert result.index.tolist() == [0, 10, 25]
    assert result.loc[25, "annual_return"] < result.loc[0, "annual_return"]
