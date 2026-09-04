import numpy as np
import pandas as pd
import pytest

from src.market_risk import (
    Portfolio,
    generate_returns,
    historical_var_es,
    parametric_var,
    portfolio_returns,
    rolling_backtest,
    stress_tests,
)


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        Portfolio(1_000_000, np.array([0.5, 0.5, 0.2, 0.0]))


def test_generated_returns_are_reproducible():
    pd.testing.assert_frame_equal(generate_returns(10, 1), generate_returns(10, 1))


def test_portfolio_returns_are_weighted_returns():
    portfolio = Portfolio(1_000_000, np.array([1.0, 0.0, 0.0, 0.0]))
    returns = generate_returns(10)
    pd.testing.assert_series_equal(
        portfolio_returns(returns, portfolio),
        returns["Euro Equities"].rename("portfolio_return"),
    )


def test_expected_shortfall_is_not_below_var():
    pnl = pd.Series([-100, -70, -20, 0, 10, 30, 40, 50])
    var, es = historical_var_es(pnl, 0.75)
    assert es >= var


def test_parametric_var_increases_with_confidence():
    pnl = pd.Series([-100, -50, 0, 25, 50, 75])
    assert parametric_var(pnl, 0.99) > parametric_var(pnl, 0.95)


def test_backtest_flags_losses_below_var_threshold():
    pnl = pd.Series([0.0] * 20 + [-100.0] + [0.0] * 5)
    result = rolling_backtest(pnl, window=20, confidence=0.95)
    assert bool(result.loc[20, "exception"])


def test_combined_crisis_is_a_loss():
    portfolio = Portfolio(10_000_000, np.array([0.35, 0.25, 0.30, 0.10]))
    result = stress_tests(portfolio).set_index("scenario")
    assert result.loc["Combined crisis", "pnl_eur"] < 0

