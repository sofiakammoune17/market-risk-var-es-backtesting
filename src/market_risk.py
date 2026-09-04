"""Illustrative market-risk engine with VaR, ES, stress tests and backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ASSETS = ["Euro Equities", "US Equities", "EUR Bonds", "EURUSD"]


@dataclass(frozen=True)
class Portfolio:
    value_eur: float
    weights: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=float)
        if self.value_eur <= 0:
            raise ValueError("Portfolio value must be positive.")
        if len(weights) != len(ASSETS):
            raise ValueError(f"Exactly {len(ASSETS)} asset weights are required.")
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError("Portfolio weights must sum to 1.")
        object.__setattr__(self, "weights", weights)


def generate_returns(observations: int = 750, seed: int = 42) -> pd.DataFrame:
    """Generate reproducible correlated daily returns for an illustrative portfolio."""
    if observations < 2:
        raise ValueError("At least two observations are required.")
    annual_vol = np.array([0.20, 0.22, 0.06, 0.10])
    correlation = np.array(
        [
            [1.00, 0.72, -0.25, -0.15],
            [0.72, 1.00, -0.20, -0.10],
            [-0.25, -0.20, 1.00, 0.05],
            [-0.15, -0.10, 0.05, 1.00],
        ]
    )
    daily_cov = np.outer(annual_vol, annual_vol) * correlation / 252
    rng = np.random.default_rng(seed)
    data = rng.multivariate_normal(np.zeros(len(ASSETS)), daily_cov, observations)
    index = pd.bdate_range(end="2026-08-31", periods=observations)
    return pd.DataFrame(data, index=index, columns=ASSETS)


def portfolio_returns(returns: pd.DataFrame, portfolio: Portfolio) -> pd.Series:
    missing = set(ASSETS).difference(returns.columns)
    if missing:
        raise ValueError(f"Missing return columns: {sorted(missing)}")
    return returns[ASSETS].dot(portfolio.weights).rename("portfolio_return")


def historical_var_es(pnl: pd.Series, confidence: float = 0.99) -> tuple[float, float]:
    _validate_confidence(confidence)
    losses = -pd.Series(pnl, dtype=float).dropna()
    var = float(losses.quantile(confidence))
    tail = losses[losses >= var]
    return var, float(tail.mean())


def parametric_var(pnl: pd.Series, confidence: float = 0.99) -> float:
    _validate_confidence(confidence)
    pnl = pd.Series(pnl, dtype=float).dropna()
    z_score = NormalDist().inv_cdf(confidence)
    return float(-pnl.mean() + z_score * pnl.std(ddof=1))


def monte_carlo_var_es(
    returns: pd.DataFrame,
    portfolio: Portfolio,
    confidence: float = 0.99,
    simulations: int = 100_000,
    seed: int = 7,
) -> tuple[float, float]:
    _validate_confidence(confidence)
    if simulations < 1_000:
        raise ValueError("Use at least 1,000 Monte Carlo simulations.")
    rng = np.random.default_rng(seed)
    simulated = rng.multivariate_normal(
        returns[ASSETS].mean().to_numpy(),
        returns[ASSETS].cov().to_numpy(),
        simulations,
    )
    pnl = simulated.dot(portfolio.weights) * portfolio.value_eur
    return historical_var_es(pd.Series(pnl), confidence)


def rolling_backtest(
    pnl: pd.Series, window: int = 250, confidence: float = 0.99
) -> pd.DataFrame:
    _validate_confidence(confidence)
    pnl = pd.Series(pnl, dtype=float).dropna()
    if len(pnl) <= window:
        raise ValueError("P&L history must be longer than the rolling window.")
    var = -pnl.shift(1).rolling(window).quantile(1 - confidence)
    result = pd.DataFrame({"realised_pnl_eur": pnl, "historical_var_eur": var}).dropna()
    result["exception"] = result["realised_pnl_eur"] < -result["historical_var_eur"]
    return result


def stress_tests(portfolio: Portfolio) -> pd.DataFrame:
    scenarios = {
        "Equity sell-off": np.array([-0.15, -0.18, 0.025, -0.04]),
        "Rates shock": np.array([-0.03, -0.02, -0.06, 0.01]),
        "EUR appreciation": np.array([0.01, 0.00, 0.00, 0.10]),
        "Combined crisis": np.array([-0.20, -0.22, -0.05, -0.08]),
    }
    rows = []
    for name, shocks in scenarios.items():
        pnl = float(shocks.dot(portfolio.weights) * portfolio.value_eur)
        rows.append({"scenario": name, "pnl_eur": pnl, "impact_pct": pnl / portfolio.value_eur})
    return pd.DataFrame(rows)


def risk_summary(returns: pd.DataFrame, portfolio: Portfolio) -> pd.DataFrame:
    p_returns = portfolio_returns(returns, portfolio)
    pnl = p_returns * portfolio.value_eur
    rows = []
    for confidence in (0.95, 0.99):
        hist_var, hist_es = historical_var_es(pnl, confidence)
        mc_var, mc_es = monte_carlo_var_es(returns, portfolio, confidence)
        rows.extend(
            [
                {"confidence": confidence, "method": "Historical", "var_eur": hist_var, "es_eur": hist_es},
                {"confidence": confidence, "method": "Parametric", "var_eur": parametric_var(pnl, confidence), "es_eur": np.nan},
                {"confidence": confidence, "method": "Monte Carlo", "var_eur": mc_var, "es_eur": mc_es},
            ]
        )
    return pd.DataFrame(rows)


def save_outputs(output_dir: Path = Path("outputs")) -> None:
    portfolio = Portfolio(10_000_000, np.array([0.35, 0.25, 0.30, 0.10]))
    returns = generate_returns()
    pnl = portfolio_returns(returns, portfolio) * portfolio.value_eur
    summary = risk_summary(returns, portfolio)
    backtest = rolling_backtest(pnl)
    stresses = stress_tests(portfolio)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "risk_summary.csv", index=False)
    backtest.to_csv(output_dir / "var_backtest.csv")
    stresses.to_csv(output_dir / "stress_tests.csv", index=False)

    chart = backtest.tail(250)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(chart.index, chart["realised_pnl_eur"], label="Realised P&L", linewidth=1)
    ax.plot(chart.index, -chart["historical_var_eur"], label="99% VaR threshold", linewidth=1.5)
    exceptions = chart[chart["exception"]]
    ax.scatter(exceptions.index, exceptions["realised_pnl_eur"], color="red", label="Exceptions", zorder=3)
    ax.set_title("Rolling one-day 99% historical VaR backtest")
    ax.set_ylabel("EUR")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "var_backtest.png", dpi=180)
    plt.close(fig)

    print(summary.round(2).to_string(index=False))
    print(f"\nBacktesting exceptions: {int(backtest['exception'].sum())} / {len(backtest)}")
    print("\nStress tests")
    print(stresses.round(4).to_string(index=False))


def _validate_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")


if __name__ == "__main__":
    save_outputs()

