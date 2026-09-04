# Market Risk: VaR, Expected Shortfall & Backtesting

Reproducible Python case study for measuring and monitoring the one-day market risk of a diversified illustrative portfolio.

## Business objective

The project reproduces a simplified daily Market Risk workflow:

- consolidate asset returns and portfolio weights;
- calculate historical, parametric and Monte Carlo Value at Risk;
- calculate Expected Shortfall beyond the VaR threshold;
- backtest rolling VaR against realised portfolio P&L;
- monitor exceptions and a simplified risk limit;
- apply historical-style stress scenarios;
- export auditable tables and charts.

All positions and simulated returns are illustrative. No output is investment advice or a regulatory capital calculation.

## Portfolio

The illustrative EUR 10 million portfolio contains four risk factors: Euro equities, US equities, EUR government bonds and EUR/USD. The return generator uses a fixed seed, annual volatilities and a correlation matrix so results are reproducible.

## Risk measures

Losses are reported as positive EUR amounts.

- **Historical VaR:** empirical loss quantile;
- **Parametric VaR:** normal approximation using portfolio mean and volatility;
- **Monte Carlo VaR:** simulated multivariate-normal risk-factor returns;
- **Expected Shortfall:** average loss conditional on exceeding VaR;
- **Backtesting:** comparison of rolling one-day historical VaR with realised losses;
- **Stress tests:** simultaneous shocks to the four portfolio risk factors.

## Repository structure

```text
src/market_risk.py           Risk engine and output generation
tests/test_market_risk.py    Financial-logic tests
outputs/                     Generated CSV reports and charts
requirements.txt             Dependencies
```

## Run

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m src.market_risk
pytest -q
```

## Interpretation for interviews

VaR estimates a loss threshold at a chosen confidence level; it does not describe the average severity beyond that threshold. Expected Shortfall complements VaR by averaging tail losses. Backtesting is essential because a model that systematically underestimates risk produces too many exceptions. The three VaR methods differ because they rely on different distributional and sampling assumptions.

## Skills demonstrated

Market Risk • VaR • Expected Shortfall • Backtesting • Stress Testing • Limits • Correlation • Python • pandas • NumPy • Reporting

## Author

Sofia Kammoune — MBA Trading & Finance de Marché, ESLSCA Business School Paris.

