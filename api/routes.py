from fastapi import APIRouter
from api.schemas import PortfolioCreate
from Backend.data_layer import fetch_stock_data, calculate_returns, get_mean_returns_and_covariance, fetch_market_data
from Backend.risk_metrics import (
    calculate_annualized_performance,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_historical_var,
    calculate_parametric_var,
    calculate_cvar,
    calculate_max_drawdown,
    calculate_portfolio_beta,
    calculate_risk_contribution
)
from Backend.monte_carlo import run_monte_carlo_simulation, simulated_var_cvar
from Backend.optimization import optimize_portfolio
from api.ai_summary import generate_portfolio_summary
import numpy as np

router = APIRouter()

@router.post("/portfolio")
def create_portfolio(data: PortfolioCreate):

    prices = fetch_stock_data(data.tickers, data.start_date)
    returns = calculate_returns(prices)

    mean_returns, cov_matrix = get_mean_returns_and_covariance(returns)

    weights = np.array(data.weights)

    portfolio_daily = returns.dot(weights)

    ann_ret, ann_vol = calculate_annualized_performance(weights, mean_returns, cov_matrix)
    sharpe = calculate_sharpe_ratio(ann_ret, ann_vol)

    portfolio_mean_daily = np.dot(weights, mean_returns)
    portfolio_volatility_daily = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    sortino = calculate_sortino_ratio(portfolio_daily, ann_ret)
    hist_var = calculate_historical_var(portfolio_daily)
    param_var = calculate_parametric_var(portfolio_mean_daily, portfolio_volatility_daily)
    cvar = calculate_cvar(portfolio_daily, hist_var)
    max_dd = calculate_max_drawdown(portfolio_daily)

    market_prices = fetch_market_data(data.start_date)
    market_returns = market_prices.pct_change().dropna()

    beta = calculate_portfolio_beta(portfolio_daily, market_returns)

    risk_contributions = calculate_risk_contribution(weights, cov_matrix)

    simulations = run_monte_carlo_simulation(mean_returns, cov_matrix, weights)
    sim_var, sim_cvar = simulated_var_cvar(simulations)

    opt_sharpe = optimize_portfolio(mean_returns, cov_matrix, objective="sharpe")
    opt_vol = optimize_portfolio(mean_returns, cov_matrix, objective="volatility")

    risk_metrics = {
        "annual_return": float(ann_ret),
        "annual_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "historical_var_95": float(hist_var),
        "parametric_var_95": float(param_var),
        "cvar": float(cvar),
        "max_drawdown": float(max_dd),
        "portfolio_beta": float(beta),
        "risk_contributions": risk_contributions.tolist()
    }

    summary = generate_portfolio_summary({
        "risk_metrics": risk_metrics,
        "monte_carlo": {
            "simulated_var": float(sim_var),
            "simulated_cvar": float(sim_cvar)
        },
        "optimization": {
            "optimal_sharpe_weights": opt_sharpe["weights"].tolist(),
            "optimal_volatility_weights": opt_vol["weights"].tolist()
        }
    })

    return {
        "risk_metrics": risk_metrics,
        "monte_carlo": {
            "simulated_var_95": float(sim_var),
            "simulated_cvar": float(sim_cvar)
        },
        "optimization": {
            "optimal_weights_sharpe": opt_sharpe["weights"].tolist(),
            "optimal_sharpe": float(opt_sharpe["sharpe"]),
            "optimal_weights_min_vol": opt_vol["weights"].tolist(),
            "optimal_volatility": float(opt_vol["volatility"])
        },
        "summary": summary
    }