from fastapi import APIRouter
from api.schemas import PortfolioCreate
from Backend.data_layer import fetch_stock_data, calculate_returns, get_mean_returns_and_covariance, calculate_portfolio_returns
from Backend.risk_metrics import calculate_annualized_performance, calculate_sharpe_ratio, calculate_historical_var, calculate_max_drawdown
from Backend.monte_carlo import run_monte_carlo_simulation
from database.operations import save_portfolio, save_assets, save_risk_report
import numpy as np

router = APIRouter()

@router.post("/portfolio")
def create_portfolio(data: PortfolioCreate):
    portfolio_id = save_portfolio(data.name)
    save_assets(portfolio_id, data.tickers, data.weights)

    prices = fetch_stock_data(data.tickers, data.start_date)
    returns = calculate_returns(prices)
    mean_returns, cov_matrix = get_mean_returns_and_covariance(returns)

    weights = np.array(data.weights)
    ann_ret, ann_vol = calculate_annualized_performance(weights, mean_returns, cov_matrix)
    sharpe = calculate_sharpe_ratio(ann_ret, ann_vol)
    portfolio_daily = calculate_portfolio_returns(returns, weights)
    var_95 = calculate_historical_var(portfolio_daily)
    max_dd = calculate_max_drawdown(portfolio_daily)

    metrics = {
        "annual_return": ann_ret,
        "volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "var_95": var_95,
        "max_drawdown": max_dd
    }

    save_risk_report(portfolio_id, metrics)

    simulations = run_monte_carlo_simulation(mean_returns, cov_matrix, weights)

    return {
        "portfolio_id": portfolio_id,
        "metrics": metrics,
        "simulation_mean_return": float(np.mean(simulations))
    }