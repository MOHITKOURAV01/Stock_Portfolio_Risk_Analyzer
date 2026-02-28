import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS_PER_YEAR = 252

def calculate_annualized_performance(weights, mean_returns, cov_matrix):
    portfolio_mean_daily = np.dot(weights, mean_returns)
    portfolio_var_daily = np.dot(weights.T, np.dot(cov_matrix, weights))
    
    annual_return = portfolio_mean_daily * TRADING_DAYS_PER_YEAR
    annual_volatility = np.sqrt(portfolio_var_daily) * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return annual_return, annual_volatility

def calculate_sharpe_ratio(annual_return, annual_volatility, risk_free_rate=0.02):
    if annual_volatility == 0:
        return 0.0
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    return sharpe_ratio

def calculate_sortino_ratio(portfolio_returns_daily, annual_return, risk_free_rate=0.02):
    downside_returns = portfolio_returns_daily[portfolio_returns_daily < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
        
    downside_std_annualized = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sortino_ratio = (annual_return - risk_free_rate) / downside_std_annualized
    return sortino_ratio

def calculate_historical_var(portfolio_returns_daily, confidence_level=0.95):
    alpha = 1 - confidence_level
    var_hist = np.percentile(portfolio_returns_daily, alpha * 100)
    return var_hist

def calculate_parametric_var(portfolio_mean_daily, portfolio_volatility_daily, confidence_level=0.95):
    z_score = norm.ppf(1 - confidence_level)
    var_param = portfolio_mean_daily + (z_score * portfolio_volatility_daily)
    return var_param

def calculate_cvar(portfolio_returns_daily, var_threshold):
    losses_beyond_var = portfolio_returns_daily[portfolio_returns_daily <= var_threshold]
    if len(losses_beyond_var) == 0:
        return 0.0
    cvar = losses_beyond_var.mean()
    return cvar

def calculate_max_drawdown(portfolio_returns_daily):
    cumulative = (1 + portfolio_returns_daily).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_portfolio_beta(portfolio_returns_daily, market_returns_daily):
    aligned_data = pd.concat([portfolio_returns_daily, market_returns_daily], axis=1).dropna()
    aligned_data.columns = ["Portfolio", "Market"]
    
    if len(aligned_data) < 2:
        return 0.0
        
    cov_pm = np.cov(aligned_data["Portfolio"], aligned_data["Market"])[0][1]
    var_market = np.var(aligned_data["Market"])
    
    if var_market == 0:
        return 0.0
        
    beta = cov_pm / var_market
    return beta

def calculate_risk_contribution(weights, cov_matrix):
    weights = np.array(weights)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    
    if portfolio_variance == 0:
        return np.zeros(len(weights))
        
    marginal_contribution = np.dot(cov_matrix, weights)
    risk_contribution = (weights * marginal_contribution) / portfolio_variance
    return risk_contribution
