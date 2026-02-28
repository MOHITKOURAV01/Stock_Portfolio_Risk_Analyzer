import numpy as np
from scipy.optimize import minimize
from Backend.risk_metrics import calculate_annualized_performance, calculate_sharpe_ratio

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.02, objective='sharpe'):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]
    
    if objective == 'sharpe':
        def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
            ann_ret, ann_vol = calculate_annualized_performance(weights, mean_returns, cov_matrix)
            return -calculate_sharpe_ratio(ann_ret, ann_vol, risk_free_rate)
        
        result = minimize(neg_sharpe, initial_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
                          
    elif objective == 'volatility':
        def get_volatility(weights, mean_returns, cov_matrix, risk_free_rate):
            _, ann_vol = calculate_annualized_performance(weights, mean_returns, cov_matrix)
            return ann_vol
            
        result = minimize(get_volatility, initial_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        raise ValueError("Objective must be 'sharpe' or 'volatility'")
        
    if not result.success:
        raise Exception(f"Optimization failed: {result.message}")
        
    opt_weights = result.x
    opt_ret, opt_vol = calculate_annualized_performance(opt_weights, mean_returns, cov_matrix)
    opt_sharpe = calculate_sharpe_ratio(opt_ret, opt_vol, risk_free_rate)
    
    return {
        "weights": opt_weights,
        "return": opt_ret,
        "volatility": opt_vol,
        "sharpe": opt_sharpe
    }
