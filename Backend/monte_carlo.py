import numpy as np
import pandas as pd

def run_monte_carlo_simulation(mean_returns, cov_matrix, weights, initial_portfolio_value=1.0, num_simulations=10000, time_horizon=252):
    mu = np.array(mean_returns)
    cov = np.array(cov_matrix)
    simulated_returns = np.random.multivariate_normal(mu, cov, num_simulations * time_horizon)
    
    simulated_returns = simulated_returns.reshape((num_simulations, time_horizon, len(mu)))
    sim_portfolio_returns = np.dot(simulated_returns, weights)
    cumulative_growth = np.cumprod(1 + sim_portfolio_returns, axis=1)
    final_values = initial_portfolio_value * cumulative_growth[:, -1]
    total_returns = (final_values / initial_portfolio_value) - 1
    
    return total_returns

def simulated_var_cvar(simulated_returns, confidence_level=0.95):
    alpha = 1 - confidence_level
    sim_var = np.percentile(simulated_returns, alpha * 100)
    
    cvar_losses = simulated_returns[simulated_returns <= sim_var]
    if len(cvar_losses) == 0:
        sim_cvar = 0.0
    else:
        sim_cvar = cvar_losses.mean()
        
    return sim_var, sim_cvar
