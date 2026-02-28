import yfinance as yf
import pandas as pd
import numpy as np

def fetch_stock_data(tickers, start_date, end_date=None):
    data = yf.download(tickers, start=start_date, end=end_date)
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']
    else:
        data = pd.DataFrame(data['Close'])
        data.columns = tickers

    data = data.dropna()
    return data

def fetch_market_data(start_date, end_date=None, ticker="^GSPC"):

    market_data = yf.download(ticker, start=start_date, end=end_date)
    return market_data["Close"].dropna()

def calculate_returns(prices_df):

    returns = prices_df.pct_change().dropna()
    return returns

def get_mean_returns_and_covariance(returns_df):

    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    return mean_returns, cov_matrix

def calculate_portfolio_returns(returns_df, weights):

    portfolio_returns = returns_df.dot(weights)
    return portfolio_returns
