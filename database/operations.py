from database.db import get_connection
from database.models import Portfolio, Asset, RiskReport
from sqlalchemy.orm import Session

def save_portfolio(name: str):
    db = get_connection()
    portfolio = Portfolio(name=name)
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)
    db.close()
    return portfolio.id

def save_assets(portfolio_id: int, tickers, weights):
    db = get_connection()
    for ticker, weight in zip(tickers, weights):
        asset = Asset(
            portfolio_id=portfolio_id,
            ticker=ticker,
            weight=weight
        )
        db.add(asset)
    db.commit()
    db.close()

def save_risk_report(portfolio_id: int, metrics: dict):
    db = get_connection()
    report = RiskReport(
        portfolio_id=portfolio_id,
        annual_return=metrics["annual_return"],
        volatility=metrics["volatility"],
        sharpe_ratio=metrics["sharpe_ratio"],
        var_95=metrics["var_95"],
        max_drawdown=metrics["max_drawdown"]
    )
    db.add(report)
    db.commit()
    db.close()