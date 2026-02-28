from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship, declarative_base
from database.db import engine

Base = declarative_base()

class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)

class Asset(Base):
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    ticker = Column(String)
    weight = Column(Float)

class RiskReport(Base):
    __tablename__ = "risk_reports"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    annual_return = Column(Float)
    volatility = Column(Float)
    sharpe_ratio = Column(Float)
    var_95 = Column(Float)
    max_drawdown = Column(Float)

def create_tables():
    Base.metadata.create_all(bind=engine)