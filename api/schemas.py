from pydantic import BaseModel
from typing import List

class PortfolioCreate(BaseModel):
    name: str
    tickers: List[str]
    weights: List[float]
    start_date: str