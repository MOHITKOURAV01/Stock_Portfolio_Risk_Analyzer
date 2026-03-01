# 📊 Stock Portfolio Risk Analyzer

Professional-grade portfolio risk analytics tool for retail investors, built using Python, Streamlit, and quantitative finance models.

---

## 1. Problem Statement

### Problem Title
Lack of Accessible Portfolio Risk Analytics for Retail Investors

### Problem Description
Retail investors increasingly manage diversified stock portfolios. However, understanding true portfolio risk requires more than observing daily gains and losses.

Professional risk metrics such as Value at Risk (VaR), CVaR, Sharpe Ratio, beta, and correlation matrices provide deeper insight into portfolio exposure. These tools are typically available only through expensive institutional platforms.

Most retail investors:
- Do not measure portfolio volatility properly
- Overestimate diversification
- Ignore tail risk
- Make decisions based on intuition rather than statistical analysis

### Target Users
- Retail investors
- Finance students
- Quant enthusiasts
- Long-term portfolio managers
- Academic researchers

### Existing Gaps
- No lightweight desktop tool integrating risk metrics + simulations
- Limited free tools for Monte Carlo risk modeling
- Lack of interactive visualization for portfolio analytics
- Poor understanding of diversification risk

---

## 2. Problem Understanding & Approach

### Root Cause Analysis
Retail investors lack:
- Structured risk analytics
- Statistical modeling tools
- Visualization of portfolio exposure
- Scenario-based stress testing

Existing brokerage dashboards focus mainly on:
- P&L tracking
- Basic performance metrics
- No advanced risk decomposition

### Solution Strategy
Build a desktop-based interactive portfolio risk engine that:

- Accepts portfolio holdings
- Fetches historical data using yfinance
- Computes professional-grade risk metrics
- Runs Monte Carlo simulations
- Supports scenario analysis
- Visualizes risk exposure interactively

---

## 3. Proposed Solution

### Solution Overview
A Python-based Stock Portfolio Risk Analyzer that combines statistical finance models with interactive dashboards to help investors understand and manage portfolio risk.

### Core Idea
Convert historical stock data into actionable risk insights using:

- Quantitative finance formulas
- Monte Carlo simulations
- Matrix algebra
- Modern portfolio theory (Markowitz)

### Key Features

- **Multi-Market Support:** Unified tracking for both **Indian (NSE)** and **US (NYSE/NASDAQ)** stock markets.
- **Multi-Currency Toggle:** Real-time conversion between **USD ($)** and **INR (₹)** for consistent portfolio viewing.
- **Smart Portfolio Import:** 
    - Manual ticker entry with auto-complete.
    - **Bulk Upload:** Import holdings via Excel or CSV files.
    - **Anumati AA Integration:** Securely sync portfolio from CDSL/NSDL using Phone and PAN card (Demo Mode).
- **Dual Analytical Modes:**
    - **Basic Mode:** Simplified health scores and distribution charts for beginners.
    - **Pro Analytics:** Institutional-grade metrics (VaR, CVaR, Sharpe, Beta) for advanced users.
- **AI-Powered Insights:** 
    - **Quant AI Assistant:** Context-aware chatbot (Gemini 2.5 Flash) to analyze your specific risk profile.
    - Professional interpretation of complex risk data.
- **Advanced Quantitative Engines:**
    - Monte Carlo simulation (10k paths) for tail-risk estimation.
    - Markowitz Mean-Variance Portfolio Optimization.
    - Risk Contribution decomposition per asset.
    - Historical and Parametric Value at Risk (VaR).

---

## 4. System Architecture

### High-Level Flow
User → Streamlit Frontend → Python Analytics Engine → Risk Models → Visualization → Response

### Architecture Description

1. User inputs portfolio tickers and weights.
2. Backend fetches historical adjusted close data using yfinance.
3. Data is cleaned and converted to log returns.
4. Statistical metrics and covariance matrix are computed.
5. Risk models (VaR, CVaR, Monte Carlo, Optimization) are executed.
6. Results are visualized using Plotly.
7. Interactive dashboard displays analytics.

### Architecture Diagram :- 



```mermaid
graph TD
    %% User Interaction Layer
    subgraph Client_Layer [User Interface - Streamlit]
        User([Retail Investor])
        Dashboard[Interactive Dashboard]
        Inputs[Tickers, Weights & Parameters]
    end

    %% Processing & Logic Layer
    subgraph Logic_Layer [Analytics Engine - Python]
        Parser[Input Validator]
        
        subgraph Engine_Components
            Calc[Returns & Volatility Calc]
            MC_Sim[Monte Carlo Engine]
            Opt[Markowitz Portfolio Optimizer]
        end
        
        Models[Quant Models: VaR, CVaR, Sharpe]
    end

    %% Data Acquisition Layer
    subgraph Data_Layer [External Data Source]
        YF[yfinance API]
        DB[(Local Cache / CSV)]
    end

    %% Visualization & Output Layer
    subgraph Output_Layer [Reporting & Viz]
        Plotly[Plotly Chart Generator]
        Report[Risk Summary Table]
    end

    %% Data Flow Connections
    User --> Inputs
    Inputs --> Parser
    Parser --> YF
    YF --> DB
    DB --> Calc
    
    Calc --> MC_Sim
    Calc --> Opt
    MC_Sim --> Models
    Opt --> Models
    
    Models --> Plotly
    Models --> Report
    Plotly --> Dashboard
    Report --> Dashboard
    Dashboard --> User
```



## 5. Database Design

### ER Diagram :- 



```mermaid
erDiagram
    USER ||--o{ PORTFOLIO : owns
    PORTFOLIO ||--|{ ASSETS : contains
    ASSETS ||--o{ PRICE_HISTORY : tracks
    PORTFOLIO ||--|| RISK_REPORT : generates

    USER {
        string user_id PK
        string name
        string email
    }

    PORTFOLIO {
        int portfolio_id PK
        string portfolio_name
        float total_value
        float risk_free_rate
    }

    ASSETS {
        string ticker_symbol PK
        float weight
        float shares_count
        float average_buy_price
    }

    PRICE_HISTORY {
        date trading_date PK
        string ticker_symbol FK
        float adj_close
        float daily_return
        float volatility
    }

    RISK_REPORT {
        int report_id PK
        float value_at_risk_95
        float conditional_var
        float sharpe_ratio
        float max_drawdown
        datetime generated_at
    }
```



### ER Diagram Description

Entities:

- User Portfolio
- Assets (Ticker, Weight)
- Historical Price Data
- Computed Metrics
- Simulation Results

Relationships:

- One portfolio contains multiple assets.
- Each asset maps to historical price records.
- Risk metrics are computed per portfolio.

Note: MVP version may not require persistent database (in-memory computation).

---

## 6. Dataset Selected

### Dataset Name
Historical Stock Price Data

### Source
Yahoo Finance via yfinance Python library

### Data Type
- Daily Adjusted Close Prices
- Volume Data
- Market Index Data (for beta calculation)

### Selection Reason
- Free and accessible
- Decades of historical data
- Supports equities and ETFs
- Sufficient for risk modeling

### Preprocessing Steps
- Download adjusted close prices
- Align dates across tickers
- Handle missing values
- Compute log returns
- Calculate covariance matrix
- Normalize portfolio weights

---

## 7. Model Selected

### Model Name
Modern Portfolio Theory + Statistical Risk Modeling

Includes:
- Historical Simulation VaR
- Parametric Gaussian VaR
- CVaR (Expected Shortfall)
- Monte Carlo Simulation
- Markowitz Mean-Variance Optimization

### Selection Reasoning
- Industry-standard risk measurement techniques
- Computationally efficient
- Interpretable results
- Suitable for retail-level portfolios

### Alternatives Considered
- GARCH volatility models
- Student-t distribution modeling
- Black-Litterman model
- Factor models (Fama-French)

### Evaluation Metrics
- Portfolio volatility
- VaR at 95% and 99%
- Sharpe ratio
- Maximum drawdown
- Risk-adjusted returns

---

## 8. Technology Stack

### Frontend
- Streamlit
- Plotly (interactive charts)

### Backend
- Python
- NumPy
- Pandas

### ML/AI
- Monte Carlo simulation
- Statistical risk modeling
- Optimization algorithms

### Database
- Optional: SQLite 
- MVP: In-memory processing

### Deployment
- Local desktop application
- Optional: Streamlit Cloud / Render

---

## 9. API Documentation & Testing

### API Endpoints List

(For MVP, internal Python functions are used instead of REST APIs)

- Fetch Data Module
- Risk Calculation Module
- Monte Carlo Simulation Module
- Optimization Module

### API Testing Screenshots
(Add Postman / Thunder Client screenshots here if converted to FastAPI)

---

## 10. Module-wise Development & Deliverables

### Checkpoint 1: Research & Planning
- Risk model selection
- Mathematical validation
- UI wireframe design

### Checkpoint 2: Backend Development
- Data fetching engine
- Return computation
- Covariance matrix
- VaR & Sharpe implementation

### Checkpoint 3: Frontend Development
- Streamlit dashboard
- Portfolio input UI
- Visualization integration

### Checkpoint 4: Model Training
(Not ML training-based; statistical parameter estimation)
- Estimate mean returns
- Estimate covariance matrix

### Checkpoint 5: Model Integration
- Monte Carlo simulation
- Efficient frontier generation
- Optimization constraints

### Checkpoint 6: Deployment
- Streamlit deployment
- Performance optimization
- Documentation completion

---

## 11. End-to-End Workflow

1. User inputs portfolio.
2. Historical data fetched via yfinance.
3. Returns and covariance matrix computed.
4. Risk metrics calculated.
5. Monte Carlo simulation executed.
6. Efficient frontier generated.
7. Interactive dashboard displays results.

---

## 12. Demo & Video

- Live Demo Link: [Live Demo](https://stock-portfolio-risk-analyzer.streamlit.app/)
- Demo Video Link: [Demo Video](https://drive.google.com/file/d/1cx1fcwWKGUQ7HwutU79fcPmHzKjGKnO4/view?usp=sharing)
- GitHub Repository: [Github Repo](https://github.com/MOHITKOURAV01/Stock_Portfolio_Risk_Analyzer)
- Model: [Colab Model](https://colab.research.google.com/drive/1TS-V3BQzpyRnD8z2RfCm5I5ggw4K072t?usp=sharing)
-PPT: [PPT](https://www.canva.com/design/DAHCqkaDW08/60dbrUfDXKUhV_l_ut9uRg/edit?utm_content=DAHCqkaDW08&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

---

## 13. Hackathon Deliverables Summary

- Functional risk analytics dashboard
- Monte Carlo simulation engine
- Portfolio optimization module
- Interactive visualization interface

---

## 14. Team Roles & Responsibilities

| Member Name | Role | Responsibilities |
|-------------|------|-----------------|
| Meet Ramatri | Quant Developer | Risk modeling, Monte Carlo simulation |
| Ambuj Vashistha | Backend Engineer | Data processing & optimization |
| Mohit Kourav | Frontend Developer | Streamlit dashboard & visualization |

---

## 15. Future Scope & Scalability

### Short-Term
- Add CVaR optimization
- Add downside risk analysis
- Add portfolio comparison feature

### Long-Term
- Add Black-Litterman model
- Add factor-based modeling
- Add real-time streaming data
- Convert into SaaS platform

---

## 16. Known Limitations

- Assumes normal distribution in parametric VaR
- Relies on historical data assumptions
- No real-time high-frequency data
- Market regime shifts not modeled

---

## 17. Impact

- Improves retail investor risk awareness
- Encourages data-driven decision making
- Demonstrates quantitative finance implementation
- Bridges gap between institutional analytics and retail tools
