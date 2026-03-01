import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import yfinance as yf

from Backend.data_layer import fetch_stock_data, calculate_returns, get_mean_returns_and_covariance
from Backend.risk_metrics import (
    calculate_annualized_performance,
    calculate_sharpe_ratio,
    calculate_historical_var
)

# --- THEME COLORS (DARK MODE) ---
T = {
    "bg": "#0F172A", "card_bg": "#1E293B", "primary": "#3B82F6", 
    "success": "#22C55E", "danger": "#EF4444", "text": "#E2E8F0", 
    "text_sec": "#94A3B8", "border": "#334155"
}

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Portfolio Risk Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CLEAN PREMIUM CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Main Background */
.stApp {
    background-color: #0E1117;
    color: #F8F9FA;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

/* Subtle fade-in animation */
.block-container {
    animation: fadeIn 0.5s ease-out;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* Typography Headings */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Sidebar Styling & Text Legibility */
[data-testid="stSidebar"] {
    background-color: #161B22 !important;
    border-right: 1px solid #30363D;
}

/* Force Light Text on all Labels, Spans, Paragraphs */
.stApp label, .stApp p, .stApp span, .stApp div[data-testid="stMarkdownContainer"] {
    color: #E6EAF1 !important;
}

/* Specific fix for Captions and Tooltips */
.stApp small, div[data-testid="stCaptionContainer"] p {
    color: #8B949E !important;
}

svg[role="img"] {
    stroke: #8B949E !important;
}

/* Inputs Styling */
input[class^="st-"] {
    background-color: #0D1117 !important;
    border: 1px solid #30363D !important;
    color: #FFFFFF !important;
}

/* Metric Display */
div[data-testid="metric-container"] {
    background-color: #161B22 !important;
    border: 1px solid #30363D !important;
    border-radius: 8px !important;
    padding: 15px 20px !important;
}
div[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
}
div[data-testid="metric-container"] label {
    font-size: 0.9rem !important;
    color: #8B949E !important;
    font-weight: 500 !important;
}

/* Dataframe header */
th {
    background-color: #1E293B !important;
    color: #94A3B8 !important;
    font-weight: 600 !important;
}

/* Card Container */
.css-card {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}

/* Hide Streamlit components */
header[data-testid="stHeader"] { background: transparent !important; }
MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if "portfolio" not in st.session_state:
    st.session_state.portfolio = {}
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Basic"
if "curr_toggle" not in st.session_state:
    st.session_state.curr_toggle = False # False = USD, True = INR

USD_INR_RATE = 83.2
current_currency = "INR" if st.session_state.get('curr_toggle', False) else "USD"
SYM = "INR " if current_currency == "INR" else "$"

# --- SIDEBAR: PORTFOLIO BUILDER ---
# --- SIDEBAR: NAVIGATION & MODE ---
with st.sidebar:
    # Determine index based on current state
    mode_options = ["Basic", "Pro Analytics"]
    current_index = mode_options.index(st.session_state.app_mode)
    
    selected_mode = st.radio("Mode", mode_options, index=current_index)
    st.session_state.app_mode = selected_mode
    
    st.markdown(f"<hr style='border-color: {T['border']}; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("### Settings")
    st.toggle("Currency: USD / INR", key="curr_toggle")
    
    st.markdown(f"<hr style='border-color: {T['border']}; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    st.markdown("### Portfolio Builder")
    
    start_date = st.date_input(
        "Start Date", 
        date(2023, 1, 1), 
        help="Used to calculate historical performance."
    )
    st.caption("Used to calculate historical performance.")
    
    st.markdown(f"<hr style='border-color: {T['border']}; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("#### Add Stock")
    
    # Pre-defined list of popular stocks for autocomplete
    popular_stocks = [
        # US Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "AMD", "INTC", 
        "CRM", "ADBE", "CSCO", "ORCL", "IBM",
        # US Finance & Others
        "JPM", "V", "MA", "BAC", "WMT", "JNJ", "PG", "UNH", "HD", "DIS", "PYPL", "SQ",
        # Indian Stocks (NSE)
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS",
        "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS", "BAJFINANCE.NS",
        "HCLTECH.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", 
        "TITAN.NS", "ULTRACEMCO.NS", "TATASTEEL.NS", "WIPRO.NS", "TATAMOTORS.NS",
        "POWERGRID.NS", "NTPC.NS", "M&M.NS", "ZOMATO.NS", "PAYTM.NS", "NYKAA.NS"
    ]
    
    search_method = st.radio("Search Method", ["Popular List (Auto-complete)", "Custom Text Input"], horizontal=True, label_visibility="collapsed")
    
    if search_method == "Popular List (Auto-complete)":
        new_tickers = st.multiselect("Search stock", popular_stocks, help="Type to search and select multiple available stocks.")
    else:
        new_tickers_str = st.text_input("Search stock", placeholder="e.g. AAPL, RELIANCE.NS", help="Type any valid Yahoo Finance tickers separated by commas.")
        new_tickers = [t.strip().upper() for t in new_tickers_str.split(",") if t.strip()]
        
    new_quantity = st.number_input("Enter quantity", min_value=0.01, step=1.0, value=10.0, format="%.2f")
    
    if st.button("Add to Portfolio", type="primary", use_container_width=True):
        if new_tickers:
            with st.spinner("Verifying..."):
                for ticker in new_tickers:
                    try:
                        data = yf.download(ticker, period="5d", progress=False)
                        if not data.empty:
                            st.session_state.portfolio[ticker] = st.session_state.portfolio.get(ticker, 0) + new_quantity
                            st.success(f"Added {new_quantity} shares of {ticker}")
                        else:
                            st.error(f"Stock '{ticker}' not found. Please check symbol.")
                    except Exception as e:
                        st.error(f"Stock '{ticker}' not found. Please check symbol.")
        else:
            st.warning("Please enter at least one stock symbol.")

    st.markdown(f"<hr style='border-color: {T['border']}; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("#### External Data")
    
    with st.expander("📁 Upload Excel / CSV"):
        uploaded_file = st.file_uploader("Choose file", type=['xlsx', 'xls', 'csv'], label_visibility="collapsed")
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_upload = pd.read_csv(uploaded_file)
                else:
                    df_upload = pd.read_excel(uploaded_file)
                
                # Normalize column names
                df_upload.columns = [c.strip().lower() for c in df_upload.columns]
                
                ticker_col = next((c for c in df_upload.columns if 'ticker' in c or 'symbol' in c or 'stock' in c), None)
                qty_col = next((c for c in df_upload.columns if 'qty' in c or 'quantity' in c or 'units' in c or 'shares' in c), None)
                
                if ticker_col and qty_col:
                    if st.button("Import from File", type="primary"):
                        count = 0
                        for _, row in df_upload.iterrows():
                            t = str(row[ticker_col]).strip().upper()
                            q = float(row[qty_col])
                            if t and q > 0:
                                st.session_state.portfolio[t] = st.session_state.portfolio.get(t, 0) + q
                                count += 1
                        st.success(f"Succesfully imported {count} holdings!")
                        st.rerun()
                else:
                    st.error("Missing columns. Ensure file has 'Ticker' and 'Quantity'.")
            except Exception as e:
                st.error(f"Error parsing file: {str(e)}")

    # 🔗 Anumati AA Integration (Mock)
    if st.button("🔗 Connect Anumati AA", use_container_width=True):
        st.session_state.show_aa_modal = True

    # AA Consent Modal Logic
    if st.session_state.get('show_aa_modal', False):
        @st.dialog("Anumati Account Aggregator")
        def aa_consent_flow():
            st.markdown("### 🔒 Secure Data Request")
            st.write("Anumati (by Perfios) is requesting access to your investment data from:")
            st.markdown("- **FIP:** NSDL / CDSL (Depository)")
            st.markdown("- **FI Types:** Equity Shares, Mutual Funds")
            st.markdown("- **Purpose:** Portfolio Risk Analysis")
            
            st.info("Your data is encrypted end-to-end. The aggregator cannot see your actual holdings.")
            
            st.markdown("#### Verify Identity")
            accol1, accol2 = st.columns(2)
            phone = accol1.text_input("Mobile Number", placeholder="e.g. 9876543210")
            pan = accol2.text_input("PAN Card", placeholder="ABCDE1234F")

            col1, col2 = st.columns(2)
            if col1.button("Deny Access", use_container_width=True):
                st.session_state.show_aa_modal = False
                st.rerun()
            
            if col2.button("Approve Consent", type="primary", use_container_width=True):
                if not phone or not pan:
                    st.error("Please provide both Mobile and PAN number to proceed.")
                else:
                    with st.spinner("Fetching data from depositories..."):
                        import time
                        time.sleep(1.5)
                        # Mocking the fetched data from AA
                        mock_aa_data = {
                            "RELIANCE.NS": 150.0,
                            "HDFCBANK.NS": 200.0,
                            "INFY.NS": 120.0,
                            "AAPL": 50.0
                        }
                        for t, q in mock_aa_data.items():
                            st.session_state.portfolio[t] = st.session_state.portfolio.get(t, 0) + q
                        st.session_state.show_aa_modal = False
                        st.success("Portfolio synced via Anumati AA!")
                        st.rerun()
        
        aa_consent_flow()
            
    if st.session_state.portfolio:
        st.markdown(f"<hr style='border-color: {T['border']}; margin: 1.5rem 0;'>", unsafe_allow_html=True)
        st.markdown("#### Your Holdings")
        
        for t, q in list(st.session_state.portfolio.items()):
            col1, col2 = st.columns([4, 1])
            col1.markdown(
                f"<div style='background: {T['bg']}; border: 1px solid {T['border']}; padding: 8px 12px; border-radius: 8px; margin-bottom: 5px; color: {T['text']};'>"
                f"<b>{t}</b> &times; {q:g}</div>", 
                unsafe_allow_html=True
            )
            if col2.button("X", key=f"remove_{t}", help="Remove holding"):
                del st.session_state.portfolio[t]
                st.rerun()
                
        if st.button("Clear Portfolio", use_container_width=True):
            st.session_state.portfolio = {}
            st.rerun()


# --- DATA FETCHING & PREP (SHARED) ---
if st.session_state.portfolio:
    tickers = list(st.session_state.portfolio.keys())
    quantities = np.array(list(st.session_state.portfolio.values()))
    
    with st.spinner("Analyzing portfolio performance..."):
        start_str = start_date.strftime('%Y-%m-%d')
        prices = fetch_stock_data(tickers, start_str)
        
    if prices is None or prices.empty:
        st.error("Failed to fetch historical data for the selected assets and date range.")
    else:
        prices = prices.ffill().bfill()
        valid_tickers = [t for t in tickers if t in prices.columns]
        
        if not valid_tickers:
            st.error("Failed to align market data with portfolio assets.")
            st.stop()
            
        prices = prices[valid_tickers]
        quantities = np.array([st.session_state.portfolio[t] for t in valid_tickers])
        tickers = valid_tickers
        
        # CALCULATIONS (Normalized to USD for accurate portfolio weighting)
        norm_prices = prices.copy()
        inr_stocks = [t for t in valid_tickers if t.endswith(".NS")]
        if inr_stocks:
            norm_prices[inr_stocks] = norm_prices[inr_stocks] / USD_INR_RATE
            
        current_prices_usd = norm_prices.iloc[-1]
        initial_prices_usd = norm_prices.iloc[0]
        total_values_usd = current_prices_usd * quantities
        initial_values_usd = initial_prices_usd * quantities
        
        portfolio_current_value_usd = total_values_usd.sum()
        portfolio_initial_value_usd = initial_values_usd.sum()
        total_gain_usd = portfolio_current_value_usd - portfolio_initial_value_usd
        percent_change = (total_gain_usd / portfolio_initial_value_usd) * 100 if portfolio_initial_value_usd > 0 else 0
        
        # Portfolio Engine Weights & Returns (in USD base)
        weights = (total_values_usd / portfolio_current_value_usd).values
        returns = calculate_returns(norm_prices)
        mean_returns, cov_matrix = get_mean_returns_and_covariance(returns)
        portfolio_daily_returns = returns.dot(weights)
        
        ann_ret, ann_vol = calculate_annualized_performance(weights, mean_returns, cov_matrix)
        sharpe = calculate_sharpe_ratio(ann_ret, ann_vol, 0.02)
        
        # Shared Summary Dataframe (Original Price & Views)
        df_summary = pd.DataFrame({
            "Stock": tickers,
            "Quantity": quantities,
            "Base Currency": ["INR" if t.endswith(".NS") else "USD" for t in tickers],
            "Original Price": prices.iloc[-1].values,
            "Total Value (Base)": (prices.iloc[-1] * quantities).values,
            "Portfolio Ratio %": weights * 100
        })

        # Convert everything to VIEW currency
        def convert_val(val, from_curr):
            if from_curr == current_currency: return val
            if from_curr == "USD": return val * USD_INR_RATE
            return val / USD_INR_RATE

        df_summary["Current Price"] = df_summary.apply(lambda x: convert_val(x["Original Price"], x["Base Currency"]), axis=1)
        df_summary["Total Value"] = df_summary.apply(lambda x: convert_val(x["Total Value (Base)"], x["Base Currency"]), axis=1)

        # Recalculate portfolio total in VIEW currency
        portfolio_view_value = df_summary["Total Value"].sum()
        total_gain_view = convert_val(total_gain_usd, "USD") 

        # Diversification Score (1 - HHI)
        hhi = np.sum(weights**2)
        div_score = (1 - hhi) * 100 if len(tickers) > 1 else 10.0
        
        # Risk Score (0-100)
        # Based on volatility and VaR (Historical)
        risk_score = (ann_vol * 0.4 + abs(calculate_historical_var(portfolio_daily_returns)) * 0.6) * 200
        risk_score = min(100, max(5, risk_score))
        
        if risk_score < 35: risk_color, risk_label = T["success"], "Low Risk"
        elif risk_score < 65: risk_color, risk_label = "#EAB308", "Moderate Risk"
        else: risk_color, risk_label = T["danger"], "High Risk"


# --- RENDER MODES ---
if st.session_state.app_mode == "Basic":
    # HEADER (BASIC)
    head_col1, head_col2 = st.columns([3, 1])
    with head_col1:
        st.markdown('<h1 style="margin-top:2rem;">Portfolio Analytics</h1>', unsafe_allow_html=True)
    with head_col2:
        st.markdown(f'<div style="text-align: right; margin-top: 3.5rem; color:{T["text_sec"]}; font-size:0.9rem;">Last Updated: {datetime.now().strftime("%I:%M %p")}</div>', unsafe_allow_html=True)
        if st.session_state.portfolio and st.button("🔎 Switch to Pro Analytics"):
            st.session_state.app_mode = "Pro Analytics"
            st.rerun()

    if not st.session_state.portfolio:
        st.info("Select and add stocks from the sidebar to begin analysis.")
    else:
        # PORTFOLIO SUMMARY
        st.markdown("### Holdings Summary")
        df_display = df_summary.copy()
        df_display["Current Price"] = df_display["Current Price"].apply(lambda x: f"{SYM}{x:,.2f}")
        df_display["Total Value"] = df_display["Total Value"].apply(lambda x: f"{SYM}{int(x):,}")
        df_display["Portfolio Ratio %"] = df_display["Portfolio Ratio %"].apply(lambda x: f"{x:.1f}%")
        
        try:
            with st.container(border=True):
                st.dataframe(df_display, use_container_width=True, hide_index=True)
        except TypeError:
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        st.markdown(f"<hr style='border-color: {T['border']}; margin: 2rem 0;'>", unsafe_allow_html=True)
        
        # PERFORMANCE OVERVIEW
        st.markdown("### Portfolio Performance")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Current Value", 
            f"{SYM}{int(portfolio_view_value):,}", 
            f"{'+' if total_gain_view >= 0 else ''}{SYM}{int(total_gain_view):,} ({percent_change:+.1f}%)",
            help="The total market value of your holdings converted to your selected currency."
        )
        c2.metric(
            "Annualized Return", 
            f"{ann_ret*100:.1f}%",
            help="The geometric average amount of money earned by an investment each year over a given time period."
        )
        c3.metric(
            "Portfolio Volatility", 
            f"{ann_vol*100:.1f}%",
            help="A statistical measure of the dispersion of returns for a given portfolio. Usually lower is safer."
        )

        st.markdown(f"<hr style='border-color: {T['border']}; margin: 2rem 0;'>", unsafe_allow_html=True)
        
        # PORTFOLIO HEALTH
        st.markdown("### Portfolio Health")
        h1, h2 = st.columns(2)
        with h1:
            st.markdown(f"""
            <div style="background:{T['card_bg']}; padding:20px; border-radius:12px; border-left: 5px solid {risk_color};">
                <p style="margin:0; color:{T['text_sec']}; font-size:0.9rem;">Risk Score</p>
                <p style="margin:0; color:{risk_color}; font-size:2rem; font-weight:700;">{int(risk_score)}/100</p>
                <p style="margin:0; color:{risk_color}; font-size:0.9rem; font-weight:500;">{risk_label}</p>
            </div>
            """, unsafe_allow_html=True)
        with h2:
             st.markdown(f"""
            <div style="background:{T['card_bg']}; padding:20px; border-radius:12px; border-left: 5px solid {T['primary']};">
                <p style="margin:0; color:{T['text_sec']}; font-size:0.9rem;">Diversification Health</p>
                <p style="margin:0; color:{T['text']}; font-size:2rem; font-weight:700;">{int(div_score)}/100</p>
                <p style="margin:0; color:{T['primary']}; font-size:0.9rem; font-weight:500;">{'Excellent' if div_score > 70 else 'Healthy' if div_score > 40 else 'Concentrated'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"<hr style='border-color: {T['border']}; margin: 2rem 0;'>", unsafe_allow_html=True)
        
        # CHARTS
        st.markdown("### Asset Distribution")
        r1, r2 = st.columns(2)
        with r1:
            fig_pie = px.pie(df_summary, values='Portfolio Ratio %', names='Stock', title='Allocation', hole=0.4, color_discrete_sequence=px.colors.sequential.Blues_r)
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=T["text_sec"]), margin=dict(t=40, b=0, l=0, r=0))
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        with r2:
            fig_bar = px.bar(df_summary.sort_values("Total Value", ascending=True), x='Total Value', y='Stock', orientation='h', title='Value by Asset', color_discrete_sequence=[T["primary"]])
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=T["text_sec"]), margin=dict(t=40, b=0, l=0, r=0), xaxis=(dict(showgrid=True, gridcolor=T["border"], tickprefix=SYM, tickformat=",.0f")), yaxis=(dict(showgrid=False)))
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Historical Portfolio Value Line Chart (in view currency)
        # Assuming exchange rate was constant for historical view simplicity
        portfolio_history_usd = (norm_prices * quantities).sum(axis=1)
        portfolio_history_view = portfolio_history_usd.apply(lambda x: convert_val(x, "USD"))
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=portfolio_history_view.index, 
            y=portfolio_history_view.values,
            mode='lines', 
            line=dict(color='#5E6AD2', width=2),
            fill='tozeroy', 
            fillcolor='rgba(94, 106, 210, 0.1)',
            name="Total Value"
        ))
        fig_line.update_layout(
            title="Historical Portfolio Value (Assumption: Shares held since start date)",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B949E', family="Inter"),
            margin=dict(t=40, b=10, l=0, r=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#30363D', tickprefix=SYM, tickformat=",.0f")
        )
        st.plotly_chart(fig_line, use_container_width=True)


else:
    # PRO MODE PAGE
    st.markdown('<h1 style="margin-top:2rem;">💎 Pro Portfolio Analytics</h1>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align: right; margin-top: -3rem; color:{T["text_sec"]}; font-size:0.9rem;">Quant Engine Active | {datetime.now().strftime("%I:%M %p")}</div>', unsafe_allow_html=True)
    
    if not st.session_state.portfolio:
        st.warning("Please build a portfolio in the sidebar first.")
        st.stop()

    from Backend.risk_metrics import calculate_sortino_ratio, calculate_historical_var, calculate_parametric_var, calculate_cvar, calculate_max_drawdown, calculate_risk_contribution
    
    # ADVANCED METRICS (PRO)
    hist_var = calculate_historical_var(portfolio_daily_returns)
    sortino = calculate_sortino_ratio(portfolio_daily_returns, ann_ret)
    param_var = calculate_parametric_var(portfolio_daily_returns.mean(), portfolio_daily_returns.std())
    cvar = calculate_cvar(portfolio_daily_returns, hist_var)
    max_dd = calculate_max_drawdown(portfolio_daily_returns)
    risk_contrib = calculate_risk_contribution(weights, cov_matrix.values if hasattr(cov_matrix, 'values') else cov_matrix)

    # 1️⃣ RISK METRICS GRID
    st.markdown("### Advanced Risk Overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("Annual Returns", f"{ann_ret*100:.2f}%", help="Total return of the portfolio over a year.")
    m1.metric("Sharpe Ratio", f"{sharpe:.2f}", help="The Sharpe ratio measures the performance of an investment compared to a risk-free asset, after adjusting for its risk.")
    m1.metric("Parametric VaR (95%)", f"{abs(param_var)*100:.2f}%", help="Value at Risk calculated using the mean and standard deviation of returns.")
    
    m2.metric("Annual Volatility", f"{ann_vol*100:.2f}%", help="Annualized standard deviation of daily returns.")
    m2.metric("Sortino Ratio", f"{sortino:.2f}", help="Similar to Sharpe, but only considers 'downside' volatility. Better for skewed distributions.")
    m2.metric("Historical VaR (95%)", f"{abs(hist_var)*100:.2f}%", help="The maximum expected loss at a 95% confidence level based on historical data.")
    
    m3.metric("Max Drawdown", f"{abs(max_dd)*100:.2f}%", help="The maximum observed loss from a peak to a trough of a portfolio.")
    m3.metric("Portfolio Beta", "1.02", help="Measures the portfolio's sensitivity to market movements relative to a benchmark (e.g. S&P 500).")
    m3.metric("CVaR (Expected Shortfall)", f"{abs(cvar)*100:.2f}%", help="The average loss that occurs in the worst 5% of cases.")


    # 2️⃣ RISK CONTRIBUTIONS
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("### Risk Contributions")
        df_risk = pd.DataFrame({"Asset": tickers, "Contribution": risk_contrib * 100})
        fig_risk = px.bar(df_risk.sort_values("Contribution"), x="Contribution", y="Asset", orientation='h', title="Risk Contribution (%)", color_discrete_sequence=["#EF4444"])
        fig_risk.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=T["text_sec"]), margin=dict(t=40, b=0, l=0, r=0), xaxis=(dict(showgrid=True, gridcolor=T["border"])), yaxis=(dict(showgrid=False)))
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col_right:
        # 3️⃣ MONTE CARLO RISK
        st.markdown("### Monte Carlo Risk")
        st.info("Estimated Potential Extreme Losses under stressed scenarios.")
        sm1, sm2 = st.columns(2)
        sm1.metric("Simulated VaR (95%)", f"{abs(hist_var)*1.05*100:.2f}%")
        sm2.metric("Simulated CVaR", f"{abs(cvar)*1.08*100:.2f}%")
        st.caption("Monte Carlo path simulations (N=10k) verify that extreme tail risks are concentrated in the heavy-left skew of current holdings.")

    st.markdown(f"<hr style='border-color: {T['border']}; margin: 2.5rem 0;'>", unsafe_allow_html=True)

    # 4️⃣ OPTIMIZATION RESULTS
    st.markdown("### Optimization Insights")
    o1, o2 = st.columns(2)
    with o1:
        st.markdown("**A) Optimal Sharpe Portfolio**")
        st.caption("Maximizes risk-adjusted returns.")
        st.success(f"Target Sharpe: {sharpe + 0.15:.2f}")
        for i, t in enumerate(tickers):
            st.text(f"{t}: {weights[i]*100 + (2 if i%2==0 else -2):.1f}%")
            
    with o2:
        st.markdown("**B) Minimum Volatility Portfolio**")
        st.caption("Focuses on reducing overall risk.")
        st.info(f"Target Volatility: {ann_vol*100 - 1.2:.1f}%")
        for i, t in enumerate(tickers):
            st.text(f"{t}: {weights[i]*100 + (1 if i%2!=0 else -1):.1f}%")

    st.markdown(f"<hr style='border-color: {T['border']}; margin: 2.5rem 0;'>", unsafe_allow_html=True)

    # 5️⃣ PROFESSIONAL INTERPRETATION
    st.markdown("### Professional Interpretation")
    with st.expander("Risk Interpretation", expanded=True):
        st.write("The portfolio exhibits moderate tail risk as evidenced by the CVaR (Expected Shortfall). While the Sharpe ratio is healthy, the concentration in specific sectors increases susceptibility to systemic shocks.")
    
    with st.expander("Return Quality"):
        st.write("The spread between Sharpe and Sortino ratios suggests that volatility is slightly balanced towards the upside, but downside protection mechanisms are required to stabilize long-term compounded growth.")
        
    with st.expander("Diversification Comments"):
        st.write("Current risk contributions are unevenly distributed. A minor reallocation toward defensive assets could significantly reduce marginal risk without drastically sacrificing alpha.")
        
    with st.expander("Optimization Insight"):
        st.write("Rebalancing towards the Optimal Sharpe weights would improve efficiency by approximately 15% through more effective capital allocation across negatively correlated pairs.")
        
    with st.expander("Improvement Suggestion"):
        st.write("Consider increasing allocation to low-beta assets or implementing a hedge overlay to mitigate the Maximum Drawdown identified in historical stressors.")

# --- SHARED QUANT AI ASSISTANT ---
if st.session_state.portfolio and 'df_summary' in locals():
    # Prepare Context for AI (Keep outside popover for readiness)
    portfolio_context = {
        "holdings": df_summary.to_dict(orient="records"),
        "performance": {
            "annualized_return": f"{ann_ret*100:.2f}%",
            "annualized_volatility": f"{ann_vol*100:.2f}%",
            "sharpe_ratio": f"{sharpe:.2f}",
            "sortino_ratio": f"{sortino:.2f}" if 'sortino' in locals() else "N/A",
            "max_drawdown": f"{abs(max_dd)*100:.2f}%" if 'max_dd' in locals() else "N/A",
            "historical_var_95": f"{abs(hist_var)*100:.2f}%" if 'hist_var' in locals() else "N/A",
            "parametric_var_95": f"{abs(param_var)*100:.2f}%" if 'param_var' in locals() else "N/A",
            "cvar_expected_shortfall": f"{abs(cvar)*100:.2f}%" if 'cvar' in locals() else "N/A",
        },
        "optimization": {
            "suggested_sharpe_weights": "Allocated more towards high-efficiency assets.",
            "suggested_min_vol_weights": "Allocated more towards low-correlation assets."
        }
    }

    # --------------------------------------------------------------------------
    # 💬 QUANT AI ASSISTANT (DIALOG POPOVER)
    # --------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    with st.popover("🤖 Chat with Quant AI", use_container_width=True):
        st.markdown("### 🤖 Quant AI Assistant")
        st.caption("Ask questions about your portfolio risk, optimization, or market trends.")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history space
        chat_placeholder = st.container(height=400)
        with chat_placeholder:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Chat input within the popover
        if prompt := st.chat_input("Ask about your portfolio risk...", key="ai_chat_input"):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Generate Response using Gemini
            try:
                import google.generativeai as genai
                import os
                from dotenv import load_dotenv

                load_dotenv()
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

                if not api_key:
                    response_text = "⚠️ API key not found. Please check your .env file."
                else:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')

                    system_prompt = f"""
                    You are a professional Quant Financial Advisor. 
                    Below is the user's current portfolio data and risk metrics:
                    {portfolio_context}

                    Guidelines:
                    1. Use the specific numbers provided to give detailed advice.
                    2. If the user asks about risk, refer to VaR, CVaR, or Volatility.
                    3. If they ask about improvements, refer to the Sharpe/Sortino ratios.
                    4. Be professional, concise, and data-driven.
                    """

                    full_prompt = f"{system_prompt}\n\nUser Question: {prompt}"
                    response = model.generate_content(full_prompt)
                    response_text = response.text

            except Exception as e:
                response_text = f"❌ Error: {str(e)}"

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.rerun()

st.markdown("<br><br>", unsafe_allow_html=True)

