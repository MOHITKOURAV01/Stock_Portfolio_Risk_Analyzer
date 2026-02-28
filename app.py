import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(page_title="Institutional Risk Dashboard", layout="wide")

# --- CUSTOM CSS ---
def local_css():
    st.markdown("""
        <style>
        /* Institutional Dark Theme - Upgraded with Clean Glassmorphism */
        .stApp {
            background-color: #050510;
            background-image: linear-gradient(rgba(5, 5, 12, 0.88), rgba(5, 5, 12, 0.88)), 
                              url("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=2070&auto=format&fit=crop");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            color: #eceff4;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }
        
        /* Premium Metric Card with Blur & Glow */
        .metric-card {
            background-color: rgba(20, 25, 35, 0.65);
            padding: 18px;
            border-radius: 8px;
            border-left: 4px solid #005a9e; /* Default, overridden inline */
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-top: 1px solid rgba(255,255,255,0.08);
            border-right: 1px solid rgba(255,255,255,0.03);
            border-bottom: 1px solid rgba(255,255,255,0.03);
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        }
        .metric-card:hover {
            box-shadow: 0 8px 25px rgba(0,0,0,0.6);
            transform: translateY(-4px);
            border-top: 1px solid rgba(255,255,255,0.15);
        }
        .metric-card h4 {
            margin-bottom: 6px;
            font-size: 0.85rem;
            color: #a0aec0;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }
        .metric-card h2 {
            margin: 0;
            font-size: 1.8rem;
            font-weight: 600;
            color: #ffffff;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        /* Smooth Title Entry */
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animated-title {
            animation: fadeInDown 1.0s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
            color: #ffffff;
            padding: 15px 0 25px 0;
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        /* Sidebar styling - Darker focus */
        [data-testid="stSidebar"] {
            background-color: rgba(10, 12, 18, 0.95) !important;
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        
        /* Buttons */
        .stButton>button {
            border-radius: 4px;
            font-weight: 600;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            transition: all 0.2s;
        }
        .stButton>button:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# --- TITLE ---
st.markdown('<div class="animated-title">Institutional Portfolio Risk Analytics</div>', unsafe_allow_html=True)

# --- CACHED DATA FETCHING & MATH ---
@st.cache_data(ttl=3600)
def fetch_data(tickers, lookback_years=5):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback_years * 365)
    
    # yfinance often returns MultiIndex columns and mixed dtypes.
    try:
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
        
        # 1. Handle MultiIndex columns (Extract 'Adj Close' or 'Close')
        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.levels[0]:
                df = data['Adj Close'].copy()
            else:
                df = data['Close'].copy()
        else:
            df = data.copy()
            # If standard DataFrame, try selecting directly if columns exist
            if 'Adj Close' in df.columns:
                df = df['Adj Close']
            elif 'Close' in df.columns:
                df = df['Close']
                
        # 2. Handle Single Ticker fallback: 
        # If df is a Series, convert it to a DataFrame with the ticker name as the column
        if isinstance(df, pd.Series):
            df = df.to_frame(name=tickers[0])
            
        # 3. Handle Missing/Empty Data
        df = df.dropna(how='all')    # Drop dates where ALL tickers are NaN
        df = df.ffill().bfill()      # Fill intermittent NaNs (forward then backward)
        
        if df.empty:
            st.error("Error: Fetched data is completely empty. Check ticker symbols.")
            return None
            
        return df
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return None

@st.cache_data
def calculate_metrics(data, risk_free_rate):
    returns = np.log(data / data.shift(1)).dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    return returns, mean_returns, cov_matrix

# --- OPTIMIZATION ALGORITHMS ---
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    p_ret, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_std

def get_optimal_portfolio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for asset in range(num_assets))
    
    initial_guess = num_assets * [1. / num_assets]
    result = minimize(negative_sharpe, initial_guess, args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# --- SIDEBAR: CONFIGURATION ---
st.sidebar.markdown("### Configuration Panel")

st.sidebar.markdown("#### Selection")
default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
available_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'BTC-USD', 'SPY', 'QQQ', 'TLT', 'GLD', 'JNJ', 'PG']
tickers = st.sidebar.multiselect("Assets", options=available_tickers, default=default_tickers)

st.sidebar.markdown("#### Allocation (%)")
weights_input = []
total_weight = 0

if len(tickers) > 0:
    for ticker in tickers:
        w = st.sidebar.number_input(f"{ticker} Weight", min_value=0.0, max_value=100.0, value=100.0/len(tickers), step=1.0, format="%.1f")
        weights_input.append(w)
        total_weight += w

calculate_disabled = False
valid_weights = []

if abs(total_weight - 100.0) > 0.01 and len(tickers) > 0:
    st.sidebar.error(f"Allocation must equal exactly 100.0%. Current: {total_weight:.1f}%")
    calculate_disabled = True
elif len(tickers) > 0:
    valid_weights = [w/100.0 for w in weights_input]

st.sidebar.markdown("#### Simulation")
num_simulations = st.sidebar.number_input("Iterative Paths", min_value=1000, max_value=50000, value=10000, step=1000)
time_horizon = st.sidebar.number_input("Time Horizon (Days)", min_value=30, max_value=1260, value=252, step=30)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=4.0, step=0.1) / 100.0

st.sidebar.markdown("#### Diagnostics")
stress_test = st.sidebar.checkbox("Apply Market Stress Regime")

if len(tickers) > 1:
    optimize_btn = st.sidebar.button("Optimize Portfolio (Max Sharpe)")
    if optimize_btn:
        with st.spinner("Calculating optimal weights..."):
            data = fetch_data(tickers)
            if data is not None and not data.empty:
                _, mean_rets, cov_mat = calculate_metrics(data, risk_free_rate)
                opt_weights = get_optimal_portfolio(mean_rets, cov_mat, risk_free_rate)
                st.sidebar.success("Optimization Result:")
                for t, w in zip(tickers, opt_weights):
                    st.sidebar.text(f" {t}: {w*100:.1f}%")
            else:
                st.sidebar.error("Could not fetch data for optimization.")

if st.sidebar.button("System Calculate", disabled=calculate_disabled or len(tickers) == 0, use_container_width=True, type="primary"):
    with st.spinner("Initializing Analytics Engine..."):
        
        # --- 1. DATA PREPARATION ---
        data = fetch_data(tickers)
        if data is None or data.empty:
            st.stop()
            
        returns, mean_returns, cov_matrix = calculate_metrics(data, risk_free_rate)
        
        # Stress Test Adjustments (Severe impact: -20% annual return, 2x variance)
        if stress_test:
            mean_returns = mean_returns - 0.20
            cov_matrix = cov_matrix * 2.0
            
        weights = np.array(valid_weights)
        
        # Portfolio Baseline
        port_return = np.sum(mean_returns * weights)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (port_return - risk_free_rate) / port_volatility if port_volatility > 0 else 0
        
        # --- 2. MONTE CARLO SIMULATION ---
        np.random.seed(42)
        initial_investment = 1000000 # Institutional baseline $1M
        
        L = np.linalg.cholesky(cov_matrix)
        dt = 1/252
        portfolio_paths = np.zeros((time_horizon, num_simulations))
        portfolio_paths[0] = initial_investment
        
        for t in range(1, time_horizon):
            z = np.random.standard_normal((len(tickers), num_simulations))
            
            # Convert pandas series to numpy arrays to avoid ValueError on multi-dimensional indexing
            mu_array = mean_returns.values if isinstance(mean_returns, pd.Series) else mean_returns
            diag_cov = np.diag(cov_matrix)
            
            # Calculate daily returns for this step
            drift = (mu_array - 0.5 * diag_cov)[:, np.newaxis] * dt
            shock = np.sqrt(dt) * np.dot(L, z)
            daily_returns = drift + shock
            
            port_daily_ret = np.sum(weights[:, np.newaxis] * daily_returns, axis=0)
            portfolio_paths[t] = portfolio_paths[t-1] * np.exp(port_daily_ret)
            
        returns_mc = (portfolio_paths[-1] - initial_investment) / initial_investment
        
        # --- 3. RISK METRICS (VaR & CVaR) ---
        confidence_level = 0.05
        # Historical / MC Simulation VaR
        sorted_returns = np.sort(returns_mc)
        var_index = int(num_simulations * confidence_level)
        var_95_percent = sorted_returns[var_index]
        mc_var_95 = initial_investment * abs(min(var_95_percent, 0))
        
        # CVaR (Expected Shortfall) accurately tracking losses beyond VaR
        cvar_returns = sorted_returns[:var_index]
        mc_cvar_95 = initial_investment * abs(np.mean(cvar_returns)) if len(cvar_returns) > 0 else mc_var_95
        
        # --- 4. HISTORICAL PERFORMANCE (Drawdown & Vol) ---
        hist_port_returns = returns.dot(weights)
        cumulative_returns = (1 + hist_port_returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / rolling_max) - 1
        max_drawdown = drawdown.min()
        
        # Rolling Volatility (60-day window)
        rolling_vol = hist_port_returns.rolling(window=60).std() * np.sqrt(252)

        # --- KPI CARDS ---
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Dynamic colored borders/glow based on performance
        ret_color = "#00e676" if port_return > 0 else "#ff1744"
        vol_color = "#ffb300"
        sharpe_color = "#00b0ff"
        var_color = "#ff5252"
        cvar_color = "#d50000"
        
        col1.markdown(f'<div class="metric-card" style="border-left-color: {ret_color};"><h4>Exp. Return</h4><h2 style="color: {ret_color}; text-shadow: 0 0 10px rgba(0,230,118,0.2);">{port_return*100:+.2f}%</h2></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card" style="border-left-color: {vol_color};"><h4>Volatility</h4><h2 style="color: #ffffff;">{port_volatility*100:.2f}%</h2></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-card" style="border-left-color: {sharpe_color};"><h4>Sharpe Ratio</h4><h2 style="color: {sharpe_color}; text-shadow: 0 0 10px rgba(0,176,255,0.2);">{sharpe_ratio:.2f}</h2></div>', unsafe_allow_html=True)
        col4.markdown(f'<div class="metric-card" style="border-left-color: {var_color};"><h4>VaR (95%)</h4><h2 style="color: {var_color}; text-shadow: 0 0 10px rgba(255,82,82,0.2);">${mc_var_95:,.0f}</h2><p style="margin:0; font-size:11px; color:#9ba4b5;">Simulated {time_horizon} days</p></div>', unsafe_allow_html=True)
        col5.markdown(f'<div class="metric-card" style="border-left-color: {cvar_color};"><h4>CVaR (95%)</h4><h2 style="color: {cvar_color}; text-shadow: 0 0 10px rgba(213,0,0,0.2);">${mc_cvar_95:,.0f}</h2><p style="margin:0; font-size:11px; color:#9ba4b5;">Expected Shortfall</p></div>', unsafe_allow_html=True)

        st.markdown("<hr style='border-color: #222;'>", unsafe_allow_html=True)

        # --- VISUALIZATIONS ---
        
        st.markdown("### Historical Analytics")
        # Drawdown & Cumulative Return Chart
        fig_hist = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.08, row_heights=[0.5, 0.25, 0.25],
                               subplot_titles=("Cumulative Returns", "Drawdown", "Rolling 60-Day Volatility"))
        
        fig_hist.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values - 1, 
                                    name='Returns', line=dict(color='#00b0ff', width=1.5)), row=1, col=1)
        fig_hist.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, 
                                    name='Drawdown', fill='tozeroy', line=dict(color='#ff5252', width=1)), row=2, col=1)
        fig_hist.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values, 
                                    name='Volatility', line=dict(color='#ffb300', width=1)), row=3, col=1)
        
        fig_hist.update_layout(template='plotly_dark', margin=dict(l=20, r=20, t=40, b=20),
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,10,0.6)',
                             height=650, hovermode='x unified', font=dict(family="Helvetica", color="#E0E0E0"), showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Split next section
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### Efficient Frontier Allocation")
            num_portfolios = 3000
            results = np.zeros((3, num_portfolios))
            for i in range(num_portfolios):
                rw = np.random.random(len(tickers))
                rw /= np.sum(rw)
                p_r, p_v = portfolio_performance(rw, mean_returns, cov_matrix)
                results[0,i] = p_v
                results[1,i] = p_r
                results[2,i] = (p_r - risk_free_rate) / p_v
                
            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(x=results[0,:], y=results[1,:], mode='markers', 
                                        marker=dict(color=results[2,:], colorscale='Viridis', showscale=True, 
                                                    size=3, opacity=0.4, colorbar=dict(title="Sharpe")), 
                                        name='Random Allocations'))
            fig_ef.add_trace(go.Scatter(x=[port_volatility], y=[port_return], mode='markers', 
                                        marker=dict(color='#ffffff', symbol='star', size=14, line=dict(width=1, color='black')), 
                                        name='Current Portfolio'))
            
            fig_ef.update_layout(template='plotly_dark', margin=dict(l=20, r=20, t=20, b=20),
                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,10,0.6)', 
                                 xaxis_title='Volatility \u03C3', yaxis_title='Expected Return \u03BC',
                                 font=dict(family="Helvetica", color="#E0E0E0"))
            st.plotly_chart(fig_ef, use_container_width=True)

        with c2:
            st.markdown(f"#### Monte Carlo Trajectories ({num_simulations})")
            fig_mc = go.Figure()
            
            sample_paths = portfolio_paths[:, :100]
            time_array = np.arange(time_horizon)
            
            for i in range(100):
                fig_mc.add_trace(go.Scatter(x=time_array, y=sample_paths[:, i], mode='lines', 
                                            line=dict(color='rgba(0, 150, 255, 0.04)'), showlegend=False, hoverinfo='skip'))
            
            fig_mc.add_trace(go.Scatter(x=time_array, y=np.mean(portfolio_paths, axis=1), mode='lines', 
                                        line=dict(color='#ffea00', width=2), name='Expected Value'))
            
            fig_mc.update_layout(template='plotly_dark', margin=dict(l=20, r=20, t=20, b=20),
                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,10,0.6)', 
                                 yaxis_title='Portfolio Value ($)', xaxis_title='Trading Days',
                                 font=dict(family="Helvetica", color="#E0E0E0"))
            st.plotly_chart(fig_mc, use_container_width=True)

        # --- DIAGNOSTICS ---
        d1, d2 = st.columns(2)
        
        with d1:
            st.markdown("#### Correlation Matrix")
            corr_matrix = returns.corr()
            fig_corr = px.imshow(corr_matrix, x=tickers, y=tickers, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, text_auto=".2f")
            fig_corr.update_layout(template='plotly_dark', margin=dict(l=20, r=20, t=20, b=20),
                                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,10,0.6)', font=dict(family="Helvetica", color="#E0E0E0"))
            st.plotly_chart(fig_corr, use_container_width=True)

        with d2:
            st.markdown("#### Marginal Risk Contribution")
            port_variance = port_volatility**2
            marginal_contrib = np.dot(cov_matrix, weights) / port_volatility
            component_contrib = weights * marginal_contrib
            percentage_contrib = component_contrib / port_volatility * 100
            
            fig_rc = go.Figure(data=[go.Pie(labels=tickers, values=percentage_contrib, hole=0.55, 
                                            marker=dict(colors=px.colors.qualitative.Prism))])
            fig_rc.update_layout(template='plotly_dark', margin=dict(l=20, r=20, t=20, b=20),
                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,10,0.6)', font=dict(family="Helvetica", color="#E0E0E0"),
                                 annotations=[dict(text='Risk %', x=0.5, y=0.5, font_size=16, showarrow=False, font=dict(family="Helvetica", color="#E0E0E0"))])
            fig_rc.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_rc, use_container_width=True)

        # --- EXPORT REPORT ---
        st.markdown("#### System Export")
        
        export_df = pd.DataFrame({
            'Metric': ['Expected Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'VaR (95%)', 'CVaR (95%)'],
            'Value': [f"{port_return*100:.2f}%", f"{port_volatility*100:.2f}%", f"{sharpe_ratio:.2f}", f"{max_drawdown*100:.2f}%", f"${mc_var_95:,.2f}", f"${mc_cvar_95:,.2f}"]
        })
        
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Diagnostic Report (CSV)",
            data=csv,
            file_name='portfolio_diagnostics.csv',
            mime='text/csv',
        )
        
elif not calculate_disabled:
    st.info("System Ready. Configure parameters and execute Analytics Engine.")
