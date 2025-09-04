"""
Classical Stock Screener - Lohusalu Capital Management
7-Step "Great Business" Framework Implementation
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Classical Screener - Lohusalu Capital Management",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #17becf);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .step-header {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

def get_sp500_tickers():
    """Get S&P 500 tickers from Wikipedia"""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        return sp500_table['Symbol'].tolist()
    except:
        # Fallback list of major tickers
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 
                'JNJ', 'V', 'WMT', 'JPM', 'MA', 'PG', 'UNH', 'DIS', 'HD', 'PYPL',
                'BAC', 'NFLX', 'ADBE', 'CRM', 'CMCSA', 'XOM', 'VZ', 'KO', 'ABT',
                'PFE', 'TMO', 'COST', 'AVGO', 'ACN', 'DHR', 'NEE', 'LIN', 'TXN']

def get_sector_tickers(sector):
    """Get tickers by sector (simplified implementation)"""
    sector_mapping = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'ADBE', 'CRM', 'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM', 'CSCO', 'AVGO'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'MDT', 'DHR', 'BMY', 'ABBV', 'MRK', 'LLY', 'GILD', 'AMGN', 'CVS', 'CI'],
        'Financial Services': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB', 'PNC', 'TFC', 'COF', 'CME', 'ICE'],
        'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX', 'LOW', 'TJX', 'F', 'GM', 'BKNG', 'MAR', 'RCL', 'CCL', 'NCLH'],
        'Communication Services': ['GOOGL', 'META', 'DIS', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'TWTR', 'SNAP', 'PINS', 'ROKU', 'SPOT', 'ZM'],
        'Industrial': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'NOC', 'GD', 'FDX', 'UNP', 'CSX', 'NSC', 'DAL'],
        'Consumer Defensive': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'WBA', 'CVS', 'TGT', 'KR', 'CL', 'GIS', 'K', 'CPB', 'SJM', 'HSY'],
        'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'BKR', 'HAL', 'DVN', 'FANG', 'APA', 'EQT'],
        'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'PEG', 'ED', 'ETR', 'WEC', 'ES', 'FE', 'AWK'],
        'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'WELL', 'DLR', 'O', 'SBAC', 'EXR', 'AVB', 'EQR', 'VTR', 'ESS', 'MAA'],
        'Materials': ['LIN', 'APD', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'PPG', 'ECL', 'IFF', 'ALB', 'CE', 'VMC', 'MLM', 'PKG']
    }
    return sector_mapping.get(sector, [])

def get_financial_data(ticker):
    """Get comprehensive financial data for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get basic info
        info = stock.info
        
        # Get financial statements
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        # Get historical data for ratios
        hist = stock.history(period="5y")
        
        return {
            'ticker': ticker,
            'info': info,
            'financials': financials,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow,
            'history': hist,
            'market_cap': info.get('marketCap', 0),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown')
        }
    except Exception as e:
        st.warning(f"Error fetching data for {ticker}: {str(e)}")
        return None

def calculate_growth_metrics(data):
    """Calculate growth metrics for the 7-step framework"""
    try:
        financials = data['financials']
        cash_flow = data['cash_flow']
        
        metrics = {}
        
        # 1. Revenue Growth (5Y)
        if 'Total Revenue' in financials.index:
            revenues = financials.loc['Total Revenue'].dropna()
            if len(revenues) >= 2:
                revenue_growth = ((revenues.iloc[0] / revenues.iloc[-1]) ** (1/len(revenues)) - 1) * 100
                metrics['revenue_growth_5y'] = revenue_growth
        
        # 2. Net Income Growth
        if 'Net Income' in financials.index:
            net_incomes = financials.loc['Net Income'].dropna()
            if len(net_incomes) >= 2:
                ni_growth = ((net_incomes.iloc[0] / net_incomes.iloc[-1]) ** (1/len(net_incomes)) - 1) * 100
                metrics['net_income_growth_5y'] = ni_growth
        
        # 3. Cash Flow Growth
        if 'Operating Cash Flow' in cash_flow.index:
            cash_flows = cash_flow.loc['Operating Cash Flow'].dropna()
            if len(cash_flows) >= 2:
                cf_growth = ((cash_flows.iloc[0] / cash_flows.iloc[-1]) ** (1/len(cash_flows)) - 1) * 100
                metrics['cash_flow_growth_5y'] = cf_growth
        
        return metrics
    except Exception as e:
        return {}

def calculate_profitability_metrics(data):
    """Calculate profitability metrics"""
    try:
        info = data['info']
        financials = data['financials']
        balance_sheet = data['balance_sheet']
        
        metrics = {}
        
        # ROE
        roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
        metrics['roe'] = roe
        
        # ROIC calculation
        if 'Net Income' in financials.index and 'Total Stockholder Equity' in balance_sheet.index:
            try:
                net_income = financials.loc['Net Income'].iloc[0]
                total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                if total_equity != 0:
                    roic = (net_income / total_equity) * 100
                    metrics['roic'] = roic
            except:
                pass
        
        # Profit Margins
        gross_margin = info.get('grossMargins', 0) * 100 if info.get('grossMargins') else 0
        profit_margin = info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0
        
        metrics['gross_margin'] = gross_margin
        metrics['profit_margin'] = profit_margin
        
        return metrics
    except Exception as e:
        return {}

def calculate_debt_metrics(data):
    """Calculate debt and financial strength metrics"""
    try:
        info = data['info']
        balance_sheet = data['balance_sheet']
        
        metrics = {}
        
        # Current Ratio
        current_ratio = info.get('currentRatio', 0)
        metrics['current_ratio'] = current_ratio
        
        # Debt to EBITDA
        debt_to_ebitda = info.get('debtToEbitda', 0)
        metrics['debt_to_ebitda'] = debt_to_ebitda
        
        # Quick Ratio
        quick_ratio = info.get('quickRatio', 0)
        metrics['quick_ratio'] = quick_ratio
        
        return metrics
    except Exception as e:
        return {}

def screen_stock(ticker, criteria):
    """Screen a single stock against the 7-step criteria"""
    data = get_financial_data(ticker)
    if not data:
        return None
    
    # Skip if market cap is too high
    market_cap = data.get('market_cap', 0)
    if market_cap > criteria['max_market_cap']:
        return None
    
    # Calculate all metrics
    growth_metrics = calculate_growth_metrics(data)
    profitability_metrics = calculate_profitability_metrics(data)
    debt_metrics = calculate_debt_metrics(data)
    
    # Combine all metrics
    all_metrics = {**growth_metrics, **profitability_metrics, **debt_metrics}
    
    # Score the stock
    score = 0
    max_score = 7
    
    # Step 1: Growth consistency (Revenue, Net Income, Cash Flow)
    if all_metrics.get('revenue_growth_5y', 0) > criteria['min_revenue_growth']:
        score += 1
    
    # Step 2: Positive Growth Rate
    if all_metrics.get('net_income_growth_5y', 0) > 0:
        score += 1
    
    # Step 3: Competitive Advantage (using profit margins as proxy)
    if all_metrics.get('gross_margin', 0) > criteria['min_gross_margin']:
        score += 1
    
    # Step 4: Profitability (ROE and ROIC)
    if all_metrics.get('roe', 0) > criteria['min_roe']:
        score += 1
    
    # Step 5: Conservative Debt
    if all_metrics.get('current_ratio', 0) > criteria['min_current_ratio']:
        score += 1
    
    if all_metrics.get('debt_to_ebitda', float('inf')) < criteria['max_debt_to_ebitda']:
        score += 1
    
    # Step 6: Operational Efficiency (using profit margin)
    if all_metrics.get('profit_margin', 0) > criteria['min_profit_margin']:
        score += 1
    
    return {
        'ticker': ticker,
        'score': score,
        'max_score': max_score,
        'percentage': (score / max_score) * 100,
        'market_cap': market_cap,
        'sector': data['info'].get('sector', 'Unknown'),
        'industry': data['info'].get('industry', 'Unknown'),
        **all_metrics
    }

def main():
    """Main function for Classical Screener"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 Classical Stock Screener</h1>
        <p>7-Step "Great Business" Framework - Quantitative Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.markdown("## 🔧 Screening Configuration")
    
    # Sector selection
    sectors = ['All Sectors', 'Technology', 'Healthcare', 'Financial Services', 
               'Consumer Cyclical', 'Communication Services', 'Industrial',
               'Consumer Defensive', 'Energy', 'Utilities', 'Real Estate', 'Materials']
    
    selected_sector = st.sidebar.selectbox(
        "Select Sector",
        options=sectors,
        index=0,
        help="Choose a specific sector or screen all sectors"
    )
    
    # Market cap filter
    st.sidebar.markdown("### Market Cap Filter")
    max_market_cap = st.sidebar.selectbox(
        "Maximum Market Cap",
        options=[300e6, 1e9, 2e9, 5e9, 10e9, 50e9, float('inf')],
        index=2,  # Default to $2B
        format_func=lambda x: f"${x/1e9:.1f}B" if x < float('inf') else "No Limit",
        help="Maximum market capitalization for screening"
    )
    
    # 7-Step Criteria Configuration
    st.sidebar.markdown("### 7-Step Criteria")
    
    with st.sidebar.expander("📈 Step 1: Growth Metrics", expanded=False):
        min_revenue_growth = st.slider(
            "Min Revenue Growth (5Y %)",
            min_value=0,
            max_value=50,
            value=10,
            help="Minimum 5-year revenue growth rate"
        )
    
    with st.sidebar.expander("💰 Step 2-4: Profitability", expanded=False):
        min_roe = st.slider(
            "Min ROE (%)",
            min_value=0,
            max_value=30,
            value=12,
            help="Minimum Return on Equity"
        )
        
        min_gross_margin = st.slider(
            "Min Gross Margin (%)",
            min_value=0,
            max_value=80,
            value=20,
            help="Minimum gross profit margin"
        )
        
        min_profit_margin = st.slider(
            "Min Profit Margin (%)",
            min_value=0,
            max_value=30,
            value=5,
            help="Minimum net profit margin"
        )
    
    with st.sidebar.expander("🛡️ Step 5: Debt Metrics", expanded=False):
        min_current_ratio = st.slider(
            "Min Current Ratio",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Minimum current ratio (current assets / current liabilities)"
        )
        
        max_debt_to_ebitda = st.slider(
            "Max Debt/EBITDA",
            min_value=0.0,
            max_value=5.0,
            value=3.0,
            step=0.1,
            help="Maximum debt to EBITDA ratio"
        )
    
    # Screening criteria dictionary
    criteria = {
        'max_market_cap': max_market_cap,
        'min_revenue_growth': min_revenue_growth,
        'min_roe': min_roe,
        'min_gross_margin': min_gross_margin,
        'min_profit_margin': min_profit_margin,
        'min_current_ratio': min_current_ratio,
        'max_debt_to_ebitda': max_debt_to_ebitda
    }
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🎯 7-Step Great Business Framework")
        
        # Display the 7 steps
        steps = [
            "📈 Consistently Increasing Sales, Net Income and Cash Flow",
            "📊 Positive Growth Rate",
            "🏰 Sustainable Competitive Advantage (Economic Moat)",
            "💰 Profitable and Operationally Efficient",
            "🛡️ Conservative Debt Structure",
            "⚖️ Business Maturity and Sector Position",
            "🎯 Risk-Adjusted Target Pricing"
        ]
        
        for i, step in enumerate(steps, 1):
            st.markdown(f"""
            <div class="step-header">
                <strong>Step {i}:</strong> {step}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("## 📊 Current Criteria")
        
        st.markdown(f"""
        <div class="metric-card">
            <strong>Sector:</strong> {selected_sector}<br>
            <strong>Max Market Cap:</strong> ${max_market_cap/1e9:.1f}B<br>
            <strong>Min Revenue Growth:</strong> {min_revenue_growth}%<br>
            <strong>Min ROE:</strong> {min_roe}%<br>
            <strong>Min Current Ratio:</strong> {min_current_ratio}<br>
            <strong>Max Debt/EBITDA:</strong> {max_debt_to_ebitda}
        </div>
        """, unsafe_allow_html=True)
    
    # Start screening button
    if st.button("🚀 Start Classical Screening", type="primary", use_container_width=True):
        
        # Get tickers to screen
        if selected_sector == 'All Sectors':
            tickers = get_sp500_tickers()
            st.info(f"Screening {len(tickers)} stocks from S&P 500...")
        else:
            tickers = get_sector_tickers(selected_sector)
            st.info(f"Screening {len(tickers)} stocks from {selected_sector} sector...")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Screen stocks
        results = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all screening tasks
            future_to_ticker = {
                executor.submit(screen_stock, ticker, criteria): ticker 
                for ticker in tickers
            }
            
            completed = 0
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    st.warning(f"Error screening {ticker}: {str(e)}")
                
                completed += 1
                progress = completed / len(tickers)
                progress_bar.progress(progress)
                status_text.text(f"Screened {completed}/{len(tickers)} stocks...")
        
        # Display results
        if results:
            st.success(f"✅ Screening complete! Found {len(results)} qualifying stocks.")
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            df = df.sort_values('score', ascending=False)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Stocks Screened", len(tickers))
            
            with col2:
                st.metric("Qualifying Stocks", len(results))
            
            with col3:
                avg_score = df['score'].mean()
                st.metric("Average Score", f"{avg_score:.1f}/7")
            
            with col4:
                top_score = df['score'].max()
                st.metric("Highest Score", f"{int(top_score)}/7")
            
            # Results visualization
            st.markdown("## 📊 Screening Results")
            
            # Score distribution
            fig_hist = px.histogram(
                df, 
                x='score', 
                nbins=8,
                title="Score Distribution",
                labels={'score': 'Score (out of 7)', 'count': 'Number of Stocks'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Top performers table
            st.markdown("### 🏆 Top Performing Stocks")
            
            display_cols = ['ticker', 'score', 'percentage', 'sector', 'market_cap', 
                          'revenue_growth_5y', 'roe', 'current_ratio']
            
            # Format the display DataFrame
            display_df = df[display_cols].copy()
            display_df['market_cap'] = display_df['market_cap'].apply(
                lambda x: f"${x/1e9:.2f}B" if x > 1e9 else f"${x/1e6:.0f}M"
            )
            display_df['percentage'] = display_df['percentage'].apply(lambda x: f"{x:.1f}%")
            
            # Rename columns for display
            display_df.columns = ['Ticker', 'Score', 'Score %', 'Sector', 'Market Cap', 
                                'Revenue Growth 5Y', 'ROE', 'Current Ratio']
            
            st.dataframe(
                display_df.head(20),
                use_container_width=True,
                hide_index=True
            )
            
            # Sector breakdown
            if len(df['sector'].unique()) > 1:
                st.markdown("### 🏢 Sector Breakdown")
                
                sector_summary = df.groupby('sector').agg({
                    'ticker': 'count',
                    'score': 'mean',
                    'market_cap': 'mean'
                }).round(2)
                
                sector_summary.columns = ['Count', 'Avg Score', 'Avg Market Cap']
                sector_summary['Avg Market Cap'] = sector_summary['Avg Market Cap'].apply(
                    lambda x: f"${x/1e9:.2f}B" if x > 1e9 else f"${x/1e6:.0f}M"
                )
                
                st.dataframe(sector_summary, use_container_width=True)
        
        else:
            st.warning("⚠️ No stocks met the screening criteria. Try adjusting the parameters.")

if __name__ == "__main__":
    main()

