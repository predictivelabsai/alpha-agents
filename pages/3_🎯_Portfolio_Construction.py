"""
Portfolio Construction Page - Multi-agent portfolio construction with enhanced features
"""

import streamlit as st
import sys
import os
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import json
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents import Stock, create_multi_agent_portfolio_system, RiskTolerance, InvestmentDecision
from agents.ranking_agent import RankingAgent
from utils.yfinance_data import yfinance_provider
from database import DatabaseManager

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Portfolio Construction - Lohusalu Capital Management",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .portfolio-stock {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
        border-left: 3px solid #1f77b4;
    }
    .recommendation-buy {
        color: #28a745;
        font-weight: bold;
    }
    .recommendation-sell {
        color: #dc3545;
        font-weight: bold;
    }
    .recommendation-hold {
        color: #ffc107;
        font-weight: bold;
    }
    .recommendation-avoid {
        color: #6c757d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def init_database():
    return DatabaseManager()

@st.cache_resource
def init_multi_agent_system():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("OpenAI API key not found. Please check your environment configuration.")
        return None
    
    return create_multi_agent_portfolio_system(
        openai_api_key=api_key,
        risk_tolerance="moderate",
        max_debate_rounds=2
    )

def main():
    """Portfolio Builder page main function"""
    
    # Initialize components
    db = init_database()
    mas = init_multi_agent_sy    # Page header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #28a745, #20c997); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1>🎯 Portfolio Construction</h1>
        <p>Build optimized portfolios using multi-agent analysis and advanced portfolio theory</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Build optimized portfolios using the multi-agent system. Add stocks to your portfolio 
    and let the agents collaborate to provide investment recommendations and risk assessments.
    """)
    
    # Sidebar settings
    st.sidebar.subheader("⚙️ Portfolio Settings")
    risk_tolerance = st.sidebar.selectbox(
        "Risk Tolerance:",
        ["Conservative", "Moderate", "Aggressive"],
        index=1
    )
    
    portfolio_size = st.sidebar.slider(
        "Target Portfolio Size",
        min_value=3,
        max_value=20,
        value=8,
        help="Number of stocks in the portfolio"
    )
    
    max_sector_weight = st.sidebar.slider(
        "Max Sector Weight (%)",
        min_value=10,
        max_value=50,
        value=30,
        help="Maximum weight for any single sector"
    )
    
    # Portfolio construction
    st.subheader("📋 Portfolio Construction")
    
    # Initialize session state for portfolio
    if 'portfolio_stocks' not in st.session_state:
        st.session_state.portfolio_stocks = []
    
    # Add stock form
    with st.expander("➕ Add Stock to Portfolio", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_symbol = st.text_input("Stock Symbol", key="new_symbol").upper()
        
        with col2:
            new_company = st.text_input("Company Name", key="new_company")
        
        with col3:
            new_sector = st.selectbox(
                "Sector",
                ["Technology", "Healthcare", "Financial", "Consumer Discretionary", 
                 "Consumer Staples", "Energy", "Utilities", "Industrials", 
                 "Materials", "Real Estate", "Communication Services"],
                key="new_sector"
            )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_price = st.number_input("Current Price ($)", min_value=0.01, value=100.0, key="new_price")
        
        with col2:
            new_market_cap = st.number_input("Market Cap (B$)", min_value=0.1, value=50.0, key="new_market_cap") * 1e9
        
        with col3:
            if st.button("➕ Add to Portfolio"):
                if new_symbol and new_company:
                    # Check if stock already exists
                    if not any(stock.symbol == new_symbol for stock in st.session_state.portfolio_stocks):
                        new_stock = Stock(
                            symbol=new_symbol,
                            company_name=new_company,
                            sector=new_sector,
                            current_price=new_price,
                            market_cap=new_market_cap,
                            pe_ratio=25.0,  # Default values
                            dividend_yield=1.0,
                            beta=1.0,
                            volume=1000000
                        )
                        st.session_state.portfolio_stocks.append(new_stock)
                        st.success(f"✅ Added {new_symbol} to portfolio")
                        st.rerun()
                    else:
                        st.warning(f"⚠️ {new_symbol} is already in the portfolio")
                else:
                    st.error("Please provide both symbol and company name")
    
    # Current portfolio
    if st.session_state.portfolio_stocks:
        st.subheader("📊 Current Portfolio")
        
        # Portfolio overview
        portfolio_df = pd.DataFrame([
            {
                'Symbol': stock.symbol,
                'Company': stock.company_name,
                'Sector': stock.sector,
                'Price': f"${stock.current_price:.2f}",
                'Market Cap': f"${stock.market_cap/1e9:.1f}B"
            }
            for stock in st.session_state.portfolio_stocks
        ])
        
        st.dataframe(portfolio_df, use_container_width=True)
        
        # Portfolio actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Analyze Portfolio", type="primary"):
                if mas is not None:
                    analyze_portfolio(st.session_state.portfolio_stocks, mas, risk_tolerance)
                else:
                    st.error("Multi-agent system not available")
        
        with col2:
            if st.button("🗑️ Clear Portfolio"):
                st.session_state.portfolio_stocks = []
                st.rerun()
        
        with col3:
            # Export portfolio
            portfolio_csv = portfolio_df.to_csv(index=False)
            st.download_button(
                label="📥 Export Portfolio",
                data=portfolio_csv,
                file_name=f"alpha_agents_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Sector allocation
        st.subheader("🏭 Sector Allocation")
        sector_counts = pd.Series([stock.sector for stock in st.session_state.portfolio_stocks]).value_counts()
        
        fig = px.pie(
            values=sector_counts.values,
            names=sector_counts.index,
            title="Portfolio Sector Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("📝 No stocks in portfolio. Add some stocks to get started!")
        
        # Quick add popular stocks
        st.subheader("🚀 Quick Add Popular Stocks")
        
        popular_stocks = [
            ("AAPL", "Apple Inc.", "Technology", 150.0, 2500e9),
            ("MSFT", "Microsoft Corporation", "Technology", 300.0, 2200e9),
            ("GOOGL", "Alphabet Inc.", "Communication Services", 120.0, 1500e9),
            ("AMZN", "Amazon.com Inc.", "Consumer Discretionary", 130.0, 1350e9),
            ("TSLA", "Tesla Inc.", "Consumer Discretionary", 200.0, 650e9),
            ("NVDA", "NVIDIA Corporation", "Technology", 450.0, 1100e9),
            ("JPM", "JPMorgan Chase & Co.", "Financial", 140.0, 420e9),
            ("JNJ", "Johnson & Johnson", "Healthcare", 160.0, 430e9)
        ]
        
        cols = st.columns(4)
        for i, (symbol, company, sector, price, market_cap) in enumerate(popular_stocks):
            with cols[i % 4]:
                if st.button(f"➕ {symbol}", key=f"quick_add_{symbol}"):
                    stock = Stock(
                        symbol=symbol,
                        company_name=company,
                        sector=sector,
                        current_price=price,
                        market_cap=market_cap,
                        pe_ratio=25.0,
                        dividend_yield=1.0,
                        beta=1.0,
                        volume=1000000
                    )
                    st.session_state.portfolio_stocks.append(stock)
                    st.success(f"✅ Added {symbol}")
                    st.rerun()

def analyze_portfolio(stocks, mas, risk_tolerance):
    """Analyze the portfolio using the multi-agent system"""
    
    st.markdown("---")
    st.subheader("🤖 Multi-Agent Portfolio Analysis")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Analyze each stock
    all_analyses = {}
    total_stocks = len(stocks)
    
    for i, stock in enumerate(stocks):
        status_text.text(f"Analyzing {stock.symbol}...")
        progress_bar.progress((i + 1) / total_stocks)
        
        stock_analyses = {}
        for agent_name, agent in mas.agents.items():
            try:
                analysis = agent.analyze(stock)
                stock_analyses[agent_name] = analysis
            except Exception as e:
                st.warning(f"Error in {agent_name} analysis for {stock.symbol}: {e}")
        
        all_analyses[stock.symbol] = stock_analyses
    
    status_text.text("Analysis complete!")
    progress_bar.progress(1.0)
    
    # Portfolio summary
    st.subheader("📊 Portfolio Summary")
    
    # Calculate portfolio metrics
    portfolio_recommendations = []
    portfolio_confidence = []
    portfolio_risk = []
    
    for stock_symbol, analyses in all_analyses.items():
        if analyses:
            recommendations = [a.recommendation.value for a in analyses.values()]
            confidences = [a.confidence_score for a in analyses.values()]
            risks = [a.risk_assessment for a in analyses.values()]
            
            # Most common recommendation
            most_common_rec = max(set(recommendations), key=recommendations.count)
            avg_confidence = sum(confidences) / len(confidences)
            most_common_risk = max(set(risks), key=risks.count)
            
            portfolio_recommendations.append(most_common_rec)
            portfolio_confidence.append(avg_confidence)
            portfolio_risk.append(most_common_risk)
    
    # Portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        buy_count = portfolio_recommendations.count('buy')
        st.metric("Buy Recommendations", f"{buy_count}/{len(stocks)}")
    
    with col2:
        avg_confidence = sum(portfolio_confidence) / len(portfolio_confidence) if portfolio_confidence else 0
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    with col3:
        high_risk_count = portfolio_risk.count('HIGH')
        st.metric("High Risk Stocks", f"{high_risk_count}/{len(stocks)}")
    
    with col4:
        sectors = [stock.sector for stock in stocks]
        unique_sectors = len(set(sectors))
        st.metric("Sector Diversity", f"{unique_sectors} sectors")
    
    # Individual stock results
    st.subheader("📈 Individual Stock Analysis")
    
    for stock in stocks:
        analyses = all_analyses.get(stock.symbol, {})
        if analyses:
            with st.expander(f"{stock.symbol} - {stock.company_name}", expanded=False):
                
                # Stock summary
                recommendations = [a.recommendation.value for a in analyses.values()]
                confidences = [a.confidence_score for a in analyses.values()]
                
                most_common_rec = max(set(recommendations), key=recommendations.count)
                avg_confidence = sum(confidences) / len(confidences)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Consensus:** {most_common_rec.upper()}")
                    st.write(f"**Average Confidence:** {avg_confidence:.2f}")
                    
                    # Agent breakdown
                    agent_results = []
                    for agent_name, analysis in analyses.items():
                        agent_results.append({
                            'Agent': agent_name.title(),
                            'Recommendation': analysis.recommendation.value,
                            'Confidence': f"{analysis.confidence_score:.2f}",
                            'Risk': analysis.risk_assessment
                        })
                    
                    st.dataframe(pd.DataFrame(agent_results), use_container_width=True)
                
                with col2:
                    # Quick metrics
                    st.metric("Current Price", f"${stock.current_price:.2f}")
                    st.metric("Market Cap", f"${stock.market_cap/1e9:.1f}B")
                    st.metric("Sector", stock.sector)
    
    # Portfolio recommendations
    st.subheader("💡 Portfolio Recommendations")
    
    buy_stocks = [stocks[i] for i, rec in enumerate(portfolio_recommendations) if rec == 'buy']
    hold_stocks = [stocks[i] for i, rec in enumerate(portfolio_recommendations) if rec == 'hold']
    avoid_stocks = [stocks[i] for i, rec in enumerate(portfolio_recommendations) if rec in ['sell', 'avoid']]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🟢 Strong Buy Candidates:**")
        for stock in buy_stocks:
            st.markdown(f'<div class="portfolio-stock">📈 {stock.symbol} - {stock.company_name}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("**🟡 Hold/Monitor:**")
        for stock in hold_stocks:
            st.markdown(f'<div class="portfolio-stock">📊 {stock.symbol} - {stock.company_name}</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown("**🔴 Consider Reducing:**")
        for stock in avoid_stocks:
            st.markdown(f'<div class="portfolio-stock">📉 {stock.symbol} - {stock.company_name}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

