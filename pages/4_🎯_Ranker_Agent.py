"""
Ranker Agent Page - Lohusalu Capital Management
Individual page for running and analyzing the Ranker Agent
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from datetime import datetime
import asyncio
import logging
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.ranker_agent_v2 import RankerAgent

# Page configuration
st.set_page_config(
    page_title="Ranker Agent - Lohusalu Capital Management",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #6f42c1, #e83e8c);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #6f42c1;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
        margin: 0.5rem 0;
    }
    .grade-a { color: #28a745; font-weight: bold; font-size: 1.2em; }
    .grade-b { color: #17a2b8; font-weight: bold; font-size: 1.2em; }
    .grade-c { color: #ffc107; font-weight: bold; font-size: 1.2em; }
    .grade-d { color: #dc3545; font-weight: bold; font-size: 1.2em; }
    .conviction-high { color: #28a745; font-weight: bold; }
    .conviction-medium { color: #ffc107; font-weight: bold; }
    .conviction-low { color: #dc3545; font-weight: bold; }
    .risk-low { color: #28a745; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-high { color: #dc3545; font-weight: bold; }
    .thesis-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #6f42c1;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_grade_class(grade):
    """Get CSS class based on investment grade"""
    if grade.startswith('A'):
        return "grade-a"
    elif grade.startswith('B'):
        return "grade-b"
    elif grade.startswith('C'):
        return "grade-c"
    else:
        return "grade-d"

def get_conviction_class(conviction):
    """Get CSS class based on conviction level"""
    if conviction == "High":
        return "conviction-high"
    elif conviction == "Medium":
        return "conviction-medium"
    else:
        return "conviction-low"

def get_risk_class(risk):
    """Get CSS class based on risk rating"""
    if risk == "Low":
        return "risk-low"
    elif risk == "Medium":
        return "risk-medium"
    else:
        return "risk-high"

def main():
    """Main function for Ranker Agent page"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Ranker Agent</h1>
        <p>Final Scoring & Investment Recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Agent Description
    st.markdown("""
    <div class="agent-card">
        <h3>🎯 Agent Capabilities</h3>
        <ul>
            <li><strong>Composite Scoring:</strong> Combines fundamental (60%) and qualitative (40%) analysis</li>
            <li><strong>Investment Grading:</strong> A+ to D grading system with detailed breakdowns</li>
            <li><strong>Multi-Dimensional Analysis:</strong> Growth, profitability, moat, sentiment, trends</li>
            <li><strong>Investment Thesis:</strong> Comprehensive analysis with strengths, risks, catalysts</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.markdown("## 🔧 Agent Configuration")
    
    # Model selection
    model_provider = st.sidebar.selectbox(
        "Select LLM Provider",
        options=["openai", "google", "anthropic", "mistral"],
        index=0,
        help="Choose the language model provider for analysis"
    )
    
    model_mapping = {
        "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
        "google": ["gemini-pro", "gemini-pro-vision"],
        "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
        "mistral": ["mistral-large", "mistral-medium", "mistral-small"]
    }
    
    model_name = st.sidebar.selectbox(
        "Select Model",
        options=model_mapping[model_provider],
        index=0,
        help="Choose the specific model for analysis"
    )
    
    # Analysis mode
    st.sidebar.markdown("### 📊 Analysis Mode")
    
    analysis_mode = st.sidebar.radio(
        "Select Analysis Mode",
        options=["Single Stock Analysis", "Portfolio Ranking", "Mock Data Demo"],
        index=0,
        help="Choose the type of analysis to perform"
    )
    
    if analysis_mode == "Single Stock Analysis":
        display_single_stock_interface(model_provider, model_name)
    elif analysis_mode == "Portfolio Ranking":
        display_portfolio_ranking_interface(model_provider, model_name)
    else:
        display_mock_demo_interface(model_provider, model_name)
    
    # Display recent traces
    display_recent_traces()

def display_single_stock_interface(model_provider, model_name):
    """Display interface for single stock analysis"""
    
    # Stock input
    st.sidebar.markdown("### 📈 Stock Input")
    
    ticker = st.sidebar.text_input(
        "Stock Ticker",
        value="AAPL",
        help="Enter the stock ticker symbol"
    ).upper()
    
    company_name = st.sidebar.text_input(
        "Company Name",
        value="Apple Inc.",
        help="Enter the full company name"
    )
    
    sector = st.sidebar.selectbox(
        "Sector",
        options=["Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
                "Communication Services", "Industrial", "Consumer Defensive", "Energy",
                "Utilities", "Real Estate", "Materials"],
        index=0,
        help="Select the company's sector"
    )
    
    # Mock data inputs for demonstration
    st.sidebar.markdown("### 📊 Mock Analysis Data")
    
    with st.sidebar.expander("Fundamental Data", expanded=False):
        fundamental_score = st.slider("Fundamental Score", 0, 100, 75)
        revenue_growth = st.slider("Revenue Growth (%)", -10, 50, 15)
        roe = st.slider("ROE (%)", 0, 40, 18)
        current_ratio = st.slider("Current Ratio", 0.5, 5.0, 2.1, 0.1)
        debt_to_equity = st.slider("Debt/Equity", 0.0, 2.0, 0.3, 0.1)
        pe_ratio = st.slider("P/E Ratio", 5, 50, 22)
        upside_potential = st.slider("Upside Potential (%)", -20, 100, 25)
        market_cap = st.selectbox("Market Cap", 
                                 options=[1e9, 5e9, 10e9, 50e9, 100e9],
                                 format_func=lambda x: f"${x/1e9:.0f}B")
    
    with st.sidebar.expander("Qualitative Data", expanded=False):
        qualitative_score = st.slider("Qualitative Score", 0, 100, 80)
        moat_strength = st.selectbox("Moat Strength", ["Wide", "Narrow", "None"])
        moat_score = st.slider("Moat Score", 0, 100, 85)
        sentiment = st.selectbox("Sentiment", ["Positive", "Neutral", "Negative"])
        sentiment_score = st.slider("Sentiment Score", 0, 100, 75)
        trend_alignment = st.selectbox("Trend Alignment", ["Strong", "Moderate", "Weak"])
        trends_score = st.slider("Trends Score", 0, 100, 80)
        market_position = st.selectbox("Market Position", ["Leader", "Strong", "Moderate", "Weak"])
        competitive_score = st.slider("Competitive Score", 0, 100, 85)
    
    save_trace = st.sidebar.checkbox("Save Analysis Trace", value=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🚀 Run Investment Scoring")
        
        if st.button("🎯 Score Investment", type="primary", use_container_width=True):
            if ticker and company_name:
                run_single_stock_analysis(
                    ticker, company_name, sector, model_provider, model_name,
                    fundamental_score, revenue_growth, roe, current_ratio, debt_to_equity,
                    pe_ratio, upside_potential, market_cap, qualitative_score,
                    moat_strength, moat_score, sentiment, sentiment_score,
                    trend_alignment, trends_score, market_position, competitive_score,
                    save_trace
                )
            else:
                st.error("Please enter both ticker and company name")
    
    with col2:
        st.markdown("## 📋 Configuration")
        st.markdown(f"""
        - **Model:** {model_provider.title()} - {model_name}
        - **Stock:** {ticker} ({company_name})
        - **Sector:** {sector}
        - **Fundamental Score:** {fundamental_score}/100
        - **Qualitative Score:** {qualitative_score}/100
        """)

def display_portfolio_ranking_interface(model_provider, model_name):
    """Display interface for portfolio ranking"""
    
    st.sidebar.markdown("### 📊 Portfolio Parameters")
    
    max_positions = st.sidebar.slider(
        "Maximum Positions",
        min_value=5,
        max_value=20,
        value=10,
        help="Maximum number of stocks in portfolio"
    )
    
    save_trace = st.sidebar.checkbox("Save Analysis Trace", value=True)
    
    # Main content
    st.markdown("## 📊 Portfolio Ranking & Construction")
    
    if st.button("🏗️ Build Portfolio", type="primary", use_container_width=True):
        run_portfolio_analysis(model_provider, model_name, max_positions, save_trace)

def display_mock_demo_interface(model_provider, model_name):
    """Display interface for mock data demonstration"""
    
    st.sidebar.markdown("### 🎭 Demo Parameters")
    
    num_stocks = st.sidebar.slider(
        "Number of Stocks",
        min_value=5,
        max_value=20,
        value=10,
        help="Number of mock stocks to analyze"
    )
    
    save_trace = st.sidebar.checkbox("Save Analysis Trace", value=True)
    
    # Main content
    st.markdown("## 🎭 Mock Data Demonstration")
    
    if st.button("🎲 Run Demo Analysis", type="primary", use_container_width=True):
        run_mock_demo(model_provider, model_name, num_stocks, save_trace)

def run_single_stock_analysis(ticker, company_name, sector, model_provider, model_name,
                             fundamental_score, revenue_growth, roe, current_ratio, 
                             debt_to_equity, pe_ratio, upside_potential, market_cap,
                             qualitative_score, moat_strength, moat_score, sentiment,
                             sentiment_score, trend_alignment, trends_score,
                             market_position, competitive_score, save_trace):
    """Run single stock analysis"""
    
    try:
        # Initialize agent
        with st.spinner("Initializing Ranker Agent..."):
            agent = RankerAgent(model_provider=model_provider, model_name=model_name)
        
        # Prepare mock data
        fundamental_data = {
            'fundamental_score': fundamental_score,
            'metrics': {
                'revenue_growth': revenue_growth,
                'roe': roe,
                'current_ratio': current_ratio,
                'debt_to_equity': debt_to_equity,
                'pe_ratio': pe_ratio,
                'profit_margin': roe * 0.5,  # Approximate
                'gross_margin': roe * 0.8,   # Approximate
                'roic': roe * 0.9            # Approximate
            },
            'upside_potential': upside_potential,
            'market_cap': market_cap
        }
        
        qualitative_data = {
            'qualitative_score': qualitative_score,
            'moat_analysis': {
                'moat_strength': moat_strength,
                'moat_score': moat_score
            },
            'sentiment_analysis': {
                'overall_sentiment': sentiment,
                'sentiment_score': sentiment_score
            },
            'secular_trends': {
                'trend_alignment': trend_alignment,
                'trend_score': trends_score
            },
            'competitive_position': {
                'market_position': market_position,
                'competitive_score': competitive_score
            }
        }
        
        # Run scoring
        with st.spinner("Scoring investment..."):
            investment_score = agent.score_investment(
                ticker, company_name, sector, fundamental_data, qualitative_data
            )
        
        # Display results
        display_investment_score(investment_score)
        
        # Save trace if requested
        if save_trace:
            with st.spinner("Saving analysis trace..."):
                trace_file = agent.save_analysis_trace([investment_score], None)
                if trace_file:
                    st.success(f"✅ Analysis trace saved to: {trace_file}")
        
        st.success("🎉 Investment scoring completed successfully!")
        
    except Exception as e:
        st.error(f"❌ Error in investment scoring: {str(e)}")
        st.exception(e)

def run_portfolio_analysis(model_provider, model_name, max_positions, save_trace):
    """Run portfolio analysis with mock data"""
    
    try:
        # Initialize agent
        with st.spinner("Initializing Ranker Agent..."):
            agent = RankerAgent(model_provider=model_provider, model_name=model_name)
        
        # Generate mock investment scores
        with st.spinner("Generating mock portfolio data..."):
            mock_stocks = generate_mock_portfolio_data(max_positions * 2)  # Generate more than needed
            
            investment_scores = []
            for stock_data in mock_stocks:
                score = agent.score_investment(
                    stock_data['ticker'],
                    stock_data['company_name'],
                    stock_data['sector'],
                    stock_data['fundamental_data'],
                    stock_data['qualitative_data']
                )
                investment_scores.append(score)
        
        # Rank investments
        with st.spinner("Ranking investments..."):
            ranked_scores = agent.rank_investments(investment_scores)
        
        # Create portfolio recommendation
        with st.spinner("Creating portfolio recommendation..."):
            portfolio_recommendation = agent.create_portfolio_recommendation(
                ranked_scores, max_positions
            )
        
        # Display results
        display_portfolio_results(ranked_scores, portfolio_recommendation)
        
        # Save trace if requested
        if save_trace:
            with st.spinner("Saving analysis trace..."):
                trace_file = agent.save_analysis_trace(ranked_scores, portfolio_recommendation)
                if trace_file:
                    st.success(f"✅ Analysis trace saved to: {trace_file}")
        
        st.success("🎉 Portfolio analysis completed successfully!")
        
    except Exception as e:
        st.error(f"❌ Error in portfolio analysis: {str(e)}")
        st.exception(e)

def run_mock_demo(model_provider, model_name, num_stocks, save_trace):
    """Run mock demonstration"""
    
    try:
        # Initialize agent
        with st.spinner("Initializing Ranker Agent..."):
            agent = RankerAgent(model_provider=model_provider, model_name=model_name)
        
        # Generate mock data
        with st.spinner("Generating mock analysis data..."):
            mock_stocks = generate_mock_portfolio_data(num_stocks)
            
            investment_scores = []
            for stock_data in mock_stocks:
                score = agent.score_investment(
                    stock_data['ticker'],
                    stock_data['company_name'],
                    stock_data['sector'],
                    stock_data['fundamental_data'],
                    stock_data['qualitative_data']
                )
                investment_scores.append(score)
        
        # Display demo results
        display_demo_results(investment_scores)
        
        st.success("🎉 Mock demonstration completed successfully!")
        
    except Exception as e:
        st.error(f"❌ Error in mock demonstration: {str(e)}")
        st.exception(e)

def generate_mock_portfolio_data(num_stocks):
    """Generate mock portfolio data for demonstration"""
    
    mock_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
        "CRM", "ADBE", "PYPL", "ZOOM", "SHOP", "SQ", "ROKU", "TWLO",
        "DDOG", "SNOW", "PLTR", "CRWD", "ZS", "OKTA", "NET", "FTNT"
    ]
    
    sectors = ["Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
               "Communication Services", "Industrial", "Consumer Defensive"]
    
    mock_data = []
    
    for i in range(min(num_stocks, len(mock_tickers))):
        ticker = mock_tickers[i]
        
        # Generate random but realistic data
        fundamental_score = np.random.normal(65, 15)
        fundamental_score = max(20, min(95, fundamental_score))
        
        qualitative_score = np.random.normal(70, 12)
        qualitative_score = max(25, min(95, qualitative_score))
        
        mock_data.append({
            'ticker': ticker,
            'company_name': f"{ticker} Corporation",
            'sector': np.random.choice(sectors),
            'fundamental_data': {
                'fundamental_score': fundamental_score,
                'metrics': {
                    'revenue_growth': np.random.normal(12, 8),
                    'roe': np.random.normal(15, 5),
                    'current_ratio': np.random.normal(2.0, 0.5),
                    'debt_to_equity': np.random.normal(0.4, 0.2),
                    'pe_ratio': np.random.normal(25, 8),
                    'profit_margin': np.random.normal(8, 3),
                    'gross_margin': np.random.normal(35, 10),
                    'roic': np.random.normal(12, 4)
                },
                'upside_potential': np.random.normal(20, 15),
                'market_cap': np.random.choice([1e9, 5e9, 10e9, 50e9])
            },
            'qualitative_data': {
                'qualitative_score': qualitative_score,
                'moat_analysis': {
                    'moat_strength': np.random.choice(["Wide", "Narrow", "None"], p=[0.3, 0.5, 0.2]),
                    'moat_score': np.random.normal(70, 15)
                },
                'sentiment_analysis': {
                    'overall_sentiment': np.random.choice(["Positive", "Neutral", "Negative"], p=[0.5, 0.3, 0.2]),
                    'sentiment_score': np.random.normal(65, 12)
                },
                'secular_trends': {
                    'trend_alignment': np.random.choice(["Strong", "Moderate", "Weak"], p=[0.4, 0.4, 0.2]),
                    'trend_score': np.random.normal(68, 10)
                },
                'competitive_position': {
                    'market_position': np.random.choice(["Leader", "Strong", "Moderate", "Weak"], p=[0.3, 0.4, 0.2, 0.1]),
                    'competitive_score': np.random.normal(72, 12)
                }
            }
        })
    
    return mock_data

def display_investment_score(investment_score):
    """Display single investment score results"""
    
    st.markdown("## 🎯 Investment Score Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        grade_class = get_grade_class(investment_score.investment_grade)
        st.markdown(f"""
        <div class="metric-card">
            <h4>Investment Grade</h4>
            <p class="{grade_class}">{investment_score.investment_grade}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric(
            "Composite Score",
            f"{investment_score.composite_score:.1f}/100",
            help="Overall investment score"
        )
    
    with col3:
        conviction_class = get_conviction_class(investment_score.conviction_level)
        st.markdown(f"""
        <div class="metric-card">
            <h4>Conviction Level</h4>
            <p class="{conviction_class}">{investment_score.conviction_level}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        risk_class = get_risk_class(investment_score.risk_rating)
        st.markdown(f"""
        <div class="metric-card">
            <h4>Risk Rating</h4>
            <p class="{risk_class}">{investment_score.risk_rating}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Score breakdown
    st.markdown("### 📊 Score Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fundamental vs Qualitative
        fig = go.Figure(data=[
            go.Bar(name='Fundamental', x=['Score'], y=[investment_score.fundamental_score]),
            go.Bar(name='Qualitative', x=['Score'], y=[investment_score.qualitative_score])
        ])
        fig.update_layout(
            title="Fundamental vs Qualitative Scores",
            yaxis_title="Score",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Component scores radar chart
        categories = ['Growth', 'Profitability', 'Financial Strength', 'Valuation', 
                     'Moat', 'Sentiment', 'Trends', 'Competitive']
        scores = [
            investment_score.growth_score,
            investment_score.profitability_score,
            investment_score.financial_strength_score,
            investment_score.valuation_score,
            investment_score.moat_score,
            investment_score.sentiment_score,
            investment_score.trends_score,
            investment_score.competitive_score
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name='Component Scores'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            title="Component Score Breakdown",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Investment thesis
    st.markdown("### 📝 Investment Thesis")
    st.markdown(f"""
    <div class="thesis-box">
        {investment_score.investment_thesis}
    </div>
    """, unsafe_allow_html=True)
    
    # Key factors
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 💪 Key Strengths")
        for i, strength in enumerate(investment_score.key_strengths, 1):
            st.markdown(f"{i}. {strength}")
    
    with col2:
        st.markdown("### ⚠️ Key Risks")
        for i, risk in enumerate(investment_score.key_risks, 1):
            st.markdown(f"{i}. {risk}")
    
    with col3:
        st.markdown("### 🚀 Catalysts")
        for i, catalyst in enumerate(investment_score.catalysts, 1):
            st.markdown(f"{i}. {catalyst}")
    
    # Additional metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Financial Metrics")
        st.markdown(f"- **Upside Potential:** {investment_score.upside_potential:.1f}%")
        st.markdown(f"- **Time Horizon:** {investment_score.time_horizon}")
        st.markdown(f"- **Sector:** {investment_score.sector}")
    
    with col2:
        st.markdown("### 🔍 Scoring Reasoning")
        with st.expander("View detailed reasoning", expanded=False):
            st.markdown(investment_score.reasoning)

def display_portfolio_results(ranked_scores, portfolio_recommendation):
    """Display portfolio analysis results"""
    
    st.markdown("## 🏗️ Portfolio Construction Results")
    
    # Portfolio summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Portfolio Size",
            len(portfolio_recommendation.recommended_stocks),
            help="Number of positions in portfolio"
        )
    
    with col2:
        st.metric(
            "Expected Return",
            f"{portfolio_recommendation.expected_return:.1f}%",
            help="Average upside potential"
        )
    
    with col3:
        st.metric(
            "Risk Profile",
            portfolio_recommendation.risk_profile,
            help="Overall portfolio risk assessment"
        )
    
    with col4:
        st.metric(
            "Diversification",
            f"{portfolio_recommendation.diversification_score:.0f}/100",
            help="Portfolio diversification score"
        )
    
    # Top holdings
    st.markdown("### 🏆 Portfolio Holdings")
    
    holdings_data = []
    for i, stock in enumerate(portfolio_recommendation.recommended_stocks, 1):
        holdings_data.append({
            'Rank': i,
            'Ticker': stock.ticker,
            'Company': stock.company_name,
            'Sector': stock.sector,
            'Grade': stock.investment_grade,
            'Score': f"{stock.composite_score:.1f}",
            'Upside': f"{stock.upside_potential:.1f}%",
            'Risk': stock.risk_rating,
            'Conviction': stock.conviction_level
        })
    
    holdings_df = pd.DataFrame(holdings_data)
    st.dataframe(holdings_df, use_container_width=True, hide_index=True)
    
    # Sector allocation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏢 Sector Allocation")
        
        sector_df = pd.DataFrame([
            {'Sector': sector, 'Allocation': allocation}
            for sector, allocation in portfolio_recommendation.portfolio_composition.items()
        ])
        
        fig = px.pie(
            sector_df,
            values='Allocation',
            names='Sector',
            title="Portfolio Sector Allocation"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Score Distribution")
        
        scores = [stock.composite_score for stock in portfolio_recommendation.recommended_stocks]
        
        fig = px.histogram(
            x=scores,
            nbins=10,
            title="Portfolio Score Distribution"
        )
        fig.update_xaxis(title="Composite Score")
        fig.update_yaxis(title="Number of Stocks")
        st.plotly_chart(fig, use_container_width=True)
    
    # Portfolio thesis
    st.markdown("### 📝 Portfolio Investment Thesis")
    st.markdown(f"""
    <div class="thesis-box">
        {portfolio_recommendation.portfolio_thesis}
    </div>
    """, unsafe_allow_html=True)

def display_demo_results(investment_scores):
    """Display mock demonstration results"""
    
    st.markdown("## 🎭 Mock Analysis Results")
    
    # Summary statistics
    scores = [score.composite_score for score in investment_scores]
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Score", f"{avg_score:.1f}")
    
    with col2:
        st.metric("Highest Score", f"{max_score:.1f}")
    
    with col3:
        st.metric("Lowest Score", f"{min_score:.1f}")
    
    with col4:
        st.metric("Stocks Analyzed", len(investment_scores))
    
    # Results table
    st.markdown("### 📊 Analysis Results")
    
    results_data = []
    for i, score in enumerate(sorted(investment_scores, key=lambda x: x.composite_score, reverse=True), 1):
        results_data.append({
            'Rank': i,
            'Ticker': score.ticker,
            'Sector': score.sector,
            'Grade': score.investment_grade,
            'Composite Score': f"{score.composite_score:.1f}",
            'Fundamental': f"{score.fundamental_score:.1f}",
            'Qualitative': f"{score.qualitative_score:.1f}",
            'Risk': score.risk_rating,
            'Conviction': score.conviction_level
        })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Score distribution
        fig = px.histogram(
            x=scores,
            nbins=8,
            title="Score Distribution"
        )
        fig.update_xaxis(title="Composite Score")
        fig.update_yaxis(title="Number of Stocks")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Fundamental vs Qualitative scatter
        fundamental_scores = [score.fundamental_score for score in investment_scores]
        qualitative_scores = [score.qualitative_score for score in investment_scores]
        tickers = [score.ticker for score in investment_scores]
        
        fig = px.scatter(
            x=fundamental_scores,
            y=qualitative_scores,
            hover_name=tickers,
            title="Fundamental vs Qualitative Scores"
        )
        fig.update_xaxis(title="Fundamental Score")
        fig.update_yaxis(title="Qualitative Score")
        st.plotly_chart(fig, use_container_width=True)

def display_recent_traces():
    """Display recent analysis traces"""
    
    st.markdown("## 📁 Recent Analysis Traces")
    
    trace_dir = "tracing"
    if not os.path.exists(trace_dir):
        st.info("No analysis traces found. Run an analysis to generate traces.")
        return
    
    # Get recent trace files
    trace_files = [
        f for f in os.listdir(trace_dir) 
        if f.startswith("ranker_agent_trace_") and f.endswith(".json")
    ]
    
    if not trace_files:
        st.info("No ranker agent traces found.")
        return
    
    # Sort by modification time (most recent first)
    trace_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(trace_dir, x)),
        reverse=True
    )
    
    # Display recent traces
    for trace_file in trace_files[:5]:  # Show last 5 traces
        trace_path = os.path.join(trace_dir, trace_file)
        
        try:
            with open(trace_path, 'r') as f:
                trace_data = json.load(f)
            
            timestamp = trace_data.get('timestamp', 'Unknown')
            model_info = f"{trace_data.get('model_provider', 'Unknown')} - {trace_data.get('model_name', 'Unknown')}"
            investment_scores = trace_data.get('investment_scores', [])
            
            with st.expander(f"📄 {trace_file} - {timestamp}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Analysis Summary:**")
                    st.markdown(f"- Model: {model_info}")
                    st.markdown(f"- Investments Scored: {len(investment_scores)}")
                    if investment_scores:
                        avg_score = np.mean([score.get('composite_score', 0) for score in investment_scores])
                        st.markdown(f"- Average Score: {avg_score:.1f}/100")
                
                with col2:
                    st.markdown("**Actions:**")
                    if st.button(f"📥 Download {trace_file}", key=f"download_{trace_file}"):
                        with open(trace_path, 'r') as f:
                            st.download_button(
                                label="Download JSON",
                                data=f.read(),
                                file_name=trace_file,
                                mime="application/json"
                            )
                
                # Show trace content
                if st.checkbox(f"Show trace content", key=f"show_{trace_file}"):
                    st.json(trace_data)
        
        except Exception as e:
            st.error(f"Error loading trace {trace_file}: {str(e)}")

if __name__ == "__main__":
    main()

