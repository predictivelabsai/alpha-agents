"""
Stock Analysis Page - Individual stock analysis using multi-agent system
"""

import streamlit as st
import sys
import os
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents import Stock, create_multi_agent_portfolio_system, RiskTolerance, InvestmentDecision
from database import DatabaseManager

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Stock Analysis - Alpha Agents",
    page_icon="📊",
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
    .agent-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
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
    """Stock Analysis page main function"""
    
    # Initialize components
    db = init_database()
    mas = init_multi_agent_system()
    
    # Page header
    st.markdown('<h1 class="main-header">📊 Individual Stock Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Analyze individual stocks using the multi-agent system. Each agent provides specialized 
    analysis from their domain expertise, and the system aggregates insights for investment decisions.
    """)
    
    # Sidebar settings
    st.sidebar.subheader("⚙️ Analysis Settings")
    risk_tolerance = st.sidebar.selectbox(
        "Risk Tolerance:",
        ["Conservative", "Moderate", "Aggressive"],
        index=1
    )
    
    # Stock input form
    st.subheader("🔍 Stock Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stock_symbol = st.text_input(
            "Stock Symbol",
            value="AAPL",
            help="Enter the stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
        ).upper()
    
    with col2:
        company_name = st.text_input(
            "Company Name",
            value="Apple Inc.",
            help="Enter the full company name"
        )
    
    # Additional stock details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sector = st.selectbox(
            "Sector",
            ["Technology", "Healthcare", "Financial", "Consumer Discretionary", 
             "Consumer Staples", "Energy", "Utilities", "Industrials", 
             "Materials", "Real Estate", "Communication Services"],
            index=0
        )
    
    with col2:
        current_price = st.number_input(
            "Current Price ($)",
            min_value=0.01,
            value=150.0,
            step=0.01
        )
    
    with col3:
        market_cap = st.number_input(
            "Market Cap (Billions $)",
            min_value=0.1,
            value=2500.0,
            step=0.1
        ) * 1e9  # Convert to actual market cap
    
    # Advanced metrics
    with st.expander("📈 Advanced Metrics (Optional)"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pe_ratio = st.number_input("P/E Ratio", min_value=0.0, value=25.0, step=0.1)
        
        with col2:
            dividend_yield = st.number_input("Dividend Yield (%)", min_value=0.0, value=0.5, step=0.1)
        
        with col3:
            beta = st.number_input("Beta", min_value=0.0, value=1.2, step=0.1)
        
        with col4:
            volume = st.number_input("Volume", min_value=0, value=50000000, step=1000000)
    
    # Analysis button
    if st.button("🔍 Analyze Stock", type="primary"):
        if stock_symbol and company_name and mas is not None:
            # Create stock object
            stock = Stock(
                symbol=stock_symbol,
                company_name=company_name,
                sector=sector,
                current_price=current_price,
                market_cap=market_cap,
                pe_ratio=pe_ratio if pe_ratio > 0 else None,
                dividend_yield=dividend_yield if dividend_yield > 0 else None,
                beta=beta if beta > 0 else None,
                volume=volume if volume > 0 else None
            )
            
            st.markdown("---")
            st.subheader(f"📈 Analysis Results for {stock_symbol}")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Analyze with each agent
            agent_analyses = {}
            agent_names = list(mas.agents.keys())
            
            for i, (agent_name, agent) in enumerate(mas.agents.items()):
                status_text.text(f"Running {agent_name.title()} Agent analysis...")
                progress_bar.progress((i + 1) / len(agent_names))
                
                try:
                    analysis = agent.analyze(stock)
                    agent_analyses[agent_name] = analysis
                except Exception as e:
                    st.error(f"Error in {agent_name} analysis: {e}")
                    continue
            
            status_text.text("Analysis complete!")
            progress_bar.progress(1.0)
            
            # Display results
            if agent_analyses:
                # Summary metrics
                st.subheader("📊 Analysis Summary")
                
                recommendations = [a.recommendation.value for a in agent_analyses.values()]
                confidence_scores = [a.confidence_score for a in agent_analyses.values()]
                risk_assessments = [a.risk_assessment for a in agent_analyses.values()]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    most_common_rec = max(set(recommendations), key=recommendations.count)
                    rec_class = f"recommendation-{most_common_rec}"
                    st.markdown(f'<div class="metric-card"><h3>Consensus</h3><p class="{rec_class}">{most_common_rec.upper()}</p></div>', unsafe_allow_html=True)
                
                with col2:
                    avg_confidence = sum(confidence_scores) / len(confidence_scores)
                    st.markdown(f'<div class="metric-card"><h3>Avg Confidence</h3><p>{avg_confidence:.2f}</p></div>', unsafe_allow_html=True)
                
                with col3:
                    most_common_risk = max(set(risk_assessments), key=risk_assessments.count)
                    st.markdown(f'<div class="metric-card"><h3>Risk Level</h3><p>{most_common_risk}</p></div>', unsafe_allow_html=True)
                
                with col4:
                    target_prices = [a.target_price for a in agent_analyses.values() if a.target_price]
                    if target_prices:
                        avg_target = sum(target_prices) / len(target_prices)
                        st.markdown(f'<div class="metric-card"><h3>Avg Target</h3><p>${avg_target:.2f}</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="metric-card"><h3>Avg Target</h3><p>N/A</p></div>', unsafe_allow_html=True)
                
                # Individual agent results
                st.subheader("🤖 Individual Agent Analysis")
                
                for agent_name, analysis in agent_analyses.items():
                    with st.expander(f"{agent_name.title()} Agent - {analysis.recommendation.value.upper()}", expanded=True):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write("**Reasoning:**")
                            st.write(analysis.reasoning)
                            
                            if analysis.key_factors:
                                st.write("**Key Factors:**")
                                for factor in analysis.key_factors:
                                    st.write(f"• {factor}")
                            
                            if analysis.concerns:
                                st.write("**Concerns:**")
                                for concern in analysis.concerns:
                                    st.write(f"⚠️ {concern}")
                        
                        with col2:
                            st.metric("Confidence", f"{analysis.confidence_score:.2f}")
                            st.metric("Risk Assessment", analysis.risk_assessment)
                            if analysis.target_price:
                                st.metric("Target Price", f"${analysis.target_price:.2f}")
                
                # Save analysis to database
                try:
                    db.save_stock_analysis(stock, agent_analyses)
                    st.success("✅ Analysis saved to database")
                except Exception as e:
                    st.warning(f"Could not save to database: {e}")
            
            else:
                st.error("No successful analyses completed. Please check your configuration.")
        
        else:
            st.error("Please provide stock symbol, company name, and ensure the system is properly configured.")
    
    # Recent analyses
    st.markdown("---")
    st.subheader("📚 Recent Analyses")
    
    try:
        recent_analyses = db.get_recent_analyses(limit=5)
        if recent_analyses:
            df = pd.DataFrame(recent_analyses)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No recent analyses found.")
    except Exception as e:
        st.info("Database not available for recent analyses.")

if __name__ == "__main__":
    main()

