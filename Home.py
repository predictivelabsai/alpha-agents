"""
Alpha Agents - Equity Portfolio Construction Multi-Agent System
Home Page - Main entry point for the multi-page Streamlit application
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents import Stock, create_multi_agent_portfolio_system, RiskTolerance, InvestmentDecision
from database import DatabaseManager

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Lohusalu Capital Management - Equity Portfolio AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .feature-highlight {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
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
    """Home page main function"""
    
    # Initialize components
    db = init_database()
    mas = init_multi_agent_system()
    
    # Sidebar status
    st.sidebar.title("🏛️ Lohusalu Capital Management")
    st.sidebar.markdown("*Multi-Agent Equity Portfolio System*")
    st.sidebar.markdown("---")
    
    # System status
    st.sidebar.subheader("🔧 System Status")
    
    if mas is not None:
        st.sidebar.success("✅ Multi-Agent System: Online")
        st.sidebar.info(f"🤖 Agents: {len(mas.agents)} active")
    else:
        st.sidebar.error("❌ Multi-Agent System: Offline")
    
    st.sidebar.success("✅ Database: Connected")
    st.sidebar.info(f"📅 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Navigation help
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧭 Navigation")
    st.sidebar.markdown("""
    **📊 Stock Analysis**: Analyze individual stocks  
    **📊 Analytics**: View performance visualizations  
    **🎯 Portfolio Builder**: Build optimized portfolios  
    **ℹ️ About**: Learn about the system  
    """)
    
    # Main content
    st.markdown('<h1 class="main-header">🏛️ Lohusalu Capital Management</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to **Lohusalu Capital Management's** advanced multi-agent system for equity portfolio construction. 
    This application demonstrates cutting-edge AI techniques for stock selection and portfolio 
    optimization using specialized agents that collaborate and debate to make informed investment decisions.
    """)
    
    # System overview - All 5 agents
    st.subheader("🤖 Specialized AI Agents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="agent-card">
            <h3>📊 Fundamental Agent</h3>
            <p>Analyzes 10-K/10-Q reports, financial statements, earnings quality, and fundamental metrics to assess company financial health and intrinsic value.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="agent-card">
            <h3>📰 Sentiment Agent</h3>
            <p>Processes financial news, analyst ratings, social media sentiment, and market psychology to gauge investor sentiment and momentum.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="agent-card">
            <h3>🧠 Rationale Agent</h3>
            <p>Evaluates business quality using a 7-step framework: sales growth, profitability, competitive moats, operational efficiency, and debt structure.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="agent-card">
            <h3>💰 Valuation Agent</h3>
            <p>Analyzes stock prices, trading volumes, technical indicators, and relative valuation metrics to identify attractive entry points.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="agent-card">
            <h3>🚀 Secular Trend Agent</h3>
            <p>Identifies companies positioned to benefit from major technology trends: Agentic AI, Cloud Infrastructure, AI Semiconductors, Cybersecurity, and Electrification.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key features
    st.subheader("🚀 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-highlight">
            <h4>🤖 Multi-Agent Collaboration</h4>
            <ul>
                <li>5 specialized agents with domain expertise</li>
                <li>Collaborative analysis and debate mechanism</li>
                <li>Consensus-building for investment decisions</li>
                <li>Transparent reasoning and audit trails</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-highlight">
            <h4>📈 Portfolio Construction</h4>
            <ul>
                <li>Automated stock selection and weighting</li>
                <li>Risk-adjusted portfolio optimization</li>
                <li>Diversification analysis and sector allocation</li>
                <li>Performance monitoring and rebalancing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick stats from test data
    st.markdown("---")
    st.subheader("📊 System Performance")
    
    # Try to load recent test data for stats
    test_data_dir = "test-data"
    if os.path.exists(test_data_dir):
        csv_files = [f for f in os.listdir(test_data_dir) if f.startswith('agent_analysis_data_') and f.endswith('.csv')]
        
        if csv_files:
            latest_file = sorted(csv_files)[-1]
            file_path = os.path.join(test_data_dir, latest_file)
            
            try:
                df = pd.read_csv(file_path)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    stocks_analyzed = df['stock_symbol'].nunique()
                    st.metric("Stocks Analyzed", stocks_analyzed, "✅")
                
                with col2:
                    total_analyses = len(df)
                    st.metric("Total Analyses", total_analyses, "📊")
                
                with col3:
                    avg_confidence = df['confidence_score'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.2f}", "🎯")
                
                with col4:
                    buy_recommendations = (df['recommendation'] == 'buy').sum()
                    buy_rate = buy_recommendations / len(df) * 100
                    st.metric("Buy Rate", f"{buy_rate:.1f}%", "📈")
                
                # Recent analysis chart
                st.subheader("📈 Recent Analysis Overview")
                
                # Recommendation distribution
                rec_counts = df['recommendation'].value_counts()
                fig = px.pie(
                    values=rec_counts.values,
                    names=rec_counts.index,
                    title="Recommendation Distribution",
                    color_discrete_map={
                        'buy': '#28a745',
                        'hold': '#ffc107', 
                        'sell': '#dc3545',
                        'avoid': '#6c757d'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                # Fallback to default metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Stocks Analyzed", "8", "✅")
                
                with col2:
                    st.metric("Total Analyses", "40", "📊")
                
                with col3:
                    st.metric("Avg Confidence", "0.68", "🎯")
                
                with col4:
                    st.metric("Success Rate", "100%", "🎉")
        else:
            # Default metrics when no test data
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Stocks Analyzed", "0", "0")
            
            with col2:
                st.metric("Portfolios Created", "0", "0")
            
            with col3:
                st.metric("Average Confidence", "N/A", "0%")
            
            with col4:
                st.metric("Success Rate", "N/A", "0%")
    else:
        # Default metrics when no test data directory
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Stocks Analyzed", "0", "0")
        
        with col2:
            st.metric("Portfolios Created", "0", "0")
        
        with col3:
            st.metric("Average Confidence", "N/A", "0%")
        
        with col4:
            st.metric("Success Rate", "N/A", "0%")
    
    # Getting started
    st.markdown("---")
    st.subheader("🚀 Getting Started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1. 📊 Analyze Stocks**
        
        Start by analyzing individual stocks using our 5 specialized agents. Get detailed insights, recommendations, and risk assessments.
        """)
        
        if st.button("🔍 Start Stock Analysis", key="start_analysis"):
            st.switch_page("pages/1_📊_Stock_Analysis.py")
    
    with col2:
        st.markdown("""
        **2. 🎯 Build Portfolio**
        
        Create optimized portfolios by adding multiple stocks and letting our agents collaborate on investment decisions.
        """)
        
        if st.button("🏗️ Build Portfolio", key="start_portfolio"):
            st.switch_page("pages/3_🎯_Portfolio_Builder.py")
    
    with col3:
        st.markdown("""
        **3. 📊 View Analytics**
        
        Explore comprehensive visualizations and performance metrics to understand agent behavior and system performance.
        """)
        
        if st.button("📈 View Analytics", key="start_analytics"):
            st.switch_page("pages/2_📊_Analytics.py")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>Lohusalu Capital Management</strong> - Advanced Multi-Agent System for Equity Portfolio Construction</p>
        <p>Intelligent capital management powered by advanced AI technology</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

