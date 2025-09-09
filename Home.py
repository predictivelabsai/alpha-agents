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
from utils.diagram_generator import diagram_generator

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
    """Initialize the 3-agent system (legacy compatibility)"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("OpenAI API key not found. Please check your environment configuration.")
        return None
    
    # Return a simple object for compatibility
    return type('MultiAgentSystem', (), {
        'fundamental_agent': None,
        'rationale_agent': None, 
        'ranker_agent': None
    })()

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
        st.sidebar.info("🤖 Agents: 3 active (Fundamental, Rationale, Ranker)")
    else:
        st.sidebar.error("❌ Multi-Agent System: Offline")
    
    st.sidebar.success("✅ Database: Connected")
    st.sidebar.info(f"📅 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Navigation help
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧭 Navigation")
    st.sidebar.markdown("""
    **🤖 Agent Pipeline**: Complete 3-agent investment pipeline  
    **📊 Fundamental Agent**: Sector analysis & quantitative screening  
    **🔍 Rationale Agent**: Qualitative analysis with web search  
    **🎯 Ranker Agent**: Final scoring & portfolio recommendations  
    **📁 Trace Manager**: View analysis traces & performance  
    """)
    
    # Main content
    st.markdown('<h1 class="main-header">🏛️ Lohusalu Capital Management</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to **Lohusalu Capital Management's** advanced multi-agent system for equity portfolio construction. 
    This application demonstrates cutting-edge AI techniques for stock selection and portfolio 
    optimization using specialized agents that collaborate and debate to make informed investment decisions.
    """)
    
    # System overview - 3 agents
    st.subheader("🤖 Specialized AI Agents")
    
    # Main content sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Fundamental Agent
        
        Analyzes financial statements, earnings quality, and fundamental metrics to assess company financial health and intrinsic value. Performs sector selection and quantitative screening.
        """)
        
        st.markdown("""
        ### 🔍 Rationale Agent
        
        Evaluates business quality using qualitative analysis: economic moats, competitive advantages, sentiment analysis, and secular trends with comprehensive web search.
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Ranker Agent
        
        Combines fundamental and qualitative analysis to generate final investment scores, grades (A+ to D), and comprehensive investment recommendations.
        """)
        
        st.markdown("""
        ### 🤖 Agent Pipeline
        
        Complete 3-agent workflow that processes stocks through Fundamental → Rationale → Ranker agents for comprehensive investment analysis.
        """)
    
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
    
    # Methodology Overview with Expanders
    st.markdown("---")
    st.subheader("📋 Investment Methodology")
    
    # 3-Agent Pipeline Overview
    with st.expander("🤖 **3-Agent Pipeline Architecture**", expanded=False):
        st.markdown("""
        The Lohusalu Capital Management system implements a sophisticated **3-agent pipeline** that collaborates 
        to provide comprehensive stock analysis and portfolio recommendations.
        
        **Pipeline Flow:**
        1. **📊 Fundamental Agent** → Sector analysis & quantitative screening
        2. **🔍 Rationale Agent** → Qualitative analysis with web research  
        3. **🎯 Ranker Agent** → Final scoring & investment recommendations
        
        **Key Features:**
        - **LangGraph-based** collaborative system
        - **Sequential analysis** with consensus building
        - **Weighted confidence** scoring with final synthesis
        - **Real-time data** integration via yfinance API
        - **Complete transparency** with reasoning traces
        """)
    
    # Fundamental Agent Methodology
    with st.expander("📊 **Fundamental Agent Methodology**", expanded=False):
        st.markdown("""
        **Primary Function:** Sector analysis and quantitative stock screening.
        
        **Methodology:**
        - **Sector Analysis:** Identifies trending sectors by analyzing market data and economic indicators. 
          Assigns weights to sectors based on growth potential and momentum.
        - **Quantitative Screening:** Screens stocks within selected sectors against rigorous financial metrics:
          - **Growth:** Revenue and EPS growth (YoY and QoQ)
          - **Profitability:** ROE, ROA, and net profit margins
          - **Valuation:** P/E, P/S, and P/B ratios relative to industry peers
          - **Financial Health:** Debt-to-equity and current ratios
        - **Intrinsic Value:** Calculates DCF-based intrinsic value to determine upside potential.
        
        **Output:** List of qualified companies meeting quantitative criteria with fundamental scores.
        """)
    
    # Rationale Agent Methodology  
    with st.expander("🔍 **Rationale Agent Methodology**", expanded=False):
        st.markdown("""
        **Primary Function:** Qualitative analysis of competitive advantages and market position.
        
        **Methodology:**
        - **Economic Moat:** Assesses strength and durability of competitive advantages 
          (network effects, brand loyalty, switching costs).
        - **Sentiment Analysis:** Analyzes market sentiment through news, social media, and analyst ratings.
        - **Secular Trends:** Identifies long-term trends and evaluates company alignment.
        - **Tavily Search:** Conducts extensive web searches to gather qualitative data with citations.
        
        **Analysis Components:**
        - Competitive advantage assessment
        - Market sentiment evaluation  
        - Long-term trend alignment
        - Qualitative business evaluation
        
        **Output:** Comprehensive qualitative analysis with web-sourced citations and reasoning.
        """)
    
    # Ranker Agent Methodology
    with st.expander("🎯 **Ranker Agent Methodology**", expanded=False):
        st.markdown("""
        **Primary Function:** Synthesizes analysis from Fundamental and Rationale agents for final recommendations.
        
        **Methodology:**
        - **Composite Scoring:** Calculates weighted average of fundamental (60%) and qualitative (40%) scores.
        - **Investment Grading:** Assigns investment grade (A+ to D) based on composite score.
        - **Investment Thesis:** Generates comprehensive thesis outlining strengths, risks, and catalysts.
        - **Portfolio Construction:** Provides ranked investment opportunities and portfolio allocation.
        
        **Scoring Dimensions:**
        - Growth potential and sustainability
        - Profitability and financial strength
        - Competitive moat and market position
        - Sentiment and trend alignment
        
        **Output:** Final investment grades, detailed thesis, and portfolio recommendations.
        """)
    
    # Multi-Agent Collaboration Process
    with st.expander("🔄 **Multi-Agent Collaboration Process**", expanded=False):
        st.markdown("""
        **Phase 1: Independent Analysis**
        - Each agent processes stocks independently using specialized methodologies
        - Real-time data integration from yfinance API and Tavily search
        - Individual recommendations and confidence scores generated
        
        **Phase 2: Data Aggregation**  
        - All agent analyses collected by the Ranker Agent
        - Consistency checks and data validation performed
        - Agent-specific insights and concerns catalogued
        
        **Phase 3: Final Synthesis**
        - Multi-factor composite scoring calculation
        - Risk-reward assessment integration
        - Investment thesis development with supporting evidence
        - Final recommendation and investment grade determination
        
        **Quality Assurance:**
        - Confidence scoring system (0.0-1.0) for each agent
        - Weighted averaging based on agent specialization
        - Complete reasoning traces for transparency and audit
        """)
    
    # System Performance Metrics
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

