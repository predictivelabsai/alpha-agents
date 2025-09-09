"""
Fundamental Agent Page - Lohusalu Capital Management
Individual page for running and analyzing the Fundamental Agent
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
from typing import Dict, List, Optional, Any
import asyncio
import logging

# Add src to path for imports
sys.path.append(os.path.dirname(__file__))'))

from agents.agents.fundamental_agent_v2 import FundamentalAgent

# Page configuration
st.set_page_config(
    page_title="Fundamental Agent - Lohusalu Capital Management",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
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
    .agent-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
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
    .score-high { color: #28a745; font-weight: bold; }
    .score-medium { color: #ffc107; font-weight: bold; }
    .score-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def get_score_class(score):
    """Get CSS class based on score"""
    if score >= 75:
        return "score-high"
    elif score >= 50:
        return "score-medium"
    else:
        return "score-low"

def main():
    """Main function for Fundamental Agent page"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Fundamental Agent</h1>
        <p>Sector Selection & Quantitative Stock Screening</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Agent Description
    st.markdown("""
    <div class="agent-card">
        <h3>🎯 Agent Capabilities</h3>
        <ul>
            <li><strong>Sector Analysis:</strong> Identifies trending sectors and assigns investment weights</li>
            <li><strong>Quantitative Screening:</strong> Screens stocks against financial metrics and ratios</li>
            <li><strong>Intrinsic Value:</strong> Calculates DCF-based valuations and upside potential</li>
            <li><strong>Multi-Model Support:</strong> Configurable LLM providers for analysis</li>
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
    
    # Analysis parameters
    st.sidebar.markdown("### 📈 Analysis Parameters")
    
    max_stocks = st.sidebar.slider(
        "Maximum Stocks to Analyze",
        min_value=5,
        max_value=50,
        value=20,
        help="Maximum number of stocks to return from screening"
    )
    
    save_trace = st.sidebar.checkbox(
        "Save Analysis Trace",
        value=True,
        help="Save detailed reasoning trace to JSON file"
    )
    
    # Screening criteria customization
    with st.sidebar.expander("🔍 Screening Criteria", expanded=False):
        min_market_cap = st.number_input(
            "Min Market Cap ($M)",
            min_value=50,
            max_value=1000,
            value=100,
            step=50
        ) * 1e6
        
        max_market_cap = st.number_input(
            "Max Market Cap ($B)",
            min_value=1,
            max_value=50,
            value=10,
            step=1
        ) * 1e9
        
        min_revenue_growth = st.slider(
            "Min Revenue Growth (%)",
            min_value=0,
            max_value=50,
            value=10
        )
        
        min_roe = st.slider(
            "Min ROE (%)",
            min_value=0,
            max_value=30,
            value=12
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🚀 Run Fundamental Analysis")
        
        if st.button("🔍 Start Analysis", type="primary", use_container_width=True):
            run_fundamental_analysis(
                model_provider, model_name, max_stocks, save_trace,
                min_market_cap, max_market_cap, min_revenue_growth, min_roe
            )
    
    with col2:
        st.markdown("## 📋 Current Configuration")
        st.markdown(f"""
        - **Model:** {model_provider.title()} - {model_name}
        - **Max Stocks:** {max_stocks}
        - **Market Cap:** ${min_market_cap/1e6:.0f}M - ${max_market_cap/1e9:.0f}B
        - **Min Revenue Growth:** {min_revenue_growth}%
        - **Min ROE:** {min_roe}%
        """)
    
    # Display recent traces
    display_recent_traces()

def run_fundamental_analysis(model_provider, model_name, max_stocks, save_trace,
                           min_market_cap, max_market_cap, min_revenue_growth, min_roe):
    """Run the fundamental analysis"""
    
    try:
        # Initialize agent
        with st.spinner("Initializing Fundamental Agent..."):
            agent = FundamentalAgent(model_provider=model_provider, model_name=model_name)
            
            # Update screening criteria
            agent.screening_criteria.update({
                'min_market_cap': min_market_cap,
                'max_market_cap': max_market_cap,
                'min_revenue_growth': min_revenue_growth,
                'min_roe': min_roe
            })
        
        # Run sector analysis
        with st.spinner("Analyzing sectors..."):
            sector_analyses = agent.analyze_sectors()
        
        # Display sector analysis results
        display_sector_analysis(sector_analyses)
        
        # Run stock screening
        with st.spinner("Screening stocks..."):
            stock_screenings = agent.screen_stocks(sector_analyses, max_stocks)
        
        # Display stock screening results
        display_stock_screening(stock_screenings)
        
        # Save trace if requested
        if save_trace:
            with st.spinner("Saving analysis trace..."):
                trace_file = agent.save_analysis_trace(sector_analyses, stock_screenings)
                if trace_file:
                    st.success(f"✅ Analysis trace saved to: {trace_file}")
        
        st.success("🎉 Fundamental analysis completed successfully!")
        
        # Display expandable trace
        display_analysis_trace(agent, sector_analyses, stock_screenings)
        
    except Exception as e:
        st.error(f"❌ Error in fundamental analysis: {str(e)}")
        st.exception(e)

def display_sector_analysis(sector_analyses):
    """Display sector analysis results"""
    
    st.markdown("## 🏢 Sector Analysis Results")
    
    if not sector_analyses:
        st.warning("No sector analysis results to display")
        return
    
    # Create sector summary DataFrame
    sector_df = pd.DataFrame([
        {
            'Sector': sa.sector,
            'Weight': sa.weight,
            'Momentum Score': sa.momentum_score,
            'Growth Potential': sa.growth_potential,
            'Top Stocks': ', '.join(sa.top_stocks[:3]) if sa.top_stocks else 'N/A'
        }
        for sa in sector_analyses
    ])
    
    # Display top sectors
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Sector Rankings")
        
        # Create interactive bar chart
        fig = px.bar(
            sector_df.head(8),
            x='Weight',
            y='Sector',
            orientation='h',
            color='Weight',
            color_continuous_scale='Blues',
            title="Sector Investment Weights"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🥇 Top 3 Sectors")
        
        for i, sa in enumerate(sector_analyses[:3], 1):
            score_class = get_score_class(sa.weight)
            st.markdown(f"""
            <div class="metric-card">
                <h4>#{i} {sa.sector}</h4>
                <p class="{score_class}">Weight: {sa.weight:.1f}/100</p>
                <p><small>Momentum: {sa.momentum_score:.1f} | Growth: {sa.growth_potential:.1f}</small></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed sector table
    st.markdown("### 📋 Detailed Sector Analysis")
    
    # Format the DataFrame for display
    display_df = sector_df.copy()
    display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x:.1f}")
    display_df['Momentum Score'] = display_df['Momentum Score'].apply(lambda x: f"{x:.1f}")
    display_df['Growth Potential'] = display_df['Growth Potential'].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Sector reasoning
    with st.expander("🔍 Sector Analysis Reasoning", expanded=False):
        for sa in sector_analyses[:5]:  # Show top 5
            st.markdown(f"**{sa.sector}** (Weight: {sa.weight:.1f})")
            st.markdown(f"*{sa.reasoning}*")
            st.markdown("---")

def display_stock_screening(stock_screenings):
    """Display stock screening results"""
    
    st.markdown("## 📈 Stock Screening Results")
    
    if not stock_screenings:
        st.warning("No stocks passed the screening criteria")
        return
    
    # Create stock summary DataFrame
    stock_df = pd.DataFrame([
        {
            'Ticker': ss.ticker,
            'Sector': ss.sector,
            'Fundamental Score': ss.fundamental_score,
            'Market Cap': ss.market_cap,
            'Intrinsic Value': ss.intrinsic_value,
            'Current Price': ss.current_price,
            'Upside Potential': ss.upside_potential
        }
        for ss in stock_screenings
    ])
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Stocks Found",
            len(stock_screenings),
            help="Number of stocks passing screening criteria"
        )
    
    with col2:
        avg_score = stock_df['Fundamental Score'].mean()
        st.metric(
            "Avg Score",
            f"{avg_score:.1f}",
            help="Average fundamental score"
        )
    
    with col3:
        avg_upside = stock_df['Upside Potential'].mean()
        st.metric(
            "Avg Upside",
            f"{avg_upside:.1f}%",
            help="Average upside potential"
        )
    
    with col4:
        sectors_count = stock_df['Sector'].nunique()
        st.metric(
            "Sectors",
            sectors_count,
            help="Number of different sectors represented"
        )
    
    # Top stocks visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🏆 Top Stocks by Fundamental Score")
        
        top_stocks = stock_df.head(10)
        fig = px.scatter(
            top_stocks,
            x='Upside Potential',
            y='Fundamental Score',
            size='Market Cap',
            color='Sector',
            hover_name='Ticker',
            title="Risk-Return Profile"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🥇 Top 5 Stocks")
        
        for i, ss in enumerate(stock_screenings[:5], 1):
            score_class = get_score_class(ss.fundamental_score)
            st.markdown(f"""
            <div class="metric-card">
                <h4>#{i} {ss.ticker}</h4>
                <p class="{score_class}">Score: {ss.fundamental_score:.1f}/100</p>
                <p><small>Upside: {ss.upside_potential:.1f}% | {ss.sector}</small></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed stock table
    st.markdown("### 📋 Detailed Stock Analysis")
    
    # Format the DataFrame for display
    display_df = stock_df.copy()
    display_df['Fundamental Score'] = display_df['Fundamental Score'].apply(lambda x: f"{x:.1f}")
    display_df['Market Cap'] = display_df['Market Cap'].apply(
        lambda x: f"${x/1e9:.2f}B" if x > 1e9 else f"${x/1e6:.0f}M"
    )
    display_df['Intrinsic Value'] = display_df['Intrinsic Value'].apply(lambda x: f"${x:.2f}")
    display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"${x:.2f}")
    display_df['Upside Potential'] = display_df['Upside Potential'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Stock reasoning
    with st.expander("🔍 Stock Analysis Reasoning", expanded=False):
        for ss in stock_screenings[:5]:  # Show top 5
            st.markdown(f"**{ss.ticker}** - {ss.sector}")
            st.markdown(f"*Score: {ss.fundamental_score:.1f}/100 | Upside: {ss.upside_potential:.1f}%*")
            st.markdown(f"{ss.reasoning}")
            st.markdown("---")

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
        if f.startswith("fundamental_agent_trace_") and f.endswith(".json")
    ]
    
    if not trace_files:
        st.info("No fundamental agent traces found.")
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
            
            with st.expander(f"📄 {trace_file} - {timestamp}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Analysis Summary:**")
                    st.markdown(f"- Model: {model_info}")
                    st.markdown(f"- Sectors Analyzed: {len(trace_data.get('sector_analyses', []))}")
                    st.markdown(f"- Stocks Found: {len(trace_data.get('stock_screenings', []))}")
                
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



def display_analysis_trace(agent, sector_analyses, stock_screenings):
    """Display expandable analysis trace"""
    
    with st.expander("🔍 **Analysis Trace & Reasoning**", expanded=False):
        st.markdown("### 📊 Complete Analysis Trace")
        
        # Create trace data
        trace_data = {
            "timestamp": datetime.now().isoformat(),
            "agent": "FundamentalAgent",
            "model_provider": agent.model_provider,
            "model_name": agent.model_name,
            "sector_analyses": [
                {
                    "sector": sa.sector,
                    "weight": sa.weight,
                    "momentum_score": sa.momentum_score,
                    "growth_potential": sa.growth_potential,
                    "reasoning": sa.reasoning,
                    "top_stocks": sa.top_stocks
                } for sa in sector_analyses
            ],
            "stock_screenings": [
                {
                    "symbol": ss.symbol,
                    "company_name": ss.company_name,
                    "sector": ss.sector,
                    "fundamental_score": ss.fundamental_score,
                    "intrinsic_value": ss.intrinsic_value,
                    "current_price": ss.current_price,
                    "upside_potential": ss.upside_potential,
                    "financial_metrics": ss.financial_metrics,
                    "reasoning": ss.reasoning
                } for ss in stock_screenings
            ],
            "screening_criteria": agent.screening_criteria,
            "total_sectors_analyzed": len(sector_analyses),
            "total_stocks_screened": len(stock_screenings)
        }
        
        # Display as formatted JSON
        st.json(trace_data)
        
        # Display as DataFrame
        st.markdown("### 📋 Sector Analysis Summary")
        if sector_analyses:
            sector_df = pd.DataFrame([
                {
                    'Sector': sa.sector,
                    'Weight': sa.weight,
                    'Momentum': sa.momentum_score,
                    'Growth Potential': sa.growth_potential,
                    'Top Stocks': ', '.join(sa.top_stocks[:3])
                } for sa in sector_analyses
            ])
            st.dataframe(sector_df, use_container_width=True)
        
        st.markdown("### 📈 Stock Screening Summary")
        if stock_screenings:
            stock_df = pd.DataFrame([
                {
                    'Symbol': ss.symbol,
                    'Company': ss.company_name,
                    'Sector': ss.sector,
                    'Score': ss.fundamental_score,
                    'Current Price': f"${ss.current_price:.2f}",
                    'Intrinsic Value': f"${ss.intrinsic_value:.2f}",
                    'Upside': f"{ss.upside_potential:.1f}%"
                } for ss in stock_screenings
            ])
            st.dataframe(stock_df, use_container_width=True)
        
        # Download trace button
        trace_json = json.dumps(trace_data, indent=2)
        st.download_button(
            label="📥 Download Analysis Trace (JSON)",
            data=trace_json,
            file_name=f"fundamental_analysis_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

