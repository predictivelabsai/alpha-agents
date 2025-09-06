"""
Rationale Agent Page - Lohusalu Capital Management
Individual page for running and analyzing the Rationale Agent
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

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.rationale_agent_v2 import RationaleAgent

# Page configuration
st.set_page_config(
    page_title="Rationale Agent - Lohusalu Capital Management",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #28a745, #20c997);
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
        border-left: 4px solid #28a745;
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
    .moat-wide { color: #28a745; font-weight: bold; }
    .moat-narrow { color: #ffc107; font-weight: bold; }
    .moat-none { color: #dc3545; font-weight: bold; }
    .sentiment-positive { color: #28a745; font-weight: bold; }
    .sentiment-neutral { color: #6c757d; font-weight: bold; }
    .sentiment-negative { color: #dc3545; font-weight: bold; }
    .citation-box {
        background: #e9ecef;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

def get_moat_class(moat_strength):
    """Get CSS class based on moat strength"""
    if moat_strength == "Wide":
        return "moat-wide"
    elif moat_strength == "Narrow":
        return "moat-narrow"
    else:
        return "moat-none"

def get_sentiment_class(sentiment):
    """Get CSS class based on sentiment"""
    if sentiment == "Positive":
        return "sentiment-positive"
    elif sentiment == "Neutral":
        return "sentiment-neutral"
    else:
        return "sentiment-negative"

def main():
    """Main function for Rationale Agent page"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Rationale Agent</h1>
        <p>Qualitative Analysis with Web Search Integration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Agent Description
    st.markdown("""
    <div class="agent-card">
        <h3>🎯 Agent Capabilities</h3>
        <ul>
            <li><strong>Economic Moat Analysis:</strong> Network effects, switching costs, competitive advantages</li>
            <li><strong>Sentiment Analysis:</strong> Market sentiment, analyst views, recent developments</li>
            <li><strong>Secular Trends:</strong> Long-term growth drivers and industry trends</li>
            <li><strong>Web Search Integration:</strong> Tavily API for comprehensive research with citations</li>
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
    
    # Stock input
    st.sidebar.markdown("### 📈 Stock Selection")
    
    ticker = st.sidebar.text_input(
        "Stock Ticker",
        value="AAPL",
        help="Enter the stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
    ).upper()
    
    company_name = st.sidebar.text_input(
        "Company Name",
        value="Apple Inc.",
        help="Enter the full company name"
    )
    
    # Analysis options
    st.sidebar.markdown("### 🔍 Analysis Options")
    
    analysis_types = st.sidebar.multiselect(
        "Select Analysis Types",
        options=["Economic Moat", "Sentiment Analysis", "Secular Trends", "Competitive Position"],
        default=["Economic Moat", "Sentiment Analysis", "Secular Trends", "Competitive Position"],
        help="Choose which types of analysis to perform"
    )
    
    save_trace = st.sidebar.checkbox(
        "Save Analysis Trace",
        value=True,
        help="Save detailed reasoning trace to JSON file"
    )
    
    # Search configuration
    with st.sidebar.expander("🔍 Search Configuration", expanded=False):
        max_search_results = st.slider(
            "Max Search Results per Query",
            min_value=5,
            max_value=20,
            value=10
        )
        
        search_timeout = st.slider(
            "Search Timeout (seconds)",
            min_value=15,
            max_value=60,
            value=30
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🚀 Run Qualitative Analysis")
        
        if st.button("🔍 Start Analysis", type="primary", use_container_width=True):
            if ticker and company_name:
                run_rationale_analysis(
                    ticker, company_name, model_provider, model_name,
                    analysis_types, save_trace, max_search_results, search_timeout
                )
            else:
                st.error("Please enter both ticker and company name")
    
    with col2:
        st.markdown("## 📋 Current Configuration")
        st.markdown(f"""
        - **Model:** {model_provider.title()} - {model_name}
        - **Stock:** {ticker} ({company_name})
        - **Analysis Types:** {len(analysis_types)} selected
        - **Search Results:** {max_search_results} per query
        """)
    
    # Display recent traces
    display_recent_traces()

def run_rationale_analysis(ticker, company_name, model_provider, model_name,
                          analysis_types, save_trace, max_search_results, search_timeout):
    """Run the rationale analysis"""
    
    try:
        # Initialize agent
        with st.spinner("Initializing Rationale Agent..."):
            agent = RationaleAgent(model_provider=model_provider, model_name=model_name)
            agent.max_search_results = max_search_results
            agent.search_timeout = search_timeout
        
        # Check Tavily API key
        if not os.getenv('TAVILY_API_KEY'):
            st.warning("⚠️ TAVILY_API_KEY not found. Web search functionality will be limited.")
        
        # Run individual analyses based on selection
        results = {}
        
        if "Economic Moat" in analysis_types:
            with st.spinner("Analyzing economic moat..."):
                moat_analysis = agent.analyze_economic_moat(ticker, company_name)
                results['moat'] = moat_analysis
                display_moat_analysis(moat_analysis)
        
        if "Sentiment Analysis" in analysis_types:
            with st.spinner("Analyzing market sentiment..."):
                sentiment_analysis = agent.analyze_sentiment(ticker, company_name)
                results['sentiment'] = sentiment_analysis
                display_sentiment_analysis(sentiment_analysis)
        
        if "Secular Trends" in analysis_types:
            with st.spinner("Analyzing secular trends..."):
                trends_analysis = agent.analyze_secular_trends(ticker, company_name)
                results['trends'] = trends_analysis
                display_trends_analysis(trends_analysis)
        
        if "Competitive Position" in analysis_types:
            with st.spinner("Analyzing competitive position..."):
                competitive_analysis = agent.analyze_competitive_position(ticker, company_name)
                results['competitive'] = competitive_analysis
                display_competitive_analysis(competitive_analysis)
        
        # Run complete analysis if all types selected
        if len(analysis_types) == 4:
            with st.spinner("Generating comprehensive analysis..."):
                complete_analysis = agent.run_qualitative_analysis(ticker, company_name)
                display_comprehensive_analysis(complete_analysis)
                
                # Save trace if requested
                if save_trace:
                    trace_file = agent.save_analysis_trace(complete_analysis)
                    if trace_file:
                        st.success(f"✅ Analysis trace saved to: {trace_file}")
        
        st.success("🎉 Qualitative analysis completed successfully!")
        
    except Exception as e:
        st.error(f"❌ Error in qualitative analysis: {str(e)}")
        st.exception(e)

def display_moat_analysis(moat_analysis):
    """Display economic moat analysis results"""
    
    st.markdown("## 🏰 Economic Moat Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        moat_class = get_moat_class(moat_analysis.moat_strength)
        st.markdown(f"""
        <div class="metric-card">
            <h4>Moat Strength</h4>
            <p class="{moat_class}">{moat_analysis.moat_strength}</p>
            <p><small>Score: {moat_analysis.moat_score:.1f}/100</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Primary Moat Type</h4>
            <p><strong>{moat_analysis.moat_type}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        moat_count = sum([
            moat_analysis.network_effects,
            moat_analysis.switching_costs,
            moat_analysis.cost_advantages,
            moat_analysis.intangible_assets,
            moat_analysis.efficient_scale
        ])
        st.markdown(f"""
        <div class="metric-card">
            <h4>Moat Sources</h4>
            <p><strong>{moat_count}/5</strong></p>
            <p><small>Active moat sources</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Moat components breakdown
    st.markdown("### 🔍 Moat Components")
    
    moat_components = {
        'Network Effects': moat_analysis.network_effects,
        'Switching Costs': moat_analysis.switching_costs,
        'Cost Advantages': moat_analysis.cost_advantages,
        'Intangible Assets': moat_analysis.intangible_assets,
        'Efficient Scale': moat_analysis.efficient_scale
    }
    
    # Create visualization
    components_df = pd.DataFrame([
        {'Component': k, 'Present': 'Yes' if v else 'No', 'Value': 1 if v else 0}
        for k, v in moat_components.items()
    ])
    
    fig = px.bar(
        components_df,
        x='Component',
        y='Value',
        color='Present',
        color_discrete_map={'Yes': '#28a745', 'No': '#dc3545'},
        title="Economic Moat Components"
    )
    fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Reasoning
    with st.expander("📝 Moat Analysis Reasoning", expanded=True):
        st.markdown(moat_analysis.reasoning)

def display_sentiment_analysis(sentiment_analysis):
    """Display sentiment analysis results"""
    
    st.markdown("## 📊 Sentiment Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment_class = get_sentiment_class(sentiment_analysis.overall_sentiment)
        st.markdown(f"""
        <div class="metric-card">
            <h4>Overall Sentiment</h4>
            <p class="{sentiment_class}">{sentiment_analysis.overall_sentiment}</p>
            <p><small>Score: {sentiment_analysis.sentiment_score:.1f}/100</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Recent Developments</h4>
            <p><strong>{len(sentiment_analysis.recent_developments)}</strong></p>
            <p><small>News items analyzed</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Sentiment gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = sentiment_analysis.sentiment_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sentiment Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgray"},
                    {'range': [40, 70], 'color': "gray"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment breakdown
    st.markdown("### 📈 Sentiment Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Analyst Sentiment:**")
        st.markdown(sentiment_analysis.analyst_sentiment)
        
        st.markdown("**News Sentiment:**")
        st.markdown(sentiment_analysis.news_sentiment)
    
    with col2:
        st.markdown("**Social Sentiment:**")
        st.markdown(sentiment_analysis.social_sentiment)
        
        st.markdown("**Recent Developments:**")
        for i, dev in enumerate(sentiment_analysis.recent_developments[:5], 1):
            st.markdown(f"{i}. {dev}")
    
    # Reasoning
    with st.expander("📝 Sentiment Analysis Reasoning", expanded=True):
        st.markdown(sentiment_analysis.reasoning)

def display_trends_analysis(trends_analysis):
    """Display secular trends analysis results"""
    
    st.markdown("## 📈 Secular Trends Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Trend Alignment</h4>
            <p><strong>{trends_analysis.trend_alignment}</strong></p>
            <p><small>Score: {trends_analysis.trend_score:.1f}/100</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Time Horizon</h4>
            <p><strong>{trends_analysis.time_horizon}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Growth Drivers</h4>
            <p><strong>{len(trends_analysis.growth_drivers)}</strong></p>
            <p><small>Identified drivers</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Trends breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 Primary Trends")
        for i, trend in enumerate(trends_analysis.primary_trends, 1):
            st.markdown(f"{i}. **{trend}**")
        
        st.markdown("### 📊 Growth Drivers")
        for i, driver in enumerate(trends_analysis.growth_drivers, 1):
            st.markdown(f"{i}. {driver}")
    
    with col2:
        st.markdown("### ⚠️ Headwinds")
        for i, headwind in enumerate(trends_analysis.headwinds, 1):
            st.markdown(f"{i}. {headwind}")
        
        # Trend score visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = trends_analysis.trend_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Trend Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgray"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ]
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    # Reasoning
    with st.expander("📝 Trends Analysis Reasoning", expanded=True):
        st.markdown(trends_analysis.reasoning)

def display_competitive_analysis(competitive_analysis):
    """Display competitive position analysis results"""
    
    st.markdown("## 🏆 Competitive Position Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Market Position</h4>
            <p><strong>{competitive_analysis.get('market_position', 'Unknown')}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Market Share Trend</h4>
            <p><strong>{competitive_analysis.get('market_share_trend', 'Unknown')}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Competitive Score</h4>
            <p><strong>{competitive_analysis.get('competitive_score', 0):.1f}/100</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Competitive details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏢 Key Competitors")
        competitors = competitive_analysis.get('key_competitors', [])
        for i, competitor in enumerate(competitors, 1):
            st.markdown(f"{i}. **{competitor}**")
        
        st.markdown("### 💪 Competitive Advantages")
        advantages = competitive_analysis.get('competitive_advantages', [])
        for i, advantage in enumerate(advantages, 1):
            st.markdown(f"{i}. {advantage}")
    
    with col2:
        st.markdown("### ⚠️ Competitive Threats")
        threats = competitive_analysis.get('competitive_threats', [])
        for i, threat in enumerate(threats, 1):
            st.markdown(f"{i}. {threat}")
        
        st.markdown("### 🏭 Industry Attractiveness")
        st.markdown(f"**{competitive_analysis.get('industry_attractiveness', 'Unknown')}**")
    
    # Reasoning
    with st.expander("📝 Competitive Analysis Reasoning", expanded=True):
        st.markdown(competitive_analysis.get('reasoning', 'No reasoning available'))

def display_comprehensive_analysis(analysis):
    """Display comprehensive qualitative analysis"""
    
    st.markdown("## 🎯 Comprehensive Qualitative Analysis")
    
    # Overall score
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Qualitative Score",
            f"{analysis.qualitative_score:.1f}/100",
            help="Overall qualitative assessment score"
        )
    
    with col2:
        moat_score = analysis.moat_analysis.get('moat_score', 0)
        st.metric(
            "Moat Score",
            f"{moat_score:.1f}/100",
            help="Economic moat strength score"
        )
    
    with col3:
        sentiment_score = analysis.sentiment_analysis.get('sentiment_score', 0)
        st.metric(
            "Sentiment Score",
            f"{sentiment_score:.1f}/100",
            help="Market sentiment score"
        )
    
    with col4:
        trends_score = analysis.secular_trends.get('trend_score', 0)
        st.metric(
            "Trends Score",
            f"{trends_score:.1f}/100",
            help="Secular trends alignment score"
        )
    
    # Component scores radar chart
    st.markdown("### 📊 Qualitative Score Breakdown")
    
    categories = ['Moat', 'Sentiment', 'Trends', 'Competitive']
    scores = [
        analysis.moat_analysis.get('moat_score', 0),
        analysis.sentiment_analysis.get('sentiment_score', 0),
        analysis.secular_trends.get('trend_score', 0),
        analysis.competitive_position.get('competitive_score', 0)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Qualitative Scores'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Investment thesis
    st.markdown("### 📝 Investment Thesis")
    st.markdown(analysis.reasoning)
    
    # Citations
    if analysis.citations:
        st.markdown("### 📚 Sources & Citations")
        for i, citation in enumerate(analysis.citations[:10], 1):  # Show first 10 citations
            st.markdown(f"""
            <div class="citation-box">
                {i}. <a href="{citation}" target="_blank">{citation}</a>
            </div>
            """, unsafe_allow_html=True)
    
    # Search queries used
    if analysis.search_queries_used:
        with st.expander("🔍 Search Queries Used", expanded=False):
            for i, query in enumerate(analysis.search_queries_used, 1):
                st.markdown(f"{i}. {query}")

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
        if f.startswith("rationale_agent_trace_") and f.endswith(".json")
    ]
    
    if not trace_files:
        st.info("No rationale agent traces found.")
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
            analysis_data = trace_data.get('analysis', {})
            
            with st.expander(f"📄 {trace_file} - {timestamp}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Analysis Summary:**")
                    st.markdown(f"- Model: {model_info}")
                    st.markdown(f"- Ticker: {analysis_data.get('ticker', 'Unknown')}")
                    st.markdown(f"- Company: {analysis_data.get('company_name', 'Unknown')}")
                    st.markdown(f"- Qualitative Score: {analysis_data.get('qualitative_score', 0):.1f}/100")
                
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

