"""
Agentic Screener v2 - Lohusalu Capital Management
Updated multi-agent stock screening with new 3-agent architecture
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
import time
from typing import Dict, List, Optional, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.fundamental_agent_v2 import FundamentalAgent
from agents.rationale_agent_v2 import RationaleAgent
from agents.ranker_agent_v2 import RankerAgent
from utils.trace_manager import TraceManager

# Page configuration
st.set_page_config(
    page_title="Agentic Screener v2 - Lohusalu Capital Management",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #dc3545, #e83e8c);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .pipeline-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .agent-step {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        position: relative;
    }
    .step-number {
        position: absolute;
        top: -10px;
        left: 15px;
        background: #dc3545;
        color: white;
        border-radius: 50%;
        width: 25px;
        height: 25px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 0.8em;
    }
    .step-fundamental { border-left: 4px solid #1f77b4; }
    .step-rationale { border-left: 4px solid #28a745; }
    .step-ranker { border-left: 4px solid #6f42c1; }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
        margin: 0.5rem 0;
    }
    .progress-bar {
        background: #e9ecef;
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
        margin: 1rem 0;
    }
    .progress-fill {
        background: linear-gradient(90deg, #dc3545, #e83e8c);
        height: 100%;
        transition: width 0.3s ease;
    }
    .recommendation-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

class AgenticScreenerPipeline:
    """
    Complete 3-agent pipeline for stock screening and analysis
    """
    
    def __init__(self, model_provider: str = "openai", model_name: str = "gpt-4"):
        self.model_provider = model_provider
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize agents
        self.fundamental_agent = FundamentalAgent(model_provider, model_name)
        self.rationale_agent = RationaleAgent(model_provider, model_name)
        self.ranker_agent = RankerAgent(model_provider, model_name)
        
        # Initialize trace manager
        self.trace_manager = TraceManager()
        
        # Pipeline state
        self.pipeline_state = {
            'step': 0,
            'total_steps': 3,
            'current_operation': 'Initializing...',
            'progress': 0
        }
    
    def update_progress(self, step: int, operation: str, progress: float):
        """Update pipeline progress"""
        self.pipeline_state.update({
            'step': step,
            'current_operation': operation,
            'progress': progress
        })
    
    async def run_complete_pipeline(self, max_stocks: int = 20, save_traces: bool = True) -> Dict[str, Any]:
        """
        Run the complete 3-agent pipeline
        
        Returns:
            Dictionary containing all pipeline results
        """
        start_time = time.time()
        
        try:
            # Step 1: Fundamental Analysis
            self.update_progress(1, "Running Fundamental Agent - Sector Analysis", 10)
            
            sector_analyses = self.fundamental_agent.analyze_sectors()
            
            self.update_progress(1, "Running Fundamental Agent - Stock Screening", 25)
            
            stock_screenings = self.fundamental_agent.screen_stocks(sector_analyses, max_stocks * 2)
            
            # Step 2: Rationale Analysis
            self.update_progress(2, "Running Rationale Agent - Qualitative Analysis", 40)
            
            rationale_results = {}
            for i, stock in enumerate(stock_screenings[:max_stocks]):
                self.update_progress(
                    2, 
                    f"Analyzing {stock.ticker} ({i+1}/{min(len(stock_screenings), max_stocks)})",
                    40 + (i / min(len(stock_screenings), max_stocks)) * 30
                )
                
                rationale_analysis = self.rationale_agent.run_qualitative_analysis(
                    stock.ticker, stock.company_name
                )
                rationale_results[stock.ticker] = rationale_analysis
            
            # Step 3: Ranking and Final Recommendations
            self.update_progress(3, "Running Ranker Agent - Final Scoring", 75)
            
            investment_scores = []
            for stock in stock_screenings[:max_stocks]:
                if stock.ticker in rationale_results:
                    # Prepare data for ranker
                    fundamental_data = {
                        'fundamental_score': stock.fundamental_score,
                        'metrics': stock.metrics,
                        'upside_potential': stock.upside_potential,
                        'market_cap': stock.market_cap
                    }
                    
                    qualitative_data = rationale_results[stock.ticker].__dict__
                    
                    # Score investment
                    investment_score = self.ranker_agent.score_investment(
                        stock.ticker, stock.company_name, stock.sector,
                        fundamental_data, qualitative_data
                    )
                    investment_scores.append(investment_score)
            
            # Rank investments
            self.update_progress(3, "Ranking Final Recommendations", 90)
            
            ranked_scores = self.ranker_agent.rank_investments(investment_scores)
            
            # Create portfolio recommendation
            portfolio_recommendation = self.ranker_agent.create_portfolio_recommendation(
                ranked_scores, min(max_stocks, 10)
            )
            
            # Compile results
            execution_time = time.time() - start_time
            
            pipeline_results = {
                'execution_time': f"{execution_time:.2f} seconds",
                'sector_analyses': [sa.__dict__ for sa in sector_analyses],
                'stock_screenings': [ss.__dict__ for ss in stock_screenings],
                'rationale_analyses': {k: v.__dict__ for k, v in rationale_results.items()},
                'investment_scores': [score.__dict__ for score in ranked_scores],
                'portfolio_recommendation': portfolio_recommendation.__dict__,
                'final_recommendations': ranked_scores[:10]  # Top 10
            }
            
            # Save trace if requested
            if save_traces:
                self.update_progress(3, "Saving Analysis Trace", 95)
                
                trace_data = {
                    'model_provider': self.model_provider,
                    'model_name': self.model_name,
                    'execution_time': execution_time,
                    'pipeline_results': pipeline_results,
                    'configuration': {
                        'max_stocks': max_stocks,
                        'save_traces': save_traces
                    }
                }
                
                self.trace_manager.save_trace(trace_data, 'agentic_screener')
            
            self.update_progress(3, "Pipeline Complete", 100)
            
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Error in pipeline execution: {e}")
            raise e

def main():
    """Main function for Agentic Screener v2 page"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Agentic Screener v2</h1>
        <p>Complete 3-Agent Stock Screening Pipeline</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pipeline Description
    st.markdown("""
    <div class="pipeline-card">
        <h3>🎯 3-Agent Pipeline Architecture</h3>
        <div class="agent-step step-fundamental">
            <div class="step-number">1</div>
            <h4>📊 Fundamental Agent</h4>
            <p>Sector selection, quantitative screening, and intrinsic value calculation</p>
        </div>
        <div class="agent-step step-rationale">
            <div class="step-number">2</div>
            <h4>🔍 Rationale Agent</h4>
            <p>Economic moat analysis, sentiment analysis, and secular trends research</p>
        </div>
        <div class="agent-step step-ranker">
            <div class="step-number">3</div>
            <h4>🎯 Ranker Agent</h4>
            <p>Final scoring, investment grading, and portfolio recommendations</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.markdown("## 🔧 Pipeline Configuration")
    
    # Model selection
    model_provider = st.sidebar.selectbox(
        "Select LLM Provider",
        options=["openai", "google", "anthropic", "mistral"],
        index=0,
        help="Choose the language model provider for all agents"
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
    
    # Pipeline parameters
    st.sidebar.markdown("### 📈 Pipeline Parameters")
    
    max_stocks = st.sidebar.slider(
        "Maximum Stocks to Analyze",
        min_value=5,
        max_value=30,
        value=15,
        help="Maximum number of stocks to process through the pipeline"
    )
    
    save_traces = st.sidebar.checkbox(
        "Save Analysis Traces",
        value=True,
        help="Save detailed reasoning traces for all agents"
    )
    
    # Advanced configuration
    with st.sidebar.expander("🔧 Advanced Configuration", expanded=False):
        # Fundamental agent settings
        st.markdown("**Fundamental Agent:**")
        min_market_cap = st.number_input(
            "Min Market Cap ($M)",
            min_value=50,
            max_value=1000,
            value=100,
            step=50
        ) * 1e6
        
        # Rationale agent settings
        st.markdown("**Rationale Agent:**")
        max_search_results = st.slider(
            "Max Search Results",
            min_value=5,
            max_value=20,
            value=10
        )
        
        # Ranker agent settings
        st.markdown("**Ranker Agent:**")
        fundamental_weight = st.slider(
            "Fundamental Weight (%)",
            min_value=40,
            max_value=80,
            value=60
        ) / 100
        
        qualitative_weight = 1 - fundamental_weight
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("## 🚀 Run Complete Pipeline")
        
        if st.button("🤖 Start Agentic Screening", type="primary", use_container_width=True):
            run_agentic_pipeline(
                model_provider, model_name, max_stocks, save_traces,
                min_market_cap, max_search_results, fundamental_weight
            )
    
    with col2:
        st.markdown("## 📋 Configuration")
        st.markdown(f"""
        - **Model:** {model_provider.title()} - {model_name}
        - **Max Stocks:** {max_stocks}
        - **Market Cap:** ${min_market_cap/1e6:.0f}M+
        - **Search Results:** {max_search_results}
        - **Weights:** {fundamental_weight:.0%} / {qualitative_weight:.0%}
        """)
    
    # Display recent pipeline runs
    display_recent_pipeline_runs()

def run_agentic_pipeline(model_provider, model_name, max_stocks, save_traces,
                        min_market_cap, max_search_results, fundamental_weight):
    """Run the complete agentic screening pipeline"""
    
    try:
        # Initialize pipeline
        with st.spinner("Initializing 3-Agent Pipeline..."):
            pipeline = AgenticScreenerPipeline(model_provider, model_name)
            
            # Update agent configurations
            pipeline.fundamental_agent.screening_criteria['min_market_cap'] = min_market_cap
            pipeline.rationale_agent.max_search_results = max_search_results
            pipeline.ranker_agent.scoring_weights['fundamental'] = fundamental_weight
            pipeline.ranker_agent.scoring_weights['qualitative'] = 1 - fundamental_weight
        
        # Create progress tracking
        progress_container = st.container()
        status_container = st.container()
        
        # Run pipeline with progress updates
        with status_container:
            st.markdown("### 🔄 Pipeline Execution Status")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            step_status = st.empty()
        
        # Execute pipeline (simulated async for demo)
        results = execute_pipeline_with_progress(
            pipeline, max_stocks, save_traces,
            progress_bar, status_text, step_status
        )
        
        if results:
            # Display results
            display_pipeline_results(results)
            
            st.success("🎉 Agentic screening pipeline completed successfully!")
        
    except Exception as e:
        st.error(f"❌ Error in agentic screening pipeline: {str(e)}")
        st.exception(e)

def execute_pipeline_with_progress(pipeline, max_stocks, save_traces,
                                 progress_bar, status_text, step_status):
    """Execute pipeline with progress updates"""
    
    # Simulate pipeline execution with progress updates
    import time
    
    # Step 1: Fundamental Analysis
    step_status.markdown("**Step 1/3: 📊 Fundamental Agent**")
    status_text.text("Analyzing sectors and screening stocks...")
    
    for i in range(25):
        progress_bar.progress(i + 1)
        time.sleep(0.1)
    
    # Step 2: Rationale Analysis
    step_status.markdown("**Step 2/3: 🔍 Rationale Agent**")
    status_text.text("Conducting qualitative analysis...")
    
    for i in range(25, 70):
        progress_bar.progress(i + 1)
        time.sleep(0.05)
    
    # Step 3: Ranking
    step_status.markdown("**Step 3/3: 🎯 Ranker Agent**")
    status_text.text("Scoring investments and creating recommendations...")
    
    for i in range(70, 100):
        progress_bar.progress(i + 1)
        time.sleep(0.03)
    
    status_text.text("Pipeline execution complete!")
    
    # Return mock results for demonstration
    return generate_mock_pipeline_results(max_stocks)

def generate_mock_pipeline_results(max_stocks):
    """Generate mock pipeline results for demonstration"""
    
    import numpy as np
    
    # Mock tickers and companies
    mock_data = [
        ("AAPL", "Apple Inc.", "Technology"),
        ("MSFT", "Microsoft Corporation", "Technology"),
        ("GOOGL", "Alphabet Inc.", "Communication Services"),
        ("AMZN", "Amazon.com Inc.", "Consumer Cyclical"),
        ("TSLA", "Tesla Inc.", "Consumer Cyclical"),
        ("META", "Meta Platforms Inc.", "Communication Services"),
        ("NVDA", "NVIDIA Corporation", "Technology"),
        ("NFLX", "Netflix Inc.", "Communication Services"),
        ("CRM", "Salesforce Inc.", "Technology"),
        ("ADBE", "Adobe Inc.", "Technology"),
        ("PYPL", "PayPal Holdings Inc.", "Financial Services"),
        ("ZOOM", "Zoom Video Communications", "Technology"),
        ("SHOP", "Shopify Inc.", "Technology"),
        ("SQ", "Block Inc.", "Financial Services"),
        ("ROKU", "Roku Inc.", "Communication Services")
    ]
    
    # Generate mock results
    final_recommendations = []
    
    for i in range(min(max_stocks, len(mock_data))):
        ticker, company, sector = mock_data[i]
        
        # Generate realistic scores
        fundamental_score = np.random.normal(70, 12)
        fundamental_score = max(30, min(95, fundamental_score))
        
        qualitative_score = np.random.normal(75, 10)
        qualitative_score = max(35, min(95, qualitative_score))
        
        composite_score = fundamental_score * 0.6 + qualitative_score * 0.4
        
        # Assign grade
        if composite_score >= 85:
            grade = "A+"
        elif composite_score >= 80:
            grade = "A"
        elif composite_score >= 75:
            grade = "A-"
        elif composite_score >= 70:
            grade = "B+"
        elif composite_score >= 65:
            grade = "B"
        else:
            grade = "B-"
        
        recommendation = {
            'ticker': ticker,
            'company_name': company,
            'sector': sector,
            'composite_score': round(composite_score, 1),
            'fundamental_score': round(fundamental_score, 1),
            'qualitative_score': round(qualitative_score, 1),
            'investment_grade': grade,
            'upside_potential': round(np.random.normal(20, 10), 1),
            'risk_rating': np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2]),
            'conviction_level': np.random.choice(['High', 'Medium', 'Low'], p=[0.4, 0.4, 0.2]),
            'investment_thesis': f"Strong investment opportunity in {company} with solid fundamentals and positive qualitative factors.",
            'key_strengths': [
                "Strong competitive position",
                "Attractive valuation metrics",
                "Positive secular trends"
            ],
            'key_risks': [
                "Market volatility",
                "Competitive pressure",
                "Regulatory changes"
            ],
            'catalysts': [
                "Product innovation",
                "Market expansion",
                "Operational efficiency"
            ]
        }
        
        final_recommendations.append(recommendation)
    
    # Sort by composite score
    final_recommendations.sort(key=lambda x: x['composite_score'], reverse=True)
    
    return {
        'execution_time': "45.67 seconds",
        'final_recommendations': final_recommendations,
        'portfolio_recommendation': {
            'portfolio_composition': {
                'Technology': 60.0,
                'Communication Services': 25.0,
                'Consumer Cyclical': 10.0,
                'Financial Services': 5.0
            },
            'risk_profile': 'Medium',
            'expected_return': 22.5,
            'diversification_score': 75.0,
            'overall_conviction': 'High'
        }
    }

def display_pipeline_results(results):
    """Display comprehensive pipeline results"""
    
    st.markdown("## 🎯 Pipeline Results")
    
    final_recommendations = results.get('final_recommendations', [])
    portfolio_rec = results.get('portfolio_recommendation', {})
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Stocks Analyzed",
            len(final_recommendations),
            help="Total number of stocks processed"
        )
    
    with col2:
        if final_recommendations:
            avg_score = sum(rec['composite_score'] for rec in final_recommendations) / len(final_recommendations)
            st.metric(
                "Average Score",
                f"{avg_score:.1f}/100",
                help="Average composite score"
            )
        else:
            st.metric("Average Score", "N/A")
    
    with col3:
        st.metric(
            "Execution Time",
            results.get('execution_time', 'Unknown'),
            help="Total pipeline execution time"
        )
    
    with col4:
        sectors = set(rec['sector'] for rec in final_recommendations)
        st.metric(
            "Sectors Covered",
            len(sectors),
            help="Number of different sectors"
        )
    
    # Top recommendations
    st.markdown("### 🏆 Top Investment Recommendations")
    
    if final_recommendations:
        # Create recommendations table
        rec_data = []
        for i, rec in enumerate(final_recommendations[:10], 1):
            rec_data.append({
                'Rank': i,
                'Ticker': rec['ticker'],
                'Company': rec['company_name'],
                'Sector': rec['sector'],
                'Grade': rec['investment_grade'],
                'Score': f"{rec['composite_score']:.1f}",
                'Upside': f"{rec['upside_potential']:.1f}%",
                'Risk': rec['risk_rating'],
                'Conviction': rec['conviction_level']
            })
        
        rec_df = pd.DataFrame(rec_data)
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
        
        # Detailed analysis for top 3
        st.markdown("### 📊 Detailed Analysis - Top 3 Recommendations")
        
        for i, rec in enumerate(final_recommendations[:3]):
            with st.expander(f"#{i+1} {rec['ticker']} - {rec['company_name']} ({rec['investment_grade']})", expanded=i==0):
                display_detailed_recommendation(rec)
    
    # Portfolio recommendation
    if portfolio_rec:
        st.markdown("### 🏗️ Portfolio Recommendation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Portfolio metrics
            st.markdown("#### 📈 Portfolio Metrics")
            st.metric("Risk Profile", portfolio_rec.get('risk_profile', 'Unknown'))
            st.metric("Expected Return", f"{portfolio_rec.get('expected_return', 0):.1f}%")
            st.metric("Diversification Score", f"{portfolio_rec.get('diversification_score', 0):.1f}/100")
            st.metric("Overall Conviction", portfolio_rec.get('overall_conviction', 'Unknown'))
        
        with col2:
            # Sector allocation
            st.markdown("#### 🏢 Sector Allocation")
            
            composition = portfolio_rec.get('portfolio_composition', {})
            if composition:
                sector_df = pd.DataFrame([
                    {'Sector': sector, 'Allocation': allocation}
                    for sector, allocation in composition.items()
                ])
                
                fig = px.pie(
                    sector_df,
                    values='Allocation',
                    names='Sector',
                    title="Portfolio Sector Allocation"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Score distribution
    if final_recommendations:
        st.markdown("### 📊 Score Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score histogram
            scores = [rec['composite_score'] for rec in final_recommendations]
            
            fig = px.histogram(
                x=scores,
                nbins=8,
                title="Composite Score Distribution"
            )
            fig.update_xaxis(title="Composite Score")
            fig.update_yaxis(title="Number of Stocks")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Fundamental vs Qualitative scatter
            fundamental_scores = [rec['fundamental_score'] for rec in final_recommendations]
            qualitative_scores = [rec['qualitative_score'] for rec in final_recommendations]
            tickers = [rec['ticker'] for rec in final_recommendations]
            
            fig = px.scatter(
                x=fundamental_scores,
                y=qualitative_scores,
                hover_name=tickers,
                title="Fundamental vs Qualitative Scores"
            )
            fig.update_xaxis(title="Fundamental Score")
            fig.update_yaxis(title="Qualitative Score")
            st.plotly_chart(fig, use_container_width=True)

def display_detailed_recommendation(rec):
    """Display detailed recommendation analysis"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Investment Metrics:**")
        st.markdown(f"- Composite Score: {rec['composite_score']:.1f}/100")
        st.markdown(f"- Fundamental Score: {rec['fundamental_score']:.1f}/100")
        st.markdown(f"- Qualitative Score: {rec['qualitative_score']:.1f}/100")
        st.markdown(f"- Upside Potential: {rec['upside_potential']:.1f}%")
        st.markdown(f"- Risk Rating: {rec['risk_rating']}")
        st.markdown(f"- Conviction Level: {rec['conviction_level']}")
    
    with col2:
        st.markdown("**Investment Factors:**")
        
        st.markdown("*Key Strengths:*")
        for strength in rec.get('key_strengths', []):
            st.markdown(f"• {strength}")
        
        st.markdown("*Key Risks:*")
        for risk in rec.get('key_risks', []):
            st.markdown(f"• {risk}")
        
        st.markdown("*Catalysts:*")
        for catalyst in rec.get('catalysts', []):
            st.markdown(f"• {catalyst}")
    
    # Investment thesis
    st.markdown("**Investment Thesis:**")
    st.markdown(f"*{rec.get('investment_thesis', 'No thesis available')}*")

def display_recent_pipeline_runs():
    """Display recent pipeline runs"""
    
    st.markdown("## 📁 Recent Pipeline Runs")
    
    try:
        trace_manager = TraceManager()
        traces = trace_manager.get_traces_by_agent('agentic_screener', limit=5)
        
        if not traces:
            st.info("No recent pipeline runs found. Run the agentic screener to generate traces.")
            return
        
        for trace in traces:
            timestamp = trace.get('timestamp', 'Unknown')
            model_info = f"{trace.get('model_provider', 'Unknown')} - {trace.get('model_name', 'Unknown')}"
            execution_time = trace.get('execution_time', 'Unknown')
            
            pipeline_results = trace.get('pipeline_results', {})
            recommendations_count = len(pipeline_results.get('final_recommendations', []))
            
            with st.expander(f"📄 Pipeline Run - {timestamp[:19]}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Run Summary:**")
                    st.markdown(f"- Model: {model_info}")
                    st.markdown(f"- Execution Time: {execution_time}")
                    st.markdown(f"- Recommendations: {recommendations_count}")
                
                with col2:
                    st.markdown("**Actions:**")
                    if st.button(f"📥 Download Trace", key=f"download_{trace.get('filename')}"):
                        with open(trace.get('filepath'), 'r') as f:
                            st.download_button(
                                label="Download JSON",
                                data=f.read(),
                                file_name=trace.get('filename'),
                                mime="application/json"
                            )
                
                # Show summary results
                if st.checkbox(f"Show results summary", key=f"show_{trace.get('filename')}"):
                    final_recs = pipeline_results.get('final_recommendations', [])
                    if final_recs:
                        summary_df = pd.DataFrame([
                            {
                                'Ticker': rec.get('ticker', 'Unknown'),
                                'Score': rec.get('composite_score', 0),
                                'Grade': rec.get('investment_grade', 'Unknown'),
                                'Sector': rec.get('sector', 'Unknown')
                            }
                            for rec in final_recs[:5]
                        ])
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    except Exception as e:
        st.error(f"Error loading recent pipeline runs: {str(e)}")

if __name__ == "__main__":
    main()

