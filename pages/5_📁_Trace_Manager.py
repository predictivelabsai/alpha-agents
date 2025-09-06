"""
Trace Manager Page - Lohusalu Capital Management
Comprehensive JSON tracing and reasoning trace management interface
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys
from datetime import datetime, timedelta
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.trace_manager import TraceManager, display_trace_manager_interface

# Page configuration
st.set_page_config(
    page_title="Trace Manager - Lohusalu Capital Management",
    page_icon="📁",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #fd7e14, #ffc107);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .trace-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #fd7e14;
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
    .trace-item {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .agent-fundamental { border-left: 4px solid #1f77b4; }
    .agent-rationale { border-left: 4px solid #28a745; }
    .agent-ranker { border-left: 4px solid #6f42c1; }
    .agent-agentic { border-left: 4px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

def main():
    """Main function for Trace Manager page"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📁 Trace Manager</h1>
        <p>Comprehensive JSON Tracing & Reasoning Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System Description
    st.markdown("""
    <div class="trace-card">
        <h3>🎯 Trace Management Capabilities</h3>
        <ul>
            <li><strong>JSON Trace Storage:</strong> Automatic saving of all agent reasoning and analysis</li>
            <li><strong>Expandable Traces:</strong> View detailed reasoning traces as DataFrames and JSON</li>
            <li><strong>Multi-Agent Support:</strong> Traces from Fundamental, Rationale, and Ranker agents</li>
            <li><strong>Search & Analytics:</strong> Comprehensive search, filtering, and analytics tools</li>
            <li><strong>Export & Management:</strong> CSV export, cleanup tools, and trace comparison</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize trace manager
    try:
        trace_manager = TraceManager()
        
        # Display main interface
        display_trace_manager_interface()
        
        # Additional features
        display_advanced_features(trace_manager)
        
    except Exception as e:
        st.error(f"Error initializing Trace Manager: {str(e)}")
        st.exception(e)

def display_advanced_features(trace_manager):
    """Display advanced trace management features"""
    
    st.markdown("## 🔬 Advanced Features")
    
    # Trace comparison
    with st.expander("🔄 Trace Comparison", expanded=False):
        display_trace_comparison(trace_manager)
    
    # Batch operations
    with st.expander("⚡ Batch Operations", expanded=False):
        display_batch_operations(trace_manager)
    
    # Performance analytics
    with st.expander("📊 Performance Analytics", expanded=False):
        display_performance_analytics(trace_manager)

def display_trace_comparison(trace_manager):
    """Display trace comparison interface"""
    
    st.markdown("### Compare Agent Performance Across Traces")
    
    # Get available traces
    all_traces = trace_manager.get_all_traces(limit=100)
    
    if not all_traces:
        st.info("No traces available for comparison.")
        return
    
    # Group traces by agent type
    agent_traces = {}
    for trace in all_traces:
        agent_type = trace.get('agent_type', 'unknown')
        if agent_type not in agent_traces:
            agent_traces[agent_type] = []
        agent_traces[agent_type].append(trace)
    
    # Selection interface
    col1, col2 = st.columns(2)
    
    with col1:
        selected_agent = st.selectbox(
            "Select Agent Type",
            options=list(agent_traces.keys()),
            help="Choose agent type for comparison"
        )
    
    with col2:
        if selected_agent in agent_traces:
            trace_options = [
                f"{trace.get('filename', 'Unknown')} - {trace.get('timestamp', 'Unknown')[:19]}"
                for trace in agent_traces[selected_agent][:20]  # Limit to 20 most recent
            ]
            
            selected_traces = st.multiselect(
                "Select Traces to Compare",
                options=trace_options,
                help="Choose 2 or more traces for comparison"
            )
    
    if len(selected_traces) >= 2:
        if st.button("🔄 Compare Traces", key="compare_traces_btn"):
            # Get trace filenames
            trace_filenames = []
            for selection in selected_traces:
                filename = selection.split(' - ')[0]
                trace_filenames.append(filename)
            
            # Perform comparison
            comparison = trace_manager.compare_traces(trace_filenames)
            
            if 'error' not in comparison:
                display_comparison_results(comparison)
            else:
                st.error(f"Comparison error: {comparison['error']}")

def display_comparison_results(comparison):
    """Display trace comparison results"""
    
    st.markdown("#### 📊 Comparison Results")
    
    # Basic comparison info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Traces Compared", comparison.get('trace_count', 0))
    
    with col2:
        agent_types = comparison.get('agent_types', [])
        st.metric("Agent Types", len(set(agent_types)))
    
    with col3:
        models = comparison.get('models_used', [])
        st.metric("Models Used", len(set(models)))
    
    # Detailed comparison
    if 'average_scores' in comparison:
        st.markdown("#### 📈 Score Comparison")
        
        scores_df = pd.DataFrame({
            'Trace': [f"Trace {i+1}" for i in range(len(comparison['average_scores']))],
            'Average Score': comparison['average_scores']
        })
        
        fig = px.bar(scores_df, x='Trace', y='Average Score', title="Average Scores Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("Score Difference", f"{comparison.get('score_difference', 0):.2f}")
    
    if 'stocks_found' in comparison:
        st.markdown("#### 🔍 Screening Results Comparison")
        
        stocks_df = pd.DataFrame({
            'Trace': [f"Trace {i+1}" for i in range(len(comparison['stocks_found']))],
            'Stocks Found': comparison['stocks_found']
        })
        
        fig = px.bar(stocks_df, x='Trace', y='Stocks Found', title="Stocks Found Comparison")
        st.plotly_chart(fig, use_container_width=True)

def display_batch_operations(trace_manager):
    """Display batch operations interface"""
    
    st.markdown("### Batch Operations on Traces")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📤 Bulk Export")
        
        export_agent = st.selectbox(
            "Agent Type for Bulk Export",
            options=["fundamental", "rationale", "ranker", "agentic_screener"],
            key="bulk_export_agent"
        )
        
        export_format = st.selectbox(
            "Export Format",
            options=["CSV", "JSON", "Excel"],
            key="export_format"
        )
        
        if st.button("📤 Export All Traces", key="export_traces_btn"):
            if export_format == "CSV":
                output_path = trace_manager.export_traces_to_csv(export_agent)
                if output_path:
                    st.success(f"Exported to {output_path}")
                    
                    with open(output_path, 'r') as f:
                        st.download_button(
                            label="Download CSV",
                            data=f.read(),
                            file_name=output_path,
                            mime="text/csv"
                        )
            else:
                st.info(f"{export_format} export coming soon!")
    
    with col2:
        st.markdown("#### 🗑️ Bulk Cleanup")
        
        cleanup_agent = st.selectbox(
            "Agent Type for Cleanup",
            options=["All", "fundamental", "rationale", "ranker", "agentic_screener"],
            key="cleanup_agent"
        )
        
        days_to_keep = st.slider(
            "Days to Keep",
            min_value=1,
            max_value=365,
            value=90,
            key="bulk_cleanup_days"
        )
        
        if st.button("🗑️ Cleanup Old Traces", key="cleanup_traces_btn"):
            deleted_count = trace_manager.cleanup_old_traces(days_to_keep)
            st.success(f"Deleted {deleted_count} old trace files")

def display_performance_analytics(trace_manager):
    """Display performance analytics"""
    
    st.markdown("### 📊 Performance Analytics")
    
    # Time period selection
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_period = st.selectbox(
            "Analysis Period",
            options=["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
            index=1
        )
    
    with col2:
        agent_filter = st.selectbox(
            "Agent Filter",
            options=["All", "fundamental", "rationale", "ranker", "agentic_screener"],
            key="perf_agent_filter"
        )
    
    # Convert period to days
    period_map = {
        "Last 7 days": 7,
        "Last 30 days": 30,
        "Last 90 days": 90,
        "All time": 365 * 10  # 10 years
    }
    days_back = period_map[analysis_period]
    
    # Get analytics
    analytics = trace_manager.get_trace_analytics(
        agent_type=None if agent_filter == "All" else agent_filter,
        days_back=days_back
    )
    
    if not analytics:
        st.info("No analytics data available for the selected period.")
        return
    
    # Display analytics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Traces", analytics.get('total_traces', 0))
    
    with col2:
        agent_count = len(analytics.get('agent_breakdown', {}))
        st.metric("Agent Types", agent_count)
    
    with col3:
        model_count = len(analytics.get('model_usage', {}))
        st.metric("Models Used", model_count)
    
    with col4:
        daily_avg = analytics.get('total_traces', 0) / max(days_back, 1)
        st.metric("Daily Average", f"{daily_avg:.1f}")
    
    # Detailed charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Agent usage over time
        if analytics.get('agent_breakdown'):
            agent_df = pd.DataFrame([
                {'Agent': agent, 'Count': count}
                for agent, count in analytics['agent_breakdown'].items()
            ])
            
            fig = px.pie(agent_df, values='Count', names='Agent', 
                        title="Agent Usage Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Model performance
        if analytics.get('model_usage'):
            model_df = pd.DataFrame([
                {'Model': model.replace('-', ' '), 'Usage': count}
                for model, count in analytics['model_usage'].items()
            ])
            
            fig = px.bar(model_df, x='Model', y='Usage', 
                        title="Model Usage Frequency")
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Activity timeline
    if analytics.get('daily_activity'):
        st.markdown("#### 📅 Activity Timeline")
        
        daily_df = pd.DataFrame([
            {'Date': date, 'Traces': count}
            for date, count in analytics['daily_activity'].items()
        ])
        daily_df['Date'] = pd.to_datetime(daily_df['Date'])
        daily_df = daily_df.sort_values('Date')
        
        fig = px.line(daily_df, x='Date', y='Traces', 
                     title=f"Daily Trace Activity - {analysis_period}")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Activity summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            peak_day = daily_df.loc[daily_df['Traces'].idxmax()]
            st.metric("Peak Day", peak_day['Date'].strftime('%Y-%m-%d'), 
                     f"{peak_day['Traces']} traces")
        
        with col2:
            avg_daily = daily_df['Traces'].mean()
            st.metric("Average Daily", f"{avg_daily:.1f}")
        
        with col3:
            total_days = len(daily_df)
            st.metric("Active Days", total_days)

def display_trace_details(trace_data):
    """Display detailed trace information"""
    
    st.markdown("#### 🔍 Trace Details")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Agent Information:**")
        st.markdown(f"- Type: {trace_data.get('agent_type', 'Unknown')}")
        st.markdown(f"- Version: {trace_data.get('trace_version', 'Unknown')}")
    
    with col2:
        st.markdown("**Model Information:**")
        st.markdown(f"- Provider: {trace_data.get('model_provider', 'Unknown')}")
        st.markdown(f"- Model: {trace_data.get('model_name', 'Unknown')}")
    
    with col3:
        st.markdown("**Execution Information:**")
        st.markdown(f"- Timestamp: {trace_data.get('timestamp', 'Unknown')}")
        st.markdown(f"- File: {trace_data.get('filename', 'Unknown')}")
    
    # Agent-specific details
    agent_type = trace_data.get('agent_type')
    
    if agent_type == 'fundamental':
        display_fundamental_trace_details(trace_data)
    elif agent_type == 'rationale':
        display_rationale_trace_details(trace_data)
    elif agent_type == 'ranker':
        display_ranker_trace_details(trace_data)
    elif agent_type == 'agentic_screener':
        display_agentic_screener_trace_details(trace_data)

def display_fundamental_trace_details(trace_data):
    """Display fundamental agent trace details"""
    
    st.markdown("#### 📊 Fundamental Analysis Details")
    
    sector_analyses = trace_data.get('sector_analyses', [])
    stock_screenings = trace_data.get('stock_screenings', [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Sectors Analyzed", len(sector_analyses))
        
        if sector_analyses:
            st.markdown("**Top Sectors:**")
            for i, sector in enumerate(sector_analyses[:5], 1):
                st.markdown(f"{i}. {sector.get('sector', 'Unknown')} (Weight: {sector.get('weight', 0):.1f})")
    
    with col2:
        st.metric("Stocks Found", len(stock_screenings))
        
        if stock_screenings:
            st.markdown("**Top Stocks:**")
            for i, stock in enumerate(stock_screenings[:5], 1):
                st.markdown(f"{i}. {stock.get('ticker', 'Unknown')} (Score: {stock.get('fundamental_score', 0):.1f})")

def display_rationale_trace_details(trace_data):
    """Display rationale agent trace details"""
    
    st.markdown("#### 🔍 Qualitative Analysis Details")
    
    analysis = trace_data.get('analysis', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Analysis Scores:**")
        st.markdown(f"- Qualitative Score: {analysis.get('qualitative_score', 0):.1f}/100")
        st.markdown(f"- Moat Score: {analysis.get('moat_analysis', {}).get('moat_score', 0):.1f}/100")
        st.markdown(f"- Sentiment Score: {analysis.get('sentiment_analysis', {}).get('sentiment_score', 0):.1f}/100")
    
    with col2:
        st.markdown("**Key Findings:**")
        st.markdown(f"- Moat Strength: {analysis.get('moat_analysis', {}).get('moat_strength', 'Unknown')}")
        st.markdown(f"- Sentiment: {analysis.get('sentiment_analysis', {}).get('overall_sentiment', 'Unknown')}")
        st.markdown(f"- Trend Alignment: {analysis.get('secular_trends', {}).get('trend_alignment', 'Unknown')}")

def display_ranker_trace_details(trace_data):
    """Display ranker agent trace details"""
    
    st.markdown("#### 🎯 Investment Scoring Details")
    
    investment_scores = trace_data.get('investment_scores', [])
    portfolio_rec = trace_data.get('portfolio_recommendation', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Investments Scored", len(investment_scores))
        
        if investment_scores:
            avg_score = sum(score.get('composite_score', 0) for score in investment_scores) / len(investment_scores)
            st.metric("Average Score", f"{avg_score:.1f}/100")
            
            st.markdown("**Top Investments:**")
            for i, score in enumerate(investment_scores[:5], 1):
                st.markdown(f"{i}. {score.get('ticker', 'Unknown')} - {score.get('investment_grade', 'Unknown')} ({score.get('composite_score', 0):.1f})")
    
    with col2:
        if portfolio_rec:
            st.markdown("**Portfolio Recommendation:**")
            st.markdown(f"- Risk Profile: {portfolio_rec.get('risk_profile', 'Unknown')}")
            st.markdown(f"- Expected Return: {portfolio_rec.get('expected_return', 0):.1f}%")
            st.markdown(f"- Diversification: {portfolio_rec.get('diversification_score', 0):.1f}/100")
            st.markdown(f"- Conviction: {portfolio_rec.get('overall_conviction', 'Unknown')}")

def display_agentic_screener_trace_details(trace_data):
    """Display agentic screener trace details"""
    
    st.markdown("#### 🤖 Agentic Screener Pipeline Details")
    
    pipeline_results = trace_data.get('pipeline_results', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Pipeline Execution:**")
        st.markdown(f"- Execution Time: {trace_data.get('execution_time', 'Unknown')}")
        st.markdown(f"- Total Stocks: {len(pipeline_results.get('final_recommendations', []))}")
    
    with col2:
        st.markdown("**Results Summary:**")
        recommendations = pipeline_results.get('final_recommendations', [])
        if recommendations:
            avg_score = sum(rec.get('composite_score', 0) for rec in recommendations) / len(recommendations)
            st.markdown(f"- Average Score: {avg_score:.1f}/100")
            
            sectors = set(rec.get('sector', '') for rec in recommendations)
            st.markdown(f"- Sectors Covered: {len(sectors)}")

if __name__ == "__main__":
    main()

