"""
Monitoring Page - Agent reasoning traces and chain of thought analysis
"""

import streamlit as st
import sys
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import glob
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Page configuration
st.set_page_config(
    page_title="Monitoring - Lohusalu Capital Management",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .monitoring-header {
        background: linear-gradient(90deg, #6f42c1, #e83e8c);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .trace-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #6f42c1;
        margin: 0.5rem 0;
    }
    .agent-trace {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .reasoning-step {
        background-color: #e8f4fd;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        border-left: 3px solid #1f77b4;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .expandable-section {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_log_files() -> List[Dict[str, Any]]:
    """Load all log files from the logs directory"""
    log_files = []
    logs_dir = "logs"
    
    if os.path.exists(logs_dir):
        # Get all JSON log files
        json_files = glob.glob(os.path.join(logs_dir, "*.json"))
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['file_path'] = file_path
                    data['file_name'] = os.path.basename(file_path)
                    log_files.append(data)
            except Exception as e:
                st.error(f"Error loading {file_path}: {e}")
    
    # Sort by timestamp (newest first)
    log_files.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return log_files

def display_agent_reasoning_trace(agent_analysis: Dict[str, Any], agent_name: str):
    """Display detailed reasoning trace for a single agent"""
    
    with st.expander(f"🤖 {agent_name.title()} Agent - Reasoning Trace", expanded=False):
        
        # Agent summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            recommendation = agent_analysis.get('recommendation', 'HOLD')
            rec_color = {
                'STRONG_BUY': 'confidence-high',
                'BUY': 'confidence-high', 
                'HOLD': 'confidence-medium',
                'SELL': 'confidence-low',
                'STRONG_SELL': 'confidence-low'
            }.get(recommendation, 'confidence-medium')
            
            st.markdown(f'<div class="{rec_color}">Recommendation: {recommendation}</div>', unsafe_allow_html=True)
        
        with col2:
            confidence = agent_analysis.get('confidence_score', 0.5)
            conf_color = 'confidence-high' if confidence > 0.7 else 'confidence-medium' if confidence > 0.4 else 'confidence-low'
            st.markdown(f'<div class="{conf_color}">Confidence: {confidence:.2f}</div>', unsafe_allow_html=True)
        
        with col3:
            timestamp = agent_analysis.get('analysis_timestamp', 'Unknown')
            st.markdown(f"**Timestamp:** {timestamp}")
        
        # Reasoning breakdown
        st.markdown("**🧠 Chain of Thought:**")
        
        reasoning = agent_analysis.get('reasoning', 'No detailed reasoning available')
        
        # Try to break down reasoning into steps
        reasoning_steps = reasoning.split('\n') if reasoning else ['No reasoning provided']
        
        for i, step in enumerate(reasoning_steps):
            if step.strip():
                st.markdown(f"""
                <div class="reasoning-step">
                    <strong>Step {i+1}:</strong> {step.strip()}
                </div>
                """, unsafe_allow_html=True)
        
        # Key factors and concerns
        col1, col2 = st.columns(2)
        
        with col1:
            key_factors = agent_analysis.get('key_factors', agent_analysis.get('key_catalysts', []))
            if key_factors:
                st.markdown("**✅ Key Strengths/Catalysts:**")
                for factor in key_factors:
                    st.markdown(f"• {factor}")
        
        with col2:
            concerns = agent_analysis.get('concerns', agent_analysis.get('key_risks', []))
            if concerns:
                st.markdown("**⚠️ Key Risks/Concerns:**")
                for concern in concerns:
                    st.markdown(f"• {concern}")
        
        # Additional metrics if available
        if 'target_price' in agent_analysis and agent_analysis['target_price']:
            st.markdown(f"**🎯 Target Price:** ${agent_analysis['target_price']:.2f}")
        
        if 'risk_assessment' in agent_analysis:
            st.markdown(f"**📊 Risk Assessment:** {agent_analysis['risk_assessment']}")
        
        # Raw data view
        with st.expander("📋 Raw Analysis Data", expanded=False):
            st.json(agent_analysis)

def display_screening_results(log_data: Dict[str, Any]):
    """Display screening results with agent traces"""
    
    st.subheader(f"📊 Screening Results - {log_data.get('timestamp', 'Unknown')}")
    
    # Summary metrics
    results = log_data.get('results', [])
    sectors = log_data.get('sectors_screened', [])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sectors Screened", len(sectors))
    
    with col2:
        st.metric("Stocks Analyzed", len(results))
    
    with col3:
        if results:
            avg_score = sum(r.get('composite_score', 0) for r in results) / len(results)
            st.metric("Avg Composite Score", f"{avg_score:.1f}/100")
    
    with col4:
        if results:
            buy_count = sum(1 for r in results if r.get('recommendation') in ['BUY', 'STRONG_BUY'])
            st.metric("Buy Recommendations", buy_count)
    
    # Sectors screened
    st.markdown("**🎯 Sectors Analyzed:**")
    st.markdown(" | ".join(sectors))
    
    # Top performing stocks
    if results:
        st.subheader("🏆 Top Performing Stocks")
        
        # Sort by composite score
        sorted_results = sorted(results, key=lambda x: x.get('composite_score', 0), reverse=True)
        
        for i, result in enumerate(sorted_results[:5]):
            with st.expander(f"#{i+1}: {result.get('stock_symbol', 'Unknown')} - {result.get('company_name', 'Unknown Company')}", expanded=False):
                
                # Stock summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Composite Score", f"{result.get('composite_score', 0)}/100")
                    st.metric("Recommendation", result.get('recommendation', 'HOLD'))
                
                with col2:
                    st.metric("Confidence", f"{result.get('confidence_score', 0):.2f}")
                    st.metric("Sector", result.get('sector', 'Unknown'))
                
                with col3:
                    market_cap = result.get('market_cap', 0)
                    if market_cap > 0:
                        st.metric("Market Cap", f"${market_cap/1e9:.1f}B")
                    pe_ratio = result.get('pe_ratio', 0)
                    if pe_ratio > 0:
                        st.metric("P/E Ratio", f"{pe_ratio:.1f}")
                
                # Investment thesis
                thesis = result.get('investment_thesis', 'Multi-agent analysis completed')
                st.markdown(f"**💡 Investment Thesis:** {thesis}")
                
                # Agent breakdown with reasoning traces
                agent_analyses = result.get('agent_analyses', {})
                if agent_analyses:
                    st.markdown("**🤖 Individual Agent Analysis:**")
                    
                    for agent_name, analysis in agent_analyses.items():
                        display_agent_reasoning_trace(analysis, agent_name)

def display_methodology_results(log_data: Dict[str, Any]):
    """Display methodology test results with detailed traces"""
    
    st.subheader(f"🧪 Methodology Test Results - {log_data.get('timestamp', 'Unknown')}")
    
    results = log_data.get('results', [])
    
    if results:
        # Create summary table
        summary_data = []
        for result in results:
            summary_data.append({
                'Stock': result.get('stock_symbol', 'Unknown'),
                'Agent': result.get('agent_type', 'Unknown'),
                'Recommendation': result.get('recommendation', 'HOLD'),
                'Confidence': result.get('confidence_score', 0),
                'Reasoning Length': len(result.get('reasoning', ''))
            })
        
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)
        
        # Detailed traces
        st.subheader("🔍 Detailed Agent Traces")
        
        # Group by stock
        stocks = {}
        for result in results:
            stock = result.get('stock_symbol', 'Unknown')
            if stock not in stocks:
                stocks[stock] = []
            stocks[stock].append(result)
        
        for stock, stock_results in stocks.items():
            with st.expander(f"📈 {stock} - Agent Analysis Traces", expanded=False):
                
                for result in stock_results:
                    agent_type = result.get('agent_type', 'Unknown Agent')
                    display_agent_reasoning_trace(result, agent_type)

def main():
    """Main Monitoring page"""
    
    # Header
    st.markdown("""
    <div class="monitoring-header">
        <h1>🔍 Agent Monitoring & Reasoning Traces</h1>
        <p>Deep dive into agent decision-making processes and chain of thought analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load log files
    log_files = load_log_files()
    
    if not log_files:
        st.warning("No log files found. Run some analyses first to generate monitoring data.")
        st.info("💡 **Tip:** Use the Stock Screener or Portfolio Construction pages to generate analysis logs.")
        return
    
    # Sidebar for log file selection
    st.sidebar.title("📋 Log File Selection")
    
    # Group log files by type
    screening_logs = [log for log in log_files if 'screening_results' in log.get('file_name', '')]
    methodology_logs = [log for log in log_files if 'methodology' in log.get('file_name', '')]
    other_logs = [log for log in log_files if log not in screening_logs and log not in methodology_logs]
    
    log_type = st.sidebar.selectbox(
        "Log Type",
        ["Screening Results", "Methodology Tests", "Other Logs"],
        help="Select the type of logs to monitor"
    )
    
    # Select specific log file
    if log_type == "Screening Results" and screening_logs:
        selected_logs = screening_logs
        log_display_names = [f"Screening - {log.get('timestamp', 'Unknown')}" for log in screening_logs]
    elif log_type == "Methodology Tests" and methodology_logs:
        selected_logs = methodology_logs
        log_display_names = [f"Methodology - {log.get('timestamp', 'Unknown')}" for log in methodology_logs]
    elif log_type == "Other Logs" and other_logs:
        selected_logs = other_logs
        log_display_names = [f"Other - {log.get('file_name', 'Unknown')}" for log in other_logs]
    else:
        selected_logs = []
        log_display_names = []
    
    if not selected_logs:
        st.warning(f"No {log_type.lower()} found.")
        return
    
    selected_log_name = st.sidebar.selectbox(
        "Select Log File",
        log_display_names,
        help="Choose a specific log file to analyze"
    )
    
    # Get selected log data
    selected_index = log_display_names.index(selected_log_name)
    selected_log = selected_logs[selected_index]
    
    # Display log file info
    st.subheader("📄 Log File Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("File Name", selected_log.get('file_name', 'Unknown'))
    
    with col2:
        timestamp = selected_log.get('timestamp', 'Unknown')
        if timestamp != 'Unknown':
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                st.metric("Timestamp", formatted_time)
            except:
                st.metric("Timestamp", timestamp)
        else:
            st.metric("Timestamp", "Unknown")
    
    with col3:
        file_size = os.path.getsize(selected_log['file_path']) if os.path.exists(selected_log['file_path']) else 0
        st.metric("File Size", f"{file_size/1024:.1f} KB")
    
    # Display content based on log type
    st.markdown("---")
    
    if 'screening_results' in selected_log.get('file_name', ''):
        display_screening_results(selected_log)
    elif 'methodology' in selected_log.get('file_name', ''):
        display_methodology_results(selected_log)
    else:
        # Generic log display
        st.subheader("📋 Log Content")
        
        # Try to display structured content
        if 'results' in selected_log:
            results = selected_log['results']
            if isinstance(results, list) and results:
                st.json(results[0])  # Show first result as example
                
                if len(results) > 1:
                    with st.expander(f"View all {len(results)} results", expanded=False):
                        st.json(results)
        else:
            st.json(selected_log)
    
    # Log file management
    st.markdown("---")
    st.subheader("🗂️ Log File Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Delete Selected Log", type="secondary"):
            try:
                os.remove(selected_log['file_path'])
                st.success(f"Deleted {selected_log['file_name']}")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error deleting file: {e}")
    
    with col2:
        if st.button("📥 Download Selected Log", type="secondary"):
            try:
                with open(selected_log['file_path'], 'r') as f:
                    log_content = f.read()
                
                st.download_button(
                    label="💾 Download JSON",
                    data=log_content,
                    file_name=selected_log['file_name'],
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Error preparing download: {e}")
    
    # System monitoring stats
    st.markdown("---")
    st.subheader("📊 System Monitoring Statistics")
    
    # Aggregate stats from all log files
    total_analyses = 0
    total_stocks = set()
    agent_performance = {}
    
    for log in log_files:
        results = log.get('results', [])
        if isinstance(results, list):
            total_analyses += len(results)
            
            for result in results:
                stock = result.get('stock_symbol')
                if stock:
                    total_stocks.add(stock)
                
                agent_type = result.get('agent_type', 'Unknown')
                if agent_type not in agent_performance:
                    agent_performance[agent_type] = []
                
                confidence = result.get('confidence_score', 0)
                agent_performance[agent_type].append(confidence)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Log Files", len(log_files))
    
    with col2:
        st.metric("Total Analyses", total_analyses)
    
    with col3:
        st.metric("Unique Stocks", len(total_stocks))
    
    with col4:
        if agent_performance:
            avg_confidence = sum(sum(scores) for scores in agent_performance.values()) / sum(len(scores) for scores in agent_performance.values())
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    # Agent performance chart
    if agent_performance:
        st.subheader("📈 Agent Performance Over Time")
        
        # Create performance dataframe
        perf_data = []
        for agent, scores in agent_performance.items():
            for score in scores:
                perf_data.append({
                    'Agent': agent.title(),
                    'Confidence Score': score
                })
        
        if perf_data:
            df_perf = pd.DataFrame(perf_data)
            
            fig = px.box(
                df_perf,
                x='Agent',
                y='Confidence Score',
                title="Agent Confidence Score Distribution",
                color='Agent'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

