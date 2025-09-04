"""
Analytics Page - Advanced visualizations and performance analysis
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

from visualizations import AlphaAgentsVisualizer, load_test_data

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Analytics - Alpha Agents",
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
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Analytics page main function"""
    
    st.markdown('<h1 class="main-header">📊 Advanced Analytics & Visualizations</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Explore comprehensive analytics and visualizations of the Alpha Agents system performance.
    These charts provide insights into agent behavior, consensus patterns, risk assessments, and portfolio optimization.
    """)
    
    # Load test data
    test_data_dir = os.path.join(os.path.dirname(__file__), "..", "test-data")
    
    if os.path.exists(test_data_dir):
        # Find the most recent test data file
        csv_files = [f for f in os.listdir(test_data_dir) if f.startswith('agent_analysis_data_') and f.endswith('.csv')]
        
        if csv_files:
            # Sort by filename (which includes timestamp) and get the most recent
            latest_file = sorted(csv_files)[-1]
            file_path = os.path.join(test_data_dir, latest_file)
            
            st.info(f"📊 Loading data from: {latest_file}")
            
            # Load and display visualizations
            analysis_data = load_test_data(file_path)
            
            if analysis_data:
                visualizer = AlphaAgentsVisualizer()
                
                # Data overview
                df = pd.DataFrame(analysis_data)
                st.subheader("📈 Data Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Analyses", len(analysis_data))
                with col2:
                    st.metric("Unique Stocks", df['stock_symbol'].nunique())
                with col3:
                    st.metric("Agents", df['agent'].nunique())
                with col4:
                    avg_confidence = df['confidence_score'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                
                # Create tabs for different visualization categories
                tab1, tab2, tab3, tab4 = st.tabs(["🤖 Agent Analysis", "📈 Performance", "🎯 Portfolio", "📊 Summary"])
                
                with tab1:
                    st.subheader("Agent Consensus & Risk Analysis")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(visualizer.create_agent_consensus_heatmap(analysis_data), use_container_width=True)
                    with col2:
                        st.plotly_chart(visualizer.create_risk_assessment_heatmap(analysis_data), use_container_width=True)
                    
                    st.plotly_chart(visualizer.create_agent_performance_radar(analysis_data), use_container_width=True)
                
                with tab2:
                    st.subheader("Performance Metrics")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(visualizer.create_confidence_distribution_chart(analysis_data), use_container_width=True)
                    with col2:
                        st.plotly_chart(visualizer.create_recommendation_distribution_pie(analysis_data), use_container_width=True)
                
                with tab3:
                    st.subheader("Portfolio Analysis")
                    
                    st.plotly_chart(visualizer.create_portfolio_optimization_chart(analysis_data), use_container_width=True)
                    st.plotly_chart(visualizer.create_sector_analysis_chart(analysis_data), use_container_width=True)
                    
                    # Stock comparison
                    st.subheader("Individual Stock Analysis")
                    available_stocks = sorted(df['stock_symbol'].unique())
                    selected_stock = st.selectbox("Select Stock for Detailed Analysis", available_stocks)
                    
                    if selected_stock:
                        st.plotly_chart(visualizer.create_stock_analysis_comparison(analysis_data, selected_stock), use_container_width=True)
                
                with tab4:
                    st.subheader("Performance Summary Table")
                    summary_df = visualizer.create_performance_summary_table(analysis_data)
                    if not summary_df.empty:
                        st.dataframe(summary_df, use_container_width=True)
                    
                    # Raw data preview
                    st.subheader("Raw Analysis Data")
                    with st.expander("View Raw Data"):
                        st.dataframe(pd.DataFrame(analysis_data), use_container_width=True)
                        
                        # Download button
                        csv_data = pd.DataFrame(analysis_data).to_csv(index=False)
                        st.download_button(
                            label="📥 Download Analysis Data (CSV)",
                            data=csv_data,
                            file_name=f"alpha_agents_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            else:
                st.error("Failed to load analysis data from the CSV file.")
        else:
            st.warning("No test data files found. Please run the test suite first to generate analysis data.")
            
            if st.button("🧪 Run Test Suite"):
                with st.spinner("Running comprehensive test suite..."):
                    # Run the test suite
                    import subprocess
                    result = subprocess.run(
                        ["python3", "tests/run_all_tests.py"],
                        cwd=os.path.join(os.path.dirname(__file__), ".."),
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        st.success("✅ Test suite completed successfully!")
                        st.rerun()  # Refresh the page to load new data
                    else:
                        st.error(f"❌ Test suite failed: {result.stderr}")
    else:
        st.error("Test data directory not found. Please ensure the application is properly set up.")

if __name__ == "__main__":
    main()

