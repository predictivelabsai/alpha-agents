"""
QualAgent Analysis Page for Alpha Agents
Performs AI-powered qualitative analysis on companies
"""

import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
import json
import subprocess
from datetime import datetime

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'agents', 'QualAgent'))

def qualagent_page():
    """QualAgent Analysis page"""
    
    st.title("🤖 QualAgent Analysis")
    st.markdown("AI-powered qualitative analysis using multiple LLM models")
    
    # Sidebar configuration
    st.sidebar.header("🔧 Analysis Configuration")
    
    # Analysis type
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["quick", "comprehensive", "expert-guided"],
        help="Quick: Fast analysis (~30s), Comprehensive: Detailed analysis (~3min), Expert-guided: With human feedback (~10min)"
    )
    
    # User ID
    user_id = st.sidebar.text_input("User ID", value="streamlit_user", help="Unique identifier for this analysis session")
    
    # Data source selection
    st.sidebar.header("📊 Data Source")
    data_source = st.sidebar.radio(
        "Choose Data Source",
        ["From Screener Results", "Upload CSV", "Manual Input", "Database Query"]
    )
    
    companies_to_analyze = []
    
    if data_source == "From Screener Results":
        if "screener_results" in st.session_state and not st.session_state["screener_results"].empty:
            df = st.session_state["screener_results"]
            st.success(f"✅ Found {len(df)} companies from screener results")
            
            # Show companies
            st.subheader("📋 Companies from Screener")
            st.dataframe(df[['ticker', 'company_name', 'sector']] if 'company_name' in df.columns else df[['ticker']])
            
            # Select companies to analyze
            if 'ticker' in df.columns:
                selected_tickers = st.multiselect(
                    "Select companies to analyze",
                    df['ticker'].tolist(),
                    default=df['ticker'].tolist()[:3]  # Default to first 3
                )
                companies_to_analyze = selected_tickers
            else:
                st.warning("No ticker column found in screener results")
        else:
            st.info("No screener results available. Run the Fundamental Screener first!")
    
    elif data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.subheader("📋 Uploaded Data")
            st.dataframe(df.head())
            
            if 'ticker' in df.columns:
                selected_tickers = st.multiselect(
                    "Select companies to analyze",
                    df['ticker'].tolist(),
                    default=df['ticker'].tolist()[:3]
                )
                companies_to_analyze = selected_tickers
            else:
                st.error("CSV file must contain a 'ticker' column")
    
    elif data_source == "Manual Input":
        st.subheader("📝 Manual Company Input")
        manual_tickers = st.text_area(
            "Enter ticker symbols (one per line)",
            value="NVDA\nMSFT\nAAPL",
            help="Enter stock ticker symbols, one per line"
        )
        companies_to_analyze = [ticker.strip().upper() for ticker in manual_tickers.split('\n') if ticker.strip()]
    
    elif data_source == "Database Query":
        st.subheader("🗄️ Database Query")
        st.info("Database query functionality coming soon!")
    
    # Analysis execution
    if companies_to_analyze:
        st.subheader("🚀 Run Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Run QualAgent Analysis", type="primary"):
                run_qualagent_analysis(companies_to_analyze, analysis_type, user_id)
        
        with col2:
            if st.button("💰 Estimate Cost"):
                estimate_cost(companies_to_analyze, analysis_type)
    
    # Display results
    display_analysis_results()

def run_qualagent_analysis(companies, analysis_type, user_id):
    """Run QualAgent analysis on selected companies"""
    
    st.subheader("🔄 Running Analysis...")
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    for i, company in enumerate(companies):
        status_text.text(f"Analyzing {company}...")
        progress_bar.progress((i + 1) / len(companies))
        
        try:
            # Run QualAgent analysis
            result = run_single_analysis(company, analysis_type, user_id)
            results.append(result)
            
            st.success(f"✅ {company} analysis completed")
            
        except Exception as e:
            st.error(f"❌ Error analyzing {company}: {str(e)}")
    
    # Store results in session state
    st.session_state["qualagent_results"] = results
    st.session_state["analysis_completed"] = True
    
    status_text.text("Analysis completed!")
    progress_bar.progress(1.0)

def run_single_analysis(company, analysis_type, user_id):
    """Run analysis for a single company"""
    
    # Change to QualAgent directory
    qualagent_dir = Path(__file__).parent.parent / "agents" / "QualAgent"
    
    # Run the analysis command
    cmd = [
        "python", "run_enhanced_analysis.py",
        "--user-id", user_id,
        "--company", company,
        "--analysis-type", analysis_type
    ]
    
    try:
        # Execute the command
        result = subprocess.run(
            cmd,
            cwd=qualagent_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            return {
                "company": company,
                "status": "success",
                "output": result.stdout,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "company": company,
                "status": "error",
                "error": result.stderr,
                "timestamp": datetime.now().isoformat()
            }
    
    except subprocess.TimeoutExpired:
        return {
            "company": company,
            "status": "timeout",
            "error": "Analysis timed out",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "company": company,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def estimate_cost(companies, analysis_type):
    """Estimate analysis cost"""
    
    cost_per_analysis = {
        "quick": 0.003,
        "comprehensive": 0.018,
        "expert-guided": 0.030
    }
    
    total_cost = cost_per_analysis.get(analysis_type, 0.018) * len(companies)
    
    st.info(f"💰 Estimated cost for {len(companies)} companies ({analysis_type}): ${total_cost:.3f}")

def display_analysis_results():
    """Display analysis results"""
    
    if "analysis_completed" in st.session_state and st.session_state["analysis_completed"]:
        st.subheader("📊 Analysis Results")
        
        results = st.session_state.get("qualagent_results", [])
        
        if results:
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            
            # Show individual results
            for result in results:
                if result["status"] == "success":
                    with st.expander(f"✅ {result['company']} - Analysis Results"):
                        st.text(result["output"])
                else:
                    with st.expander(f"❌ {result['company']} - Error"):
                        st.error(result["error"])
            
            # Download results
            if st.button("📥 Download Results"):
                results_json = json.dumps(results, indent=2)
                st.download_button(
                    "Download Results JSON",
                    results_json,
                    file_name=f"qualagent_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    qualagent_page()
