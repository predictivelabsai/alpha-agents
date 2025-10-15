"""
QualAgent Enhanced Analysis - Streamlit Interface
Comprehensive qualitative analysis with multi-LLM support, interactive weight management, and enhanced scoring
"""

import streamlit as st
import pandas as pd
import json
import time
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import pickle
from io import BytesIO
import base64

# Add QualAgent to path
qual_agent_path = Path(__file__).parent.parent / "agents" / "QualAgent"
sys.path.insert(0, str(qual_agent_path))

# Database utility functions (copied from utils/db_util.py to avoid import issues)
def get_database_engine():
    """Get database engine - direct implementation to avoid import issues"""
    import os
    from sqlalchemy import create_engine
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable must be set for PostgreSQL connection")
    return create_engine(db_url)

def load_fundamental_screen_postgresql(run_id=None, limit=None, industry_filter=None, sector_filter=None, min_score=None):
    """Load screening results from PostgreSQL database with enhanced filtering"""
    try:
        from sqlalchemy import text
        import pandas as pd
        
        engine = get_database_engine()
        
        # Check if table exists first
        try:
            table_check = "SELECT COUNT(*) as count FROM fundamental_screen LIMIT 1"
            count_result = pd.read_sql_query(text(table_check), engine)
            if count_result.empty or count_result.iloc[0]['count'] == 0:
                st.warning("No data found in fundamental_screen table")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Table access error: {e}")
            return pd.DataFrame()
        
        # Build dynamic query with proper SQLAlchemy parameters
        query = "SELECT * FROM fundamental_screen WHERE 1=1"
        params = {}
        
        if run_id:
            query += " AND run_id = :run_id"
            params['run_id'] = run_id
        
        if industry_filter:
            query += " AND industry ILIKE :industry_filter"
            params['industry_filter'] = f"%{industry_filter}%"
        
        if sector_filter:
            query += " AND sector ILIKE :sector_filter"
            params['sector_filter'] = f"%{sector_filter}%"
        
        if min_score is not None:
            query += " AND score >= :min_score"
            params['min_score'] = min_score
        
        query += " ORDER BY created_at DESC, score DESC"
        
        if limit:
            query += " LIMIT :limit"
            params['limit'] = limit
        
        # Execute query with proper parameter handling
        if params:
            df = pd.read_sql_query(text(query), engine, params=params)
        else:
            df = pd.read_sql_query(text(query), engine)
        
        # Debug: Check if we got data
        if df.empty:
            st.warning("No data found in database")
            return pd.DataFrame()
        
        return df
    except Exception as e:
        st.error(f"Database load error: {e}")
        return pd.DataFrame()

def get_database_filters():
    """Get available filter options from database"""
    try:
        from sqlalchemy import text
        
        engine = get_database_engine()
        
        # Check if table exists first
        table_check_query = "SELECT COUNT(*) as count FROM fundamental_screen LIMIT 1"
        try:
            count_df = pd.read_sql_query(text(table_check_query), engine)
            if count_df.empty or count_df.iloc[0]['count'] == 0:
                return {'industries': [], 'sectors': [], 'runs': []}
        except Exception:
            return {'industries': [], 'sectors': [], 'runs': []}
        
        # Get unique industries
        industries_query = "SELECT DISTINCT industry FROM fundamental_screen WHERE industry IS NOT NULL ORDER BY industry"
        industries_df = pd.read_sql_query(text(industries_query), engine)
        industries = industries_df['industry'].tolist() if not industries_df.empty else []
        
        # Get unique sectors
        sectors_query = "SELECT DISTINCT sector FROM fundamental_screen WHERE sector IS NOT NULL ORDER BY sector"
        sectors_df = pd.read_sql_query(text(sectors_query), engine)
        sectors = sectors_df['sector'].tolist() if not sectors_df.empty else []
        
        # Get unique run_ids
        runs_query = "SELECT DISTINCT run_id, created_at FROM fundamental_screen ORDER BY created_at DESC LIMIT 10"
        runs_df = pd.read_sql_query(text(runs_query), engine)
        runs = runs_df['run_id'].tolist() if not runs_df.empty else []
        
        return {
            'industries': industries,
            'sectors': sectors,
            'runs': runs
        }
    except Exception as e:
        st.warning(f"Could not load filter options: {e}")
        return {
            'industries': [],
            'sectors': [],
            'runs': []
        }

# Initialize variables for imports
LLMAPITester = None
InteractiveWeightManager = None
WeightingScheme = None

try:
    # Method 1: Standard package imports
    from utils.test_llm_api import LLMAPITester
    from utils.weight_manager import InteractiveWeightManager
    from engines.enhanced_scoring_system import WeightingScheme
    import_success = "✅ Standard package imports successful!"

except ImportError as e1:
    try:
        # Method 2: Direct module imports
        sys.path.insert(0, str(qual_agent_path / "utils"))
        sys.path.insert(0, str(qual_agent_path / "engines"))

        import test_llm_api
        import weight_manager
        import enhanced_scoring_system

        LLMAPITester = test_llm_api.LLMAPITester
        InteractiveWeightManager = weight_manager.InteractiveWeightManager
        WeightingScheme = enhanced_scoring_system.WeightingScheme
        import_success = "✅ Direct module imports successful!"

    except ImportError as e2:
        st.error(f"❌ Failed to import QualAgent modules:")
        st.error(f"Standard import error: {e1}")
        st.error(f"Direct import error: {e2}")
        st.error(f"🔍 QualAgent path: {qual_agent_path}")
        st.error(f"🔍 Path exists: {qual_agent_path.exists()}")

        # Show some debug info
        with st.expander("🔍 Debug Information"):
            st.write(f"Python sys.path entries with QualAgent:")
            for i, path in enumerate(sys.path):
                if 'QualAgent' in path:
                    st.write(f"{i}: {path}")

            if qual_agent_path.exists():
                st.write(f"Files in {qual_agent_path}:")
                try:
                    files = list(qual_agent_path.iterdir())
                    for f in files[:10]:  # Show first 10 files
                        st.write(f"- {f.name}")
                except:
                    st.write("Could not list files")

        st.stop()

# Show success message in sidebar
if 'import_success' in locals():
    st.sidebar.success(import_success)

st.set_page_config(
    page_title="Qualitative Analysis Agent",
    page_icon="🧠",
    layout="wide"
)

def init_session_state():
    """Initialize session state variables"""
    if 'uploaded_companies' not in st.session_state:
        st.session_state.uploaded_companies = None
    if 'api_test_results' not in st.session_state:
        st.session_state.api_test_results = None
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = []
    if 'current_weights' not in st.session_state:
        st.session_state.current_weights = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'show_weight_editor' not in st.session_state:
        st.session_state.show_weight_editor = False
    if 'editing_category' not in st.session_state:
        st.session_state.editing_category = "Competitive Moats"

# Helper functions for LLM Results Explorer
def safe_float_format(value, default=0.0, format_str=".1%"):
    """Safely convert value to float and format, handles string inputs"""
    try:
        # Handle None or empty values
        if value is None or value == "":
            float_val = default
        else:
            float_val = float(value)

        if format_str == ".1%":
            return f"{float_val:.1%}"
        elif format_str == ".2f":
            return f"{float_val:.2f}"
        else:
            return str(float_val)
    except (ValueError, TypeError):
        # Return a safe default formatted string
        if format_str == ".1%":
            return f"{default:.1%}"
        elif format_str == ".2f":
            return f"{default:.2f}"
        else:
            return str(value if value is not None else default)

def get_results_directory():
    """Get the results directory path"""
    return Path(__file__).parent.parent / "agents" / "QualAgent" / "results"

def parse_result_filename(filename):
    """Parse result filename to extract ticker and timestamp"""
    if filename.startswith("multi_llm_analysis_") and filename.endswith(".json"):
        parts = filename[19:-5].split("_")  # Remove prefix and suffix
        if len(parts) >= 2:
            # Handle new timestamp format: YYYYMMDD_HHMMSS (2 parts) vs old format (1 part)
            if len(parts) >= 3 and len(parts[-2]) == 8 and len(parts[-1]) == 6:
                # New format: ticker_YYYYMMDD_HHMMSS
                ticker = "_".join(parts[:-2])  # Join all but last 2 parts
                timestamp = f"{parts[-2]}_{parts[-1]}"  # Rejoin timestamp parts
            else:
                # Old format: ticker_unixtime or other formats
                ticker = "_".join(parts[:-1])  # Join all but last part
                timestamp = parts[-1]
            return ticker, timestamp
    return None, None

def get_available_timestamps():
    """Get all available timestamps from result files"""
    results_dir = get_results_directory()
    timestamps = set()

    if results_dir.exists():
        for file in results_dir.glob("multi_llm_analysis_*.json"):
            _, timestamp = parse_result_filename(file.name)
            if timestamp:
                timestamps.add(timestamp)

    return sorted(list(timestamps), reverse=True)  # Most recent first

def get_results_for_timestamp(timestamp):
    """Get all company results for a specific timestamp"""
    results_dir = get_results_directory()
    results = []

    if results_dir.exists():
        for file in results_dir.glob(f"multi_llm_analysis_*_{timestamp}.json"):
            ticker, _ = parse_result_filename(file.name)
            if ticker:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    results.append({
                        'ticker': ticker,
                        'file_path': file,
                        'data': data,
                        'composite_score': data.get('composite_score', {}).get('score', 0)
                    })
                except Exception as e:
                    st.warning(f"Error loading {file.name}: {e}")

    # Sort by composite score (descending)
    results.sort(key=lambda x: x['composite_score'], reverse=True)
    return results

def display_company_llm_analysis(company_data, show_expanded=False):
    """Display detailed LLM analysis for a single company in user-friendly format"""
    data = company_data['data']
    ticker = company_data['ticker']

    # Company header
    company_info = data.get('company', {})
    st.subheader(f"🏢 {company_info.get('company_name', ticker)} ({ticker})")

    # Composite score
    composite = data.get('composite_score', {})
    col1, col2, col3 = st.columns(3)
    with col1:
        score_fmt = safe_float_format(composite.get('score', 0), 0.0, ".2f")
        st.metric("📊 Composite Score", f"{score_fmt}/5.0")
    with col2:
        confidence_fmt = safe_float_format(composite.get('confidence', 0), 0.0, ".1%")
        st.metric("🎯 Confidence", confidence_fmt)
    with col3:
        st.metric("📈 Components", composite.get('components_count', 0))

    # Analysis models used
    models_used = data.get('metadata', {}).get('models_used', [])
    st.info(f"🤖 **Models Used:** {', '.join(models_used)}")

    # Individual Model Results (Main Content)
    st.markdown("### 🧠 Individual LLM Analysis")

    individual_results = data.get('individual_model_results', {})
    for model_name, model_data in individual_results.items():
        with st.expander(f"📝 {model_name.upper()} Analysis", expanded=show_expanded):

            # Model metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Provider:** {model_data.get('provider', 'Unknown')}")
            with col2:
                if model_data.get('tokens_used'):
                    st.write(f"**Tokens:** {model_data['tokens_used']:,}")
            with col3:
                if model_data.get('cost_usd'):
                    st.write(f"**Cost:** ${model_data['cost_usd']:.4f}")

            st.markdown("---")

            # Parse the model's analysis
            parsed_output = model_data.get('parsed_output', {})

            # Competitive Moat Analysis
            moat_analysis = parsed_output.get('competitive_moat_analysis', {})
            if moat_analysis:
                st.markdown("#### 🏰 Competitive Moat Analysis")
                for moat_name, moat_data in moat_analysis.items():
                    if isinstance(moat_data, dict) and 'score' in moat_data:
                        with st.container():
                            st.markdown(f"**{moat_name.replace('_', ' ').title()}**")

                            # Score and confidence
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                score = moat_data.get('score', 0)
                                confidence = moat_data.get('confidence', 0)
                                confidence_fmt = safe_float_format(confidence, 0.0, ".1%")
                                st.write(f"Score: **{score}/5** (Confidence: {confidence_fmt})")

                            with col2:
                                justification = moat_data.get('justification', '')
                                st.write(justification)

                            st.markdown("---")

            # Strategic Insights
            strategic_insights = parsed_output.get('strategic_insights', {})
            if strategic_insights:
                st.markdown("#### 🎯 Strategic Insights")

                # Growth Drivers
                growth_drivers = strategic_insights.get('key_growth_drivers', [])
                if growth_drivers:
                    st.markdown("**🚀 Key Growth Drivers:**")
                    for driver in growth_drivers:
                        if isinstance(driver, dict):
                            try:
                                confidence_fmt = safe_float_format(driver.get('confidence', 0), 0.0, ".1%")
                                st.write(f"• **{driver.get('driver', 'Unknown')}** "
                                       f"(Impact: {driver.get('impact', 'N/A')}/5, "
                                       f"Timeframe: {driver.get('timeframe', 'N/A')}, "
                                       f"Confidence: {confidence_fmt})")
                            except Exception as e:
                                st.error(f"Error formatting growth driver: {e}")
                                st.write(f"Driver data: {driver}")
                                # Fallback display
                                st.write(f"• **{driver.get('driver', 'Unknown')}** "
                                       f"(Impact: {driver.get('impact', 'N/A')}/5, "
                                       f"Timeframe: {driver.get('timeframe', 'N/A')}, "
                                       f"Confidence: {driver.get('confidence', 'N/A')})")
                        else:
                            st.write(f"• {driver}")

                # Risk Factors
                risk_factors = strategic_insights.get('major_risk_factors', [])
                if risk_factors:
                    st.markdown("**⚠️ Major Risk Factors:**")
                    for risk in risk_factors:
                        if isinstance(risk, dict):
                            try:
                                probability_fmt = safe_float_format(risk.get('probability', 0), 0.0, ".1%")
                                st.write(f"• **{risk.get('risk', 'Unknown')}** "
                                       f"(Severity: {risk.get('severity', 'N/A')}/5, "
                                       f"Probability: {probability_fmt})")
                            except Exception as e:
                                st.error(f"Error formatting risk: {e}")
                                st.write(f"Risk data: {risk}")
                                # Fallback display
                                st.write(f"• **{risk.get('risk', 'Unknown')}** "
                                       f"(Severity: {risk.get('severity', 'N/A')}/5, "
                                       f"Probability: {risk.get('probability', 'N/A')})")
                            mitigation = risk.get('mitigation', '')
                            if mitigation:
                                st.write(f"  *Mitigation:* {mitigation}")
                        else:
                            st.write(f"• {risk}")

                # Red Flags
                red_flags = strategic_insights.get('red_flags', [])
                if red_flags:
                    st.markdown("**🚨 Red Flags:**")
                    for flag in red_flags:
                        if isinstance(flag, dict):
                            st.write(f"• **{flag.get('flag', 'Unknown')}** "
                                   f"(Severity: {flag.get('severity', 'N/A')}/5)")
                            evidence = flag.get('evidence', '')
                            if evidence:
                                st.write(f"  *Evidence:* {evidence}")
                        else:
                            st.write(f"• {flag}")

                # Strategic Components
                strategic_components = [
                    'transformation_potential', 'platform_expansion', 'competitive_differentiation',
                    'market_timing', 'management_quality', 'technology_moats'
                ]

                st.markdown("**📊 Strategic Component Scores:**")
                for component in strategic_components:
                    comp_data = strategic_insights.get(component, {})
                    if isinstance(comp_data, dict) and 'score' in comp_data:
                        try:
                            score = comp_data.get('score', 0)
                            confidence = comp_data.get('confidence', 0)
                            justification = comp_data.get('justification', '')

                            confidence_fmt = safe_float_format(confidence, 0.0, ".1%")
                            st.markdown(f"**{component.replace('_', ' ').title()}:** {score}/5 (Confidence: {confidence_fmt})")
                            st.write(justification)
                            st.markdown("---")
                        except Exception as e:
                            st.error(f"Error formatting {component}: {e}")
                            st.write(f"Component data: {comp_data}")
                            # Fallback display
                            st.markdown(f"**{component.replace('_', ' ').title()}:** {comp_data.get('score', 'N/A')}/5 (Confidence: {comp_data.get('confidence', 'N/A')})")
                            st.write(comp_data.get('justification', ''))
                            st.markdown("---")

            # Competitor Analysis
            competitor_analysis = parsed_output.get('competitor_analysis', [])
            if competitor_analysis:
                st.markdown("#### 🏢 Competitor Analysis")
                for i, competitor in enumerate(competitor_analysis):
                    if isinstance(competitor, dict):
                        st.markdown(f"**{i+1}. {competitor.get('name', 'Unknown')} ({competitor.get('ticker', 'N/A')})**")
                        st.write(f"Market Share: {competitor.get('market_share', 'N/A')}")
                        st.write(f"Threat Level: {competitor.get('threat_level', 'N/A')}/5")
                        st.write(f"Position: {competitor.get('competitive_position', 'N/A')}")

                        differentiators = competitor.get('key_differentiators', [])
                        if differentiators:
                            st.write(f"Key Differentiators: {', '.join(differentiators)}")

                        response = competitor.get('strategic_response', '')
                        if response:
                            st.write(f"Strategic Response: {response}")

                        st.markdown("---")

def explore_llm_results_section():
    """Explore LLM Results section with two modes: post-analysis and standalone"""
    st.header("🔍 Explore LLM Results")
    st.markdown("**Deep dive into individual LLM reasoning and analysis details**")

    # Check if we have current analysis results
    batch_results = st.session_state.get('parsed_batch_results', [])
    has_current_results = len(batch_results) > 0

    # Mode selection
    if has_current_results:
        mode = st.radio(
            "📊 Select Results Source:",
            ["Current Analysis Results", "Explore Saved Results"],
            help="Current Analysis: Use results from the latest analysis run. Saved Results: Browse all saved analyses from disk."
        )
    else:
        mode = "Explore Saved Results"
        st.info("💡 No current analysis results found. Showing saved results from disk.")

    if mode == "Current Analysis Results" and has_current_results:
        st.markdown("### 📈 Current Analysis Results")
        st.success(f"📊 Showing results for {len(batch_results)} companies from your latest analysis")

        # Display companies in same order as Results & Downloads
        for i, company_result in enumerate(batch_results):
            ticker = company_result.get('ticker', f'Company_{i+1}')
            composite_score = company_result.get('average_composite_score', 0)

            score_fmt = safe_float_format(composite_score, 0.0, ".2f")
            with st.expander(f"🏢 {ticker} - Score: {score_fmt}/5.0", expanded=False):
                # Find the corresponding JSON file for this company
                # Try to match with saved results
                timestamps = get_available_timestamps()
                found_data = None

                for timestamp in timestamps[:3]:  # Check last 3 timestamps
                    results_for_timestamp = get_results_for_timestamp(timestamp)
                    for result in results_for_timestamp:
                        if result['ticker'] == ticker:
                            found_data = result
                            break
                    if found_data:
                        break

                if found_data:
                    display_company_llm_analysis(found_data, show_expanded=False)
                else:
                    st.warning(f"⚠️ Detailed LLM analysis not found for {ticker}. This may happen if the analysis was run very recently or files were moved.")
                    st.write("**Available data:**")
                    st.json(company_result)

    else:  # Explore Saved Results mode
        st.markdown("### 💾 Explore Saved Results")

        # Get available timestamps
        timestamps = get_available_timestamps()

        if not timestamps:
            st.warning("📂 No saved analysis results found in the results directory.")
            st.info(f"🔍 Looking for files in: {get_results_directory()}")
            return

        # Timestamp selection
        st.markdown("#### 🕐 Select Analysis Timestamp")

        # Convert timestamps to readable format for display
        timestamp_options = []
        for ts in timestamps:
            try:
                # Handle both old Unix timestamps and new human-readable format
                if ts.isdigit() and len(ts) == 10:
                    # Old Unix timestamp format
                    dt = pd.to_datetime(int(ts), unit='s')
                    readable = dt.strftime('%Y-%m-%d %H:%M:%S')
                elif '_' in ts and len(ts) == 15:
                    # New human-readable format: YYYYMMDD_HHMMSS
                    date_part, time_part = ts.split('_')
                    year, month, day = date_part[:4], date_part[4:6], date_part[6:8]
                    hour, minute, second = time_part[:2], time_part[2:4], time_part[4:6]
                    readable = f"{year}-{month}-{day} {hour}:{minute}:{second}"
                else:
                    # Fallback for any other format
                    readable = f"Analysis Run {ts}"
                timestamp_options.append(f"{readable} (ID: {ts})")
            except Exception as e:
                timestamp_options.append(f"Timestamp: {ts}")

        selected_option = st.selectbox(
            "Choose analysis run:",
            options=timestamp_options,
            help="Select which analysis run to explore. Most recent runs are shown first."
        )

        if selected_option:
            # Extract timestamp ID from selected option
            selected_timestamp = selected_option.split("(ID: ")[-1].rstrip(")")

            # Get results for selected timestamp
            results = get_results_for_timestamp(selected_timestamp)

            if not results:
                st.error(f"❌ No results found for timestamp {selected_timestamp}")
                return

            # Display summary
            st.success(f"📊 Found {len(results)} companies for selected analysis run")

            # Results overview table
            st.markdown("#### 📋 Companies Overview")
            overview_data = []
            for result in results:
                score_fmt = safe_float_format(result['composite_score'], 0.0, ".2f")
                confidence_val = result['data'].get('composite_score', {}).get('confidence', 0)
                confidence_fmt = safe_float_format(confidence_val, 0.0, ".1%")
                overview_data.append({
                    'Rank': len(overview_data) + 1,
                    'Ticker': result['ticker'],
                    'Company Name': result['data'].get('company', {}).get('company_name', 'N/A'),
                    'Composite Score': f"{score_fmt}/5.0",
                    'Confidence': confidence_fmt,
                    'Models Used': len(result['data'].get('individual_model_results', {}))
                })

            overview_df = pd.DataFrame(overview_data)
            st.dataframe(overview_df, use_container_width=True)

            # Detailed analysis for each company
            st.markdown("#### 🔍 Detailed LLM Analysis")

            for result in results:
                ticker = result['ticker']
                composite_score = result['composite_score']

                score_fmt = safe_float_format(composite_score, 0.0, ".2f")
                with st.expander(f"🏢 {ticker} - Score: {score_fmt}/5.0", expanded=False):
                    display_company_llm_analysis(result, show_expanded=False)

def main():
    init_session_state()

    st.title("🧠 QualAgent Enhanced Analysis")
    st.markdown("**Advanced qualitative analysis with multi-LLM support and interactive weight management**")

    # Create main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📁 Data Input",
        "🔌 API Testing",
        "⚖️ Weight Management",
        "🚀 Run Analysis",
        "📊 Results & Downloads",
        "🔍 Explore LLM Results"
    ])

    with tab1:
        data_input_section()

    with tab2:
        api_testing_section()

    with tab3:
        weight_management_section()

    with tab4:
        analysis_execution_section()

    with tab5:
        results_section()

    with tab6:
        explore_llm_results_section()

def data_input_section():
    """Data input section with file upload and database options"""
    st.header("📁 Data Input")

    # Data source selection
    st.subheader("📊 Data Source")
    data_source = st.radio(
        "Load Companies",
        ["From Screener", "From Database", "Upload CSV", "Manual Input"],
        horizontal=True
    )

    companies_to_analyze = []

    if data_source == "From Screener":
        if "screener_results" in st.session_state and not st.session_state["screener_results"].empty:
            df = st.session_state["screener_results"]
            st.success(f"✅ Found {len(df)} companies from screener results")
            
            # Show companies with better column handling
            st.subheader("📋 Companies from Screener")
            
            # Determine display columns
            display_cols = []
            if 'ticker' in df.columns:
                display_cols.append('ticker')
            if 'company_name' in df.columns:
                display_cols.append('company_name')
            elif 'name' in df.columns:
                display_cols.append('name')
            if 'sector' in df.columns:
                display_cols.append('sector')
            if 'score' in df.columns:
                display_cols.append('score')
            
            if display_cols:
                st.dataframe(df[display_cols], width='stretch')
            else:
                st.dataframe(df.head(), width='stretch')
            
            # Select companies with better ticker handling
            ticker_col = None
            if 'ticker' in df.columns:
                ticker_col = 'ticker'
            elif 'index' in df.columns:
                ticker_col = 'index'
            elif df.index.name == 'ticker':
                ticker_col = 'index'
            
            if ticker_col:
                st.subheader("🎯 Company Selection")
                
                # Quick selection buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    select_all = st.button("✅ Select All", key="screener_select_all")
                with col2:
                    select_top = st.button("🏆 Select Top 5", key="screener_select_top")
                with col3:
                    select_none = st.button("❌ Select None", key="screener_select_none")
                
                # Handle selection buttons
                if select_all:
                    st.session_state['screener_selected_tickers'] = df[ticker_col].tolist()
                elif select_top:
                    st.session_state['screener_selected_tickers'] = df[ticker_col].tolist()[:5]
                elif select_none:
                    st.session_state['screener_selected_tickers'] = []
                
                # Multiselect with session state
                default_selection = st.session_state.get('screener_selected_tickers', df[ticker_col].tolist()[:3])
                selected_tickers = st.multiselect(
                    "Select companies to analyze",
                    df[ticker_col].tolist(),
                    default=default_selection,
                    key="screener_company_selection"
                )
                companies_to_analyze = selected_tickers
                
                # Show selection summary
                if companies_to_analyze:
                    st.success(f"✅ Selected {len(companies_to_analyze)} companies for analysis")
                    st.write("Selected companies:", ", ".join(companies_to_analyze))
                else:
                    st.warning("⚠️ No companies selected")
            else:
                st.error("❌ No ticker column found in screener results")
                st.info("Available columns:", list(df.columns))
        else:
            st.warning("No screener results available. Run the Fundamental Screener first!")
    
    elif data_source == "From Database":
        st.subheader("🗄️ Load from PostgreSQL Database")
        
        # Test database connection first
        try:
            from sqlalchemy import text
            
            engine = get_database_engine()
            
            # Test connection with better error handling
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test"))
                test_value = result.fetchone()[0]
            
            st.success("✅ Database connection successful!")
            
            # Test if fundamental_screen table exists and has data
            try:
                test_query = "SELECT COUNT(*) as count FROM fundamental_screen LIMIT 1"
                test_result = pd.read_sql_query(test_query, engine)
                if test_result.empty or test_result.iloc[0]['count'] == 0:
                    st.warning("⚠️ No data found in fundamental_screen table")
                    st.info("💡 Run the Fundamental Screener first to populate the database")
                    st.stop()
                else:
                    st.success(f"✅ Found {test_result.iloc[0]['count']} companies in database")
            except Exception as e:
                st.error(f"❌ Table access error: {e}")
                st.info("💡 The fundamental_screen table may not exist. Run the screener first.")
                st.stop()
            
            # Only Pre-built Queries mode
            query_mode = "📊 Pre-built Queries"
            
        except Exception as e:
            st.error(f"❌ Database connection failed: {str(e)}")
            st.info("💡 Please check your DATABASE_URL in .env file")
            st.stop()
        
        if query_mode == "📊 Pre-built Queries":
            st.subheader("📊 Company Selection from Database")
            
            # Load all companies from database
            with st.spinner("Loading all companies from database..."):
                try:
                    # Load all companies without limit
                    simple_query = "SELECT * FROM fundamental_screen ORDER BY created_at DESC, score DESC"
                    all_companies = pd.read_sql_query(simple_query, engine)
                except Exception as e:
                    st.error(f"Direct query failed: {e}")
                    # Fallback to the function
                    all_companies = load_fundamental_screen_postgresql()
            
            if not all_companies.empty:
                st.success(f"✅ Loaded {len(all_companies)} companies from database")
                
                # Add filtering options
                st.subheader("🔍 Filter Companies")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Sector filter
                    if 'sector' in all_companies.columns:
                        unique_sectors = ['All Sectors'] + sorted(all_companies['sector'].dropna().unique().tolist())
                        selected_sector = st.selectbox("Filter by Sector", unique_sectors)
                        sector_filter = None if selected_sector == "All Sectors" else selected_sector
                    else:
                        sector_filter = None
                        st.info("No sector column found")
                
                with col2:
                    # Industry filter
                    if 'industry' in all_companies.columns:
                        unique_industries = ['All Industries'] + sorted(all_companies['industry'].dropna().unique().tolist())
                        selected_industry = st.selectbox("Filter by Industry", unique_industries)
                        industry_filter = None if selected_industry == "All Industries" else selected_industry
                    else:
                        industry_filter = None
                        st.info("No industry column found")
                
                with col3:
                    # Score filter
                    if 'score' in all_companies.columns:
                        min_score = st.number_input(
                            "Minimum Score", 
                            min_value=0.0, 
                            max_value=100.0, 
                            value=0.0, 
                            step=1.0,
                            help="Filter companies with score above this value"
                        )
                        score_filter = min_score if min_score > 0 else None
                    else:
                        score_filter = None
                        st.info("No score column found")
                
                # Apply filters
                filtered_companies = all_companies.copy()
                if sector_filter:
                    filtered_companies = filtered_companies[filtered_companies['sector'] == sector_filter]
                if industry_filter:
                    filtered_companies = filtered_companies[filtered_companies['industry'] == industry_filter]
                if score_filter:
                    filtered_companies = filtered_companies[filtered_companies['score'] >= score_filter]
                
                st.info(f"📊 Showing {len(filtered_companies)} companies after filtering (from {len(all_companies)} total)")
                
                # Show filtered companies
                st.subheader("📊 Filtered Companies")
                
                # Display columns with better formatting
                display_cols = []
                if 'ticker' in filtered_companies.columns:
                    display_cols.append('ticker')
                if 'company_name' in filtered_companies.columns:
                    display_cols.append('company_name')
                if 'sector' in filtered_companies.columns:
                    display_cols.append('sector')
                if 'industry' in filtered_companies.columns:
                    display_cols.append('industry')
                if 'score' in filtered_companies.columns:
                    display_cols.append('score')
                if 'market_cap' in filtered_companies.columns:
                    display_cols.append('market_cap')
                
                if display_cols:
                    # Format the dataframe for better display
                    display_df = filtered_companies[display_cols].copy()
                    if 'score' in display_df.columns:
                        display_df['score'] = display_df['score'].round(2)
                    if 'market_cap' in display_df.columns:
                        display_df['market_cap'] = display_df['market_cap'].apply(lambda x: f"${x/1e9:.1f}B" if pd.notna(x) and x > 0 else "N/A")
                    
                    st.dataframe(display_df, width='stretch')
                else:
                    st.dataframe(filtered_companies, width='content')
                
                # Direct company selection - using working approach from alpha_agents_fixed.py
                if 'ticker' in filtered_companies.columns:
                    st.subheader("🎯 Select Companies for Analysis")
                    
                    # Quick selection buttons
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        select_all = st.button("✅ Select All", key="direct_select_all")
                    with col2:
                        select_top5 = st.button("🏆 Select Top 5", key="direct_select_top5")
                    with col3:
                        select_top10 = st.button("🥇 Select Top 10", key="direct_select_top10")
                    with col4:
                        select_none = st.button("❌ Select None", key="direct_select_none")
                    
                    # Handle selection buttons - using working approach
                    if select_all:
                        st.session_state['direct_selected_tickers'] = filtered_companies['ticker'].tolist()
                    elif select_top5:
                        st.session_state['direct_selected_tickers'] = filtered_companies['ticker'].tolist()[:5]
                    elif select_top10:
                        st.session_state['direct_selected_tickers'] = filtered_companies['ticker'].tolist()[:10]
                    elif select_none:
                        st.session_state['direct_selected_tickers'] = []
                    
                    # Multiselect with better default - using working approach
                    default_selection = st.session_state.get('direct_selected_tickers', filtered_companies['ticker'].tolist()[:3])
                    selected_tickers = st.multiselect(
                        "Select companies to analyze",
                        filtered_companies['ticker'].tolist(),
                        default=default_selection,
                        key="direct_company_selection",
                        help="Choose companies for QualAgent analysis"
                    )
                    companies_to_analyze = selected_tickers
                    
                    # Show selection summary with company details
                    if companies_to_analyze:
                        st.success(f"✅ Selected {len(companies_to_analyze)} companies for analysis")
                        
                        # Show selected companies with their details
                        selected_df = filtered_companies[filtered_companies['ticker'].isin(companies_to_analyze)]
                        if not selected_df.empty:
                            st.subheader("📋 Selected Companies Details")
                            detail_cols = ['ticker']
                            if 'company_name' in selected_df.columns:
                                detail_cols.append('company_name')
                            if 'sector' in selected_df.columns:
                                detail_cols.append('sector')
                            if 'score' in selected_df.columns:
                                detail_cols.append('score')
                            
                            detail_df = selected_df[detail_cols].copy()
                            if 'score' in detail_df.columns:
                                detail_df['score'] = detail_df['score'].round(2)
                            
                            st.dataframe(detail_df, width='stretch')
                        
                        # Store selected companies for analysis
                        st.session_state["companies_to_analyze"] = companies_to_analyze
                        st.session_state["analysis_data_source"] = "database"
                        st.session_state["analysis_data"] = filtered_companies
                        
                        st.success("🎯 Companies ready for QualAgent analysis! Go to 'Run Analysis' tab to start.")
                    else:
                        st.warning("⚠️ No companies selected")
                else:
                    st.error("❌ No 'ticker' column found in database")
            else:
                st.warning("⚠️ No companies found in database")
                st.info("💡 Run the Fundamental Screener first to populate the database")
        
    
    elif data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.subheader("📋 Uploaded Data")
            st.dataframe(df.head(), width='stretch')
            
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
    

    # Store selected companies for analysis
    if companies_to_analyze:
        # Convert to DataFrame format expected by the analysis section
        companies_df = pd.DataFrame({
            'ticker': companies_to_analyze,
            'company_name': [f"{ticker} Inc." for ticker in companies_to_analyze],
            'sector': ['Technology'] * len(companies_to_analyze),
            'industry': ['Software'] * len(companies_to_analyze)
        })
        st.session_state.uploaded_companies = companies_df
        st.success(f"✅ {len(companies_to_analyze)} companies ready for analysis")
        st.dataframe(companies_df, width='stretch')

def api_testing_section():
    """API testing section"""
    st.header("🔌 API Connection Testing")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Test LLM Model Connectivity")

        if st.button("🚀 Run API Tests", type="primary"):
            with st.spinner("Testing API connections... This may take a few minutes."):
                try:
                    # Run API tests
                    tester = LLMAPITester()

                    # Test configured models
                    st.write("**Phase 1: Testing configured models...**")
                    configured_results = tester.test_all_models()

                    # Test discovery models
                    st.write("**Phase 2: Testing additional models...**")
                    discovery_results = tester.test_comprehensive_model_discovery()

                    # Combine results
                    all_results = {**configured_results, **discovery_results}
                    st.session_state.api_test_results = all_results

                    st.success("✅ API testing completed!")

                except Exception as e:
                    st.error(f"❌ API testing failed: {str(e)}")

    with col2:
        st.subheader("Environment Check")

        # Check environment variables
        env_vars = ['TOGETHER_API_KEY', 'OPENAI_API_KEY']
        for var in env_vars:
            if os.getenv(var):
                st.success(f"✅ {var} is set")
            else:
                st.warning(f"⚠️ {var} is not set")

        st.markdown("""
        **Required API Keys:**
        - `TOGETHER_API_KEY`: For TogetherAI models
        - `OPENAI_API_KEY`: For OpenAI models

        Add these to your `.env` file in the project root.
        """)

    # Display results if available
    if st.session_state.api_test_results:
        st.subheader("🔍 API Test Results")

        # Categorize models
        working_models = []
        failed_models = []

        for model, result in st.session_state.api_test_results.items():
            if result['success']:
                working_models.append(model)
            else:
                failed_models.append(model)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("✅ Working Models", len(working_models))

        with col2:
            st.metric("❌ Failed Models", len(failed_models))

        with col3:
            st.metric("📊 Success Rate", f"{len(working_models) / len(st.session_state.api_test_results) * 100:.1f}%")

        # Model selection
        st.subheader("🎯 Select Models for Analysis")
        if working_models:
            selected = st.multiselect(
                "Choose LLM models to use (recommended: 3-5 models for consensus)",
                options=working_models,
                default=working_models[:3] if len(working_models) >= 3 else working_models,
                help="More models provide better consensus but increase cost and time"
            )
            st.session_state.selected_models = selected

            if selected:
                st.success(f"✅ Selected {len(selected)} models for analysis")
        else:
            st.error("❌ No working models found. Please check your API keys and try again.")

def weight_management_section():
    """Interactive weight management section"""
    st.header("⚖️ Weight Management")

    # Initialize weight manager if not already done
    if 'weight_manager' not in st.session_state:
        st.session_state.weight_manager = InteractiveWeightManager()
        st.session_state.current_weights = st.session_state.weight_manager.current_weights

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Current Weight Configuration")

        # Load default weights button
        if st.button("🔄 Load Default Weights"):
            try:
                st.session_state.weight_manager = InteractiveWeightManager()
                st.session_state.current_weights = st.session_state.weight_manager.current_weights
                st.success("✅ Default weights loaded")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading weights: {str(e)}")

        # Display current weights
        if st.session_state.current_weights:
            weights = st.session_state.current_weights

            # Display weight summary (all weights normalized together)
            positive_weights = (weights.barriers_to_entry + weights.brand_monopoly + weights.economies_of_scale +
                              weights.network_effects + weights.switching_costs + weights.competitive_differentiation +
                              weights.technology_moats + weights.market_timing + weights.management_quality +
                              weights.key_growth_drivers + weights.transformation_potential + weights.platform_expansion)
            negative_weights = weights.major_risk_factors + weights.red_flags
            all_weights_total = positive_weights + negative_weights

            col_pos, col_neg, col_total = st.columns(3)
            with col_pos:
                st.metric("Positive Weights", f"{positive_weights:.3f}")
            with col_neg:
                st.metric("Risk Weights", f"{negative_weights:.3f}")
            with col_total:
                st.metric("Total (All Weights)", f"{all_weights_total:.3f}")

            if abs(all_weights_total - 1.0) > 0.001:
                st.warning("⚠️ All weights will be normalized so total = 1.0")
            else:
                st.success("✅ All weights sum to 1.0")

            st.info("ℹ️ **New Methodology**: All weights (positive + negative) are normalized together so their sum equals 1.0. Risk factors with negative weights directly reduce the composite score.")

            # Competitive Moats
            st.markdown("**🏰 Competitive Moats**")
            moat_weights = {
                'Barriers to Entry': weights.barriers_to_entry,
                'Brand Monopoly': weights.brand_monopoly,
                'Economies of Scale': weights.economies_of_scale,
                'Network Effects': weights.network_effects,
                'Switching Costs': weights.switching_costs
            }

            moat_total = sum(moat_weights.values())
            moat_df = pd.DataFrame([
                {
                    'Component': k,
                    'Weight': f"{v:.3f}",
                    'Percentage': f"{(v / moat_total * 100) if moat_total > 0 else 0:.1f}%"
                }
                for k, v in moat_weights.items()
            ])
            st.dataframe(moat_df, use_container_width=True, hide_index=True)

            # Strategic Insights
            st.markdown("**📈 Strategic Insights**")
            strategy_weights = {
                'Competitive Differentiation': weights.competitive_differentiation,
                'Technology Moats': weights.technology_moats,
                'Market Timing': weights.market_timing,
                'Management Quality': weights.management_quality,
                'Key Growth Drivers': weights.key_growth_drivers,
                'Transformation Potential': weights.transformation_potential,
                'Platform Expansion': weights.platform_expansion
            }

            strategy_total = sum(strategy_weights.values())
            strategy_df = pd.DataFrame([
                {
                    'Component': k,
                    'Weight': f"{v:.3f}",
                    'Percentage': f"{(v / strategy_total * 100) if strategy_total > 0 else 0:.1f}%"
                }
                for k, v in strategy_weights.items()
            ])
            st.dataframe(strategy_df, use_container_width=True, hide_index=True)

            # Risk Factors
            st.markdown("**⚠️ Risk Factors**")
            risk_weights = {
                'Major Risk Factors': weights.major_risk_factors,
                'Red Flags': weights.red_flags
            }

            risk_df = pd.DataFrame([
                {
                    'Component': k,
                    'Weight': f"{v:.3f}",
                    'Percentage': "Risk Factor"
                }
                for k, v in risk_weights.items()
            ])
            st.dataframe(risk_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Weight Configuration Options")

        # Investment philosophy presets
        st.markdown("**🎯 Investment Philosophy Presets**")

        philosophy_options = {
            "Custom": "custom",
            "📈 Growth Focus": "1",
            "🏰 Value Focus": "2",
            "💎 Quality Focus": "3",
            "🛡️ Risk-Aware": "4",
            "⚡ Tech Focus": "5"
        }

        philosophy = st.selectbox(
            "Choose investment philosophy",
            list(philosophy_options.keys()),
            help="Each philosophy emphasizes different scoring components"
        )

        if philosophy != "Custom" and st.button(f"Apply {philosophy} Weights"):
            try:
                # Apply the philosophy using the existing weight manager logic
                philosophy_num = philosophy_options[philosophy]

                # Reset to default first
                st.session_state.weight_manager.current_weights = WeightingScheme()

                # Apply the selected philosophy using the existing logic
                if philosophy_num == "1":  # Growth Focus
                    st.session_state.weight_manager.current_weights.key_growth_drivers = 0.25
                    st.session_state.weight_manager.current_weights.transformation_potential = 0.20
                    st.session_state.weight_manager.current_weights.platform_expansion = 0.15
                    st.session_state.weight_manager.current_weights.market_timing = 0.12
                    st.session_state.weight_manager.current_weights.barriers_to_entry = 0.10
                    st.session_state.weight_manager.current_weights.competitive_differentiation = 0.08
                    st.session_state.weight_manager.current_weights.technology_moats = 0.05
                    st.session_state.weight_manager.current_weights.management_quality = 0.03
                    st.session_state.weight_manager.current_weights.major_risk_factors = -0.015
                    st.session_state.weight_manager.current_weights.red_flags = -0.005

                elif philosophy_num == "2":  # Value Focus
                    st.session_state.weight_manager.current_weights.barriers_to_entry = 0.30
                    st.session_state.weight_manager.current_weights.brand_monopoly = 0.25
                    st.session_state.weight_manager.current_weights.economies_of_scale = 0.20
                    st.session_state.weight_manager.current_weights.switching_costs = 0.15
                    st.session_state.weight_manager.current_weights.competitive_differentiation = 0.05
                    st.session_state.weight_manager.current_weights.management_quality = 0.03
                    st.session_state.weight_manager.current_weights.major_risk_factors = -0.015
                    st.session_state.weight_manager.current_weights.red_flags = -0.005

                elif philosophy_num == "3":  # Quality Focus
                    st.session_state.weight_manager.current_weights.management_quality = 0.25
                    st.session_state.weight_manager.current_weights.competitive_differentiation = 0.20
                    st.session_state.weight_manager.current_weights.brand_monopoly = 0.18
                    st.session_state.weight_manager.current_weights.barriers_to_entry = 0.15
                    st.session_state.weight_manager.current_weights.technology_moats = 0.10
                    st.session_state.weight_manager.current_weights.key_growth_drivers = 0.08
                    st.session_state.weight_manager.current_weights.transformation_potential = 0.02
                    st.session_state.weight_manager.current_weights.red_flags = -0.015
                    st.session_state.weight_manager.current_weights.major_risk_factors = -0.005

                elif philosophy_num == "4":  # Risk-Aware
                    st.session_state.weight_manager.current_weights.major_risk_factors = -0.15
                    st.session_state.weight_manager.current_weights.red_flags = -0.10
                    st.session_state.weight_manager.current_weights.barriers_to_entry = 0.25
                    st.session_state.weight_manager.current_weights.switching_costs = 0.20
                    st.session_state.weight_manager.current_weights.management_quality = 0.15
                    st.session_state.weight_manager.current_weights.brand_monopoly = 0.10
                    st.session_state.weight_manager.current_weights.competitive_differentiation = 0.10
                    st.session_state.weight_manager.current_weights.economies_of_scale = 0.05

                elif philosophy_num == "5":  # Tech Focus
                    st.session_state.weight_manager.current_weights.technology_moats = 0.30
                    st.session_state.weight_manager.current_weights.network_effects = 0.20
                    st.session_state.weight_manager.current_weights.platform_expansion = 0.15
                    st.session_state.weight_manager.current_weights.transformation_potential = 0.15
                    st.session_state.weight_manager.current_weights.competitive_differentiation = 0.12
                    st.session_state.weight_manager.current_weights.key_growth_drivers = 0.05
                    st.session_state.weight_manager.current_weights.barriers_to_entry = 0.02
                    st.session_state.weight_manager.current_weights.major_risk_factors = -0.015
                    st.session_state.weight_manager.current_weights.red_flags = -0.005

                # Update session state
                st.session_state.current_weights = st.session_state.weight_manager.current_weights
                st.success(f"✅ Applied {philosophy} weights")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error applying philosophy: {str(e)}")

        # Manual weight adjustment
        st.markdown("**✏️ Manual Weight Adjustment**")

        if st.session_state.current_weights and st.button("🔧 Edit Individual Weights"):
            st.session_state.show_weight_editor = not st.session_state.show_weight_editor

        if st.session_state.get('show_weight_editor', False) and st.session_state.current_weights:

            # Category selection for editing
            editing_category = st.selectbox(
                "Select category to edit",
                ["Competitive Moats", "Strategic Insights", "Risk Factors"],
                key="editing_category_select"
            )

            weights = st.session_state.current_weights

            if editing_category == "Competitive Moats":
                st.write("**Edit Competitive Moats Weights:**")
                new_barriers = st.number_input("Barriers to Entry", min_value=0.0, max_value=1.0, value=weights.barriers_to_entry, step=0.001, format="%.3f")
                new_brand = st.number_input("Brand Monopoly", min_value=0.0, max_value=1.0, value=weights.brand_monopoly, step=0.001, format="%.3f")
                new_scale = st.number_input("Economies of Scale", min_value=0.0, max_value=1.0, value=weights.economies_of_scale, step=0.001, format="%.3f")
                new_network = st.number_input("Network Effects", min_value=0.0, max_value=1.0, value=weights.network_effects, step=0.001, format="%.3f")
                new_switching = st.number_input("Switching Costs", min_value=0.0, max_value=1.0, value=weights.switching_costs, step=0.001, format="%.3f")

                if st.button("💾 Update Competitive Moats"):
                    weights.barriers_to_entry = new_barriers
                    weights.brand_monopoly = new_brand
                    weights.economies_of_scale = new_scale
                    weights.network_effects = new_network
                    weights.switching_costs = new_switching
                    st.success("✅ Competitive Moats updated")
                    st.rerun()

            elif editing_category == "Strategic Insights":
                st.write("**Edit Strategic Insights Weights:**")
                new_diff = st.number_input("Competitive Differentiation", min_value=0.0, max_value=1.0, value=weights.competitive_differentiation, step=0.001, format="%.3f")
                new_tech = st.number_input("Technology Moats", min_value=0.0, max_value=1.0, value=weights.technology_moats, step=0.001, format="%.3f")
                new_timing = st.number_input("Market Timing", min_value=0.0, max_value=1.0, value=weights.market_timing, step=0.001, format="%.3f")
                new_mgmt = st.number_input("Management Quality", min_value=0.0, max_value=1.0, value=weights.management_quality, step=0.001, format="%.3f")
                new_growth = st.number_input("Key Growth Drivers", min_value=0.0, max_value=1.0, value=weights.key_growth_drivers, step=0.001, format="%.3f")
                new_transform = st.number_input("Transformation Potential", min_value=0.0, max_value=1.0, value=weights.transformation_potential, step=0.001, format="%.3f")
                new_platform = st.number_input("Platform Expansion", min_value=0.0, max_value=1.0, value=weights.platform_expansion, step=0.001, format="%.3f")

                if st.button("💾 Update Strategic Insights"):
                    weights.competitive_differentiation = new_diff
                    weights.technology_moats = new_tech
                    weights.market_timing = new_timing
                    weights.management_quality = new_mgmt
                    weights.key_growth_drivers = new_growth
                    weights.transformation_potential = new_transform
                    weights.platform_expansion = new_platform
                    st.success("✅ Strategic Insights updated")
                    st.rerun()

            elif editing_category == "Risk Factors":
                st.write("**Edit Risk Factors (Negative Weights):**")
                new_major_risk = st.number_input("Major Risk Factors", min_value=-1.0, max_value=0.0, value=weights.major_risk_factors, step=0.001, format="%.3f")
                new_red_flags = st.number_input("Red Flags", min_value=-1.0, max_value=0.0, value=weights.red_flags, step=0.001, format="%.3f")

                if st.button("💾 Update Risk Factors"):
                    weights.major_risk_factors = new_major_risk
                    weights.red_flags = new_red_flags
                    st.success("✅ Risk Factors updated")
                    st.rerun()

        # Weight approval
        if st.session_state.current_weights:
            st.markdown("**✅ Weight Approval**")

            col_approve, col_reset = st.columns(2)

            with col_approve:
                if st.button("✅ Approve Current Weights", type="primary"):
                    # Normalize weights before saving
                    normalized_weights = st.session_state.current_weights.normalize_weights()

                    # Save approved weights
                    weights_dict = {
                        'timestamp': time.time(),
                        'approved_by': 'streamlit_user',
                        'weights': {
                            'barriers_to_entry': normalized_weights.barriers_to_entry,
                            'brand_monopoly': normalized_weights.brand_monopoly,
                            'economies_of_scale': normalized_weights.economies_of_scale,
                            'network_effects': normalized_weights.network_effects,
                            'switching_costs': normalized_weights.switching_costs,
                            'competitive_differentiation': normalized_weights.competitive_differentiation,
                            'technology_moats': normalized_weights.technology_moats,
                            'market_timing': normalized_weights.market_timing,
                            'management_quality': normalized_weights.management_quality,
                            'key_growth_drivers': normalized_weights.key_growth_drivers,
                            'transformation_potential': normalized_weights.transformation_potential,
                            'platform_expansion': normalized_weights.platform_expansion,
                            'major_risk_factors': normalized_weights.major_risk_factors,
                            'red_flags': normalized_weights.red_flags
                        }
                    }

                    weights_file = qual_agent_path / "approved_weights.json"
                    with open(weights_file, 'w') as f:
                        json.dump(weights_dict, f, indent=2)

                    st.success(f"✅ Weights approved and saved to approved_weights.json")

            with col_reset:
                if st.button("🔄 Reset to Defaults"):
                    st.session_state.weight_manager = InteractiveWeightManager()
                    st.session_state.current_weights = st.session_state.weight_manager.current_weights
                    st.session_state.show_weight_editor = False
                    st.success("✅ Reset to default weights")
                    st.rerun()

def analysis_execution_section():
    """Analysis execution section"""
    st.header("🚀 Run Enhanced Analysis")

    # Check prerequisites
    prerequisites_met = True

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Analysis Configuration")

        # Company selection
        if st.session_state.uploaded_companies is not None:
            companies_df = st.session_state.uploaded_companies
            total_companies = len(companies_df)

            # Create batch processing options
            batch_options = []
            batch_options.append("Select All Companies")

            # Add granular batch options
            if total_companies >= 10:
                batch_options.append("Run First 10")
            if total_companies >= 25:
                batch_options.append("Run First 25")
            if total_companies >= 50:
                batch_options.append("Run First 50")
            if total_companies >= 100:
                batch_options.append("Run First 100")

            # Add "next batch" options if there are more than 50 companies
            if total_companies > 50:
                batch_options.append("Run Next 25 (51-75)")
                if total_companies > 75:
                    batch_options.append("Run Next 25 (76-100)")
                if total_companies > 100:
                    batch_options.append("Run Next 50 (101-150)")

            # Add individual company options
            company_options = batch_options + ["--- Individual Companies ---"] + companies_df['ticker'].tolist()

            selected_option = st.selectbox(
                "Select company or batch to analyze",
                options=company_options,
                help="Choose a batch option for multiple companies or select an individual company"
            )

            # Store the selection in session state
            if selected_option == "Select All Companies":
                selected_company = None
                st.session_state.batch_analysis = True
                st.session_state.batch_companies = companies_df['ticker'].tolist()
                st.info(f"📊 Batch analysis selected for all {len(companies_df)} companies")
            elif selected_option == "Run First 10":
                selected_company = None
                st.session_state.batch_analysis = True
                st.session_state.batch_companies = companies_df['ticker'].tolist()[:10]
                st.info(f"📊 Batch analysis selected for first 10 companies: {', '.join(companies_df['ticker'].tolist()[:10])}")
            elif selected_option == "Run First 25":
                selected_company = None
                st.session_state.batch_analysis = True
                st.session_state.batch_companies = companies_df['ticker'].tolist()[:25]
                st.info(f"📊 Batch analysis selected for first 25 companies")
            elif selected_option == "Run First 50":
                selected_company = None
                st.session_state.batch_analysis = True
                st.session_state.batch_companies = companies_df['ticker'].tolist()[:50]
                st.info(f"📊 Batch analysis selected for first 50 companies")
            elif selected_option == "Run First 100":
                selected_company = None
                st.session_state.batch_analysis = True
                st.session_state.batch_companies = companies_df['ticker'].tolist()[:100]
                st.info(f"📊 Batch analysis selected for first 100 companies")
            elif selected_option == "Run Next 25 (51-75)":
                selected_company = None
                st.session_state.batch_analysis = True
                st.session_state.batch_companies = companies_df['ticker'].tolist()[50:75]
                st.info(f"📊 Batch analysis selected for companies 51-75 (25 companies)")
            elif selected_option == "Run Next 25 (76-100)":
                selected_company = None
                st.session_state.batch_analysis = True
                st.session_state.batch_companies = companies_df['ticker'].tolist()[75:100]
                st.info(f"📊 Batch analysis selected for companies 76-100 (25 companies)")
            elif selected_option == "Run Next 50 (101-150)":
                selected_company = None
                st.session_state.batch_analysis = True
                st.session_state.batch_companies = companies_df['ticker'].tolist()[100:150]
                st.info(f"📊 Batch analysis selected for companies 101-150 (50 companies)")
            elif selected_option == "--- Individual Companies ---":
                st.warning("⚠️ Please select a specific company from the dropdown")
                selected_company = None
                st.session_state.batch_analysis = False
                st.session_state.batch_companies = []
            else:
                selected_company = selected_option
                st.session_state.batch_analysis = False
                st.session_state.batch_companies = []
        else:
            st.warning("⚠️ Please upload company data in the Data Input tab first")
            selected_company = None
            st.session_state.batch_analysis = False
            prerequisites_met = False

        # User ID
        user_id = st.text_input(
            "User ID",
            value="chenHX",
            help="Enter your user ID for analysis tracking"
        )

        # Analysis type
        analysis_type = st.selectbox(
            "Analysis Type",
            ["expert_guided", "comprehensive", "quick"],
            index=0,
            help="expert_guided includes weight approval and feedback collection"
        )

        # Model selection check
        if not st.session_state.selected_models:
            st.warning("⚠️ Please test APIs and select models in the API Testing tab")
            prerequisites_met = False
        else:
            st.success(f"✅ {len(st.session_state.selected_models)} models selected")

        # Weight configuration check
        weights_file = qual_agent_path / "approved_weights.json"
        if not weights_file.exists():
            st.warning("⚠️ Please configure and approve weights in the Weight Management tab")
            prerequisites_met = False
        else:
            st.success("✅ Weights configuration approved")

    with col2:
        st.subheader("💰 Cost Estimation")

        if st.button("💸 Estimate Analysis Cost"):
            if selected_company and st.session_state.selected_models:
                # Rough cost estimation
                num_models = len(st.session_state.selected_models)
                estimated_cost = num_models * 0.05  # Rough estimate
                estimated_time = num_models * 15  # seconds

                st.info(f"""
                **Cost Estimation for {selected_company}:**
                - Models: {num_models}
                - Estimated cost: ${estimated_cost:.3f}
                - Estimated time: {estimated_time}s
                """)
            else:
                st.warning("Please select a company and models first")

        st.subheader("⚙️ Advanced Options")

        # Additional parameters
        max_concurrent = st.slider("Max Concurrent Models", 1, 5, 3)
        lookback_months = st.slider("Lookback Months", 6, 36, 24)

        # Geographic focus
        geographies = st.multiselect(
            "Geographic Focus",
            ["US", "Global", "Europe", "Asia", "Emerging Markets"],
            default=["US", "Global"]
        )

    # Run analysis button
    st.subheader("🚀 Execute Analysis")

    if prerequisites_met and (selected_company or st.session_state.get('batch_analysis', False)):
        if st.button("🚀 Run Analysis", type="primary"):
            if st.session_state.get('batch_analysis', False):
                # Batch analysis
                batch_companies = st.session_state.get('batch_companies', [])
                run_batch_analysis(batch_companies, user_id, analysis_type, max_concurrent, lookback_months, geographies)
            else:
                # Single company analysis
                run_analysis(selected_company, user_id, analysis_type, max_concurrent, lookback_months, geographies)
    else:
        st.error("❌ Please complete all prerequisites before running analysis")

def run_analysis(company, user_id, analysis_type, max_concurrent, lookback_months, geographies):
    """Execute the analysis using the proven subprocess.run approach"""

    with st.spinner(f"🔄 Running enhanced analysis for {company}... This may take several minutes."):
        try:
            # Construct command
            cmd = [
                sys.executable,
                "run_enhanced_analysis.py",  # Simplified path since we use cwd
                "--user-id", user_id,
                "--company", company,
                "--analysis-type", analysis_type,
                "--custom-weights", "approved_weights.json",  # Relative path
                "--max-concurrent", str(max_concurrent),
                "--lookback-months", str(lookback_months),
                "--geographies", ",".join(geographies),
                "--auto-approve"  # Skip interactive cost approval
            ]

            # Add selected models
            if st.session_state.selected_models:
                cmd.extend(["--models", ",".join(st.session_state.selected_models)])

            # Display command being executed (with full paths for user info)
            display_cmd = [
                sys.executable,
                str(qual_agent_path / "run_enhanced_analysis.py"),
                "--user-id", user_id,
                "--company", company,
                "--analysis-type", analysis_type,
                "--custom-weights", str(qual_agent_path / "approved_weights.json"),
                "--max-concurrent", str(max_concurrent),
                "--lookback-months", str(lookback_months),
                "--geographies", ",".join(geographies),
                "--auto-approve"
            ]
            if st.session_state.selected_models:
                display_cmd.extend(["--models", ",".join(st.session_state.selected_models)])

            st.code(" ".join(display_cmd))

            # Use the working subprocess.run approach (like the reference)
            result = subprocess.run(
                cmd,
                cwd=qual_agent_path,  # Set working directory
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            # Display output
            output_container = st.empty()
            all_output = []

            if result.stdout:
                all_output.append("=== STDOUT ===")
                all_output.extend(result.stdout.strip().split('\n'))

            if result.stderr:
                all_output.append("=== STDERR ===")
                all_output.extend(result.stderr.strip().split('\n'))

            if all_output:
                output_container.text('\n'.join(all_output[-50:]))  # Show last 50 lines

            if result.returncode == 0:
                st.success(f"[SUCCESS] Analysis completed successfully for {company}!")

                # Store results info
                st.session_state.analysis_results[company] = {
                    'timestamp': time.time(),
                    'user_id': user_id,
                    'analysis_type': analysis_type,
                    'status': 'completed'
                }

                # Switch to results tab
                st.info("Check the Results & Downloads tab to view and download your analysis results")

            else:
                st.error(f"[ERROR] Analysis failed with exit code {result.returncode}")
                if result.stderr:
                    st.error(f"Error details: {result.stderr}")

        except subprocess.TimeoutExpired:
            st.error("❌ Analysis timed out after 10 minutes")
        except Exception as e:
            st.error(f"❌ Error running analysis: {str(e)}")

def run_batch_analysis(companies, user_id, analysis_type, max_concurrent, lookback_months, geographies):
    """Execute batch analysis for multiple companies"""

    if not companies:
        st.error("❌ No companies selected for batch analysis")
        return

    st.info(f"🚀 Starting batch analysis for {len(companies)} companies...")

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"🔄 Initializing batch analysis for {len(companies)} companies...")

    try:
        # Run batch analysis with the improved batch timestamp functionality
        cmd = [
            sys.executable,
            "run_enhanced_analysis.py",
            "--user-id", user_id,
            "--batch",
            "--companies", ",".join(companies),
            "--analysis-type", analysis_type,
            "--custom-weights", "approved_weights.json",
            "--max-concurrent", str(max_concurrent),
            "--lookback-months", str(lookback_months),
            "--auto-approve"
        ]

        if geographies:
            cmd.extend(["--geographies", ",".join(geographies)])

        # Add selected models
        if st.session_state.selected_models:
            cmd.extend(["--models", ",".join(st.session_state.selected_models)])

        # Display command being executed (for debugging)
        display_cmd = [
            sys.executable,
            str(qual_agent_path / "run_enhanced_analysis.py"),
            "--user-id", user_id,
            "--batch",
            "--companies", ",".join(companies),
            "--analysis-type", analysis_type,
            "--custom-weights", str(qual_agent_path / "approved_weights.json"),
            "--max-concurrent", str(max_concurrent),
            "--lookback-months", str(lookback_months),
            "--auto-approve"
        ]

        if geographies:
            display_cmd.extend(["--geographies", ",".join(geographies)])

        if st.session_state.selected_models:
            display_cmd.extend(["--models", ",".join(st.session_state.selected_models)])

        with st.expander("🔧 Command Details", expanded=False):
            st.code(" ".join(display_cmd))

        # Update progress to show execution
        progress_bar.progress(0.1)
        status_text.text("🚀 Running enhanced batch analysis with shared timestamp...")

        result = subprocess.run(
            cmd,
            cwd=qual_agent_path,
            capture_output=True,
            text=True,
            timeout=None  # No timeout limit for batch processing (can run for hours)
        )

        progress_bar.progress(0.9)

        if result.returncode == 0:
            # Success
            st.session_state.batch_results = {}
            for company in companies:
                st.session_state.batch_results[company] = {
                    'status': 'success',
                    'timestamp': time.time(),
                    'output': result.stdout
                }

            progress_bar.progress(1.0)
            status_text.text("📊 Batch analysis completed successfully!")

            st.success(f"🎉 Batch Analysis Complete!")
            st.info(f"✅ All {len(companies)} companies analyzed with shared timestamp for easy grouping")

            # Parse results with same timestamp
            parse_batch_results(companies)

        else:
            st.error("❌ Batch analysis failed")
            st.error(f"Error details: {result.stderr}")

    except subprocess.TimeoutExpired:
        st.error("❌ Batch analysis timed out after 20 minutes")
        status_text.text("⏱️ Batch analysis timed out")

    except Exception as e:
        st.error(f"❌ Error running batch analysis: {str(e)}")
        status_text.text(f"❌ Error: {str(e)}")

    # Trigger results display
    st.session_state.show_batch_results = True

def parse_batch_results(companies):
    """Parse batch analysis results from generated files"""

    # Clear previous results to avoid showing stale data
    st.session_state.parsed_batch_results = []

    st.write(f"🔍 Parsing results for {len(companies)} companies...")
    parsed_count = 0

    for company in companies:
        if st.session_state.batch_results.get(company, {}).get('status') == 'success':
            try:
                # Find the most recent results files for this company
                results_pattern = qual_agent_path / "results" / f"multi_llm_analysis_{company}_*.json"

                import glob
                result_files = glob.glob(str(results_pattern))

                if result_files:
                    # Get the most recent file
                    latest_file = max(result_files, key=os.path.getctime)

                    with open(latest_file, 'r') as f:
                        result_data = json.load(f)

                    # Extract key information from the correct JSON structure
                    composite_score = result_data.get('composite_score', {}).get('score', 0.0)
                    composite_confidence = result_data.get('composite_score', {}).get('confidence', 0.0)

                    # Extract individual LLM scores from individual_model_results
                    llm_scores = {}
                    individual_model_results = result_data.get('individual_model_results', {})

                    for llm_name, llm_data in individual_model_results.items():
                        # Skip models that had errors
                        if 'error' in llm_data:
                            continue
                        # Get parsed output which contains the scores
                        parsed_output = llm_data.get('parsed_output', {})
                        if 'composite_score' in parsed_output:
                            llm_scores[f"{llm_name}_score"] = parsed_output.get('composite_score', 0.0)
                            llm_scores[f"{llm_name}_confidence"] = parsed_output.get('composite_confidence', 0.0)

                    # Get company metadata from uploaded data
                    company_name = company  # Default to ticker
                    sector = "N/A"
                    industry = "N/A"
                    if st.session_state.uploaded_companies is not None:
                        company_row = st.session_state.uploaded_companies[
                            st.session_state.uploaded_companies['ticker'] == company
                        ]
                        if not company_row.empty:
                            if 'company_name' in company_row.columns:
                                company_name = company_row.iloc[0]['company_name']
                            if 'sector' in company_row.columns:
                                sector = company_row.iloc[0]['sector']
                            if 'industry' in company_row.columns:
                                industry = company_row.iloc[0]['industry']

                    # Add to parsed results
                    result_entry = {
                        'ticker': company,
                        'company_name': company_name,
                        'sector': sector,
                        'industry': industry,
                        'average_composite_score': composite_score,
                        'average_composite_confidence': composite_confidence,
                        'result_file': latest_file,
                        **llm_scores
                    }

                    st.session_state.parsed_batch_results.append(result_entry)
                    parsed_count += 1

                else:
                    st.write(f"⚠️ No result files found for {company}")

            except Exception as e:
                st.error(f"Error parsing results for {company}: {str(e)}")
        else:
            batch_status = st.session_state.batch_results.get(company, {}).get('status', 'unknown')
            st.write(f"⏭️ Skipping {company} (status: {batch_status})")

    st.success(f"✅ Successfully parsed {parsed_count} out of {len(companies)} companies")

def results_section():
    """Enhanced results and downloads section with batch analysis support"""
    st.header("📊 Results & Downloads")

    # Debug information
    show_batch = st.session_state.get('show_batch_results', False)
    parsed_results = st.session_state.get('parsed_batch_results', [])
    batch_results = st.session_state.get('batch_results', {})

    # Show debug info in expandable section
    with st.expander("🔍 Debug Information", expanded=False):
        st.write(f"Show batch results flag: {show_batch}")
        st.write(f"Parsed batch results count: {len(parsed_results)}")
        st.write(f"Batch results status count: {len(batch_results)}")
        if batch_results:
            status_counts = {}
            for company, result in batch_results.items():
                status = result.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            st.write(f"Status breakdown: {status_counts}")

    # Manual refresh button
    if st.button("🔄 Refresh Results", help="Manually reparse batch analysis results"):
        if batch_results:
            companies = list(batch_results.keys())
            st.write(f"Attempting to reparse results for {len(companies)} companies...")
            parse_batch_results(companies)
            st.session_state.show_batch_results = True
            st.rerun()

    # Check for batch results
    if show_batch:
        if parsed_results:
            display_batch_results_table()
        else:
            st.warning("⚠️ Batch analysis completed but no results were parsed. Try the 'Refresh Results' button above.")
            # Try to auto-parse if we have batch_results but no parsed_results
            if batch_results:
                companies = list(batch_results.keys())
                st.write("🔄 Attempting to auto-parse results...")
                parse_batch_results(companies)
                if st.session_state.get('parsed_batch_results'):
                    st.rerun()

    # Check for individual analysis results
    results_dir = qual_agent_path / "results"
    individual_results_exist = results_dir.exists() and list(results_dir.glob("*"))
    if individual_results_exist and not show_batch:
        display_individual_results()
    elif not show_batch and not individual_results_exist:
        st.info("📋 No analysis results found. Run an analysis first.")

def display_batch_results_table():
    """Display enhanced batch results table with ranking and selective download"""
    st.subheader("📊 Batch Analysis Results - Company Rankings")

    batch_results = st.session_state.get('parsed_batch_results', [])

    if not batch_results:
        st.warning("⚠️ No batch results to display")
        return

    # Convert to DataFrame and sort by average composite score (descending)
    results_df = pd.DataFrame(batch_results)
    results_df = results_df.sort_values('average_composite_score', ascending=False).reset_index(drop=True)
    results_df.index = results_df.index + 1  # Start ranking from 1

    # Display the ranking table
    st.markdown("### 🏆 Company Rankings by Composite Score")

    # Select columns to display
    display_columns = ['ticker', 'company_name', 'sector', 'industry', 'average_composite_score', 'average_composite_confidence']

    # Add individual LLM score columns if they exist
    llm_columns = [col for col in results_df.columns if col.endswith('_score') and not col.startswith('average')]
    display_columns.extend(llm_columns)

    # Filter to only include columns that actually exist in the DataFrame
    display_columns = [col for col in display_columns if col in results_df.columns]

    # Format the DataFrame for display
    display_df = results_df[display_columns].copy()

    # Format numeric columns
    numeric_columns = [col for col in display_df.columns if col not in ['ticker', 'company_name']]
    for col in numeric_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(3)

    # Rename columns for better display
    display_df = display_df.rename(columns={
        'ticker': 'Ticker',
        'company_name': 'Company Name',
        'sector': 'Sector',
        'industry': 'Industry',
        'average_composite_score': 'Avg Score',
        'average_composite_confidence': 'Avg Confidence'
    })

    # Add rank column at the beginning
    display_df.insert(0, 'Rank', range(1, len(display_df) + 1))

    # Display the table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=min(400, len(display_df) * 35 + 50)  # Dynamic height
    )

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Companies", len(results_df))
    with col2:
        st.metric("Highest Score", f"{results_df['average_composite_score'].max():.3f}")
    with col3:
        st.metric("Average Score", f"{results_df['average_composite_score'].mean():.3f}")
    with col4:
        st.metric("Lowest Score", f"{results_df['average_composite_score'].min():.3f}")

    # Selective download section
    st.subheader("📥 Selective Download")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Multi-select for top companies
        num_companies = st.slider(
            "Select top N companies for detailed reports",
            min_value=1,
            max_value=len(results_df),
            value=min(3, len(results_df)),
            help="Select how many top-ranked companies you want to download reports for"
        )

        top_companies = results_df.head(num_companies)
        selected_tickers = top_companies['ticker'].tolist()

        st.info(f"📋 Selected top {num_companies} companies: {', '.join(selected_tickers)}")

        # Show selected companies details
        selected_df = top_companies[['ticker', 'company_name', 'average_composite_score']].copy()
        selected_df.index = range(1, len(selected_df) + 1)
        st.dataframe(selected_df, use_container_width=True)

    with col2:
        # Download options
        st.markdown("**Download Options:**")

        if st.button("📊 Download Summary Table (CSV)", type="secondary"):
            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                file_name=f"company_rankings_{int(time.time())}.csv",
                mime="text/csv"
            )

        if st.button("📄 Download Top Companies Reports", type="primary"):
            download_selected_reports(selected_tickers)

        # Download all results
        if st.button("📦 Download All Results (ZIP)"):
            create_batch_download_zip(results_df['ticker'].tolist())

def download_selected_reports(selected_tickers):
    """Create downloadable package for selected companies"""
    if not selected_tickers:
        st.error("❌ No companies selected")
        return

    try:
        import zipfile
        import io

        # Create zip file in memory
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for ticker in selected_tickers:
                # Find all result files for this company
                import glob

                file_patterns = [
                    f"multi_llm_analysis_{ticker}_*.json",
                    f"multi_llm_scores_{ticker}_*.csv",
                    f"analysis_report_{ticker}_*.md",
                    f"enhanced_metadata_{ticker}_*.json"
                ]

                for pattern in file_patterns:
                    files = glob.glob(str(qual_agent_path / pattern))
                    for file_path in files:
                        if os.path.exists(file_path):
                            arc_name = f"{ticker}/{os.path.basename(file_path)}"
                            zip_file.write(file_path, arc_name)

        zip_buffer.seek(0)

        # Download button
        st.download_button(
            f"📥 Download {len(selected_tickers)} Company Reports",
            zip_buffer.getvalue(),
            file_name=f"top_{len(selected_tickers)}_companies_reports_{int(time.time())}.zip",
            mime="application/zip"
        )

        st.success(f"✅ Created download package for {len(selected_tickers)} companies")

    except Exception as e:
        st.error(f"❌ Error creating download package: {str(e)}")

def create_batch_download_zip(all_tickers):
    """Create downloadable zip for all analyzed companies"""
    try:
        import zipfile
        import io

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for ticker in all_tickers:
                # Similar logic as download_selected_reports
                import glob
                file_patterns = [
                    f"multi_llm_analysis_{ticker}_*.json",
                    f"multi_llm_scores_{ticker}_*.csv",
                    f"analysis_report_{ticker}_*.md",
                    f"enhanced_metadata_{ticker}_*.json"
                ]

                for pattern in file_patterns:
                    files = glob.glob(str(qual_agent_path / pattern))
                    for file_path in files:
                        if os.path.exists(file_path):
                            arc_name = f"{ticker}/{os.path.basename(file_path)}"
                            zip_file.write(file_path, arc_name)

        zip_buffer.seek(0)

        st.download_button(
            f"📥 Download All {len(all_tickers)} Company Reports",
            zip_buffer.getvalue(),
            file_name=f"all_companies_batch_analysis_{int(time.time())}.zip",
            mime="application/zip"
        )

        st.success(f"✅ Created download package for all {len(all_tickers)} companies")

    except Exception as e:
        st.error(f"❌ Error creating batch download: {str(e)}")

def display_individual_results():
    """Display individual analysis results (existing functionality)"""
    st.subheader("📁 Individual Analysis Results")

    results_dir = qual_agent_path / "results"
    result_files = list(results_dir.glob("*"))

    if not result_files:
        return

    # Group files by company and timestamp
    file_groups = {}
    for file_path in result_files:
        filename = file_path.name
        if "_" in filename:
            parts = filename.split("_")
            if len(parts) >= 3:
                company = parts[-2]
                timestamp = parts[-1].split(".")[0]
                key = f"{company}_{timestamp}"

                if key not in file_groups:
                    file_groups[key] = []
                file_groups[key].append(file_path)

    # Display individual results
    for analysis_key, files in file_groups.items():
        company, timestamp = analysis_key.rsplit("_", 1)

        with st.expander(f"📊 {company} - Individual Analysis", expanded=False):
            st.markdown(f"**Company**: {company}")
            st.markdown(f"**Timestamp**: {timestamp}")
            st.markdown(f"**Files**: {len(files)} result files")

            # Simple file download
            for file_path in files:
                with open(file_path, 'rb') as f:
                    file_content = f.read()

                st.download_button(
                    f"📥 {file_path.name}",
                    file_content,
                    file_name=file_path.name,
                    key=f"download_{file_path.name}_{timestamp}"
                )

if __name__ == "__main__":
    main()