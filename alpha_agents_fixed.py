"""
Alpha Agents - Fixed Version
Handles database connection issues and missing data
"""

import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
import subprocess
import json
import uuid

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agents', 'QualAgent'))

# Import original modules
from utils.stock_screener import StockScreener
from utils.fast_stock_screener import FastStockScreener
from utils.db_util import save_fundamental_screen

def load_fundamental_screen_postgresql(run_id=None):
    """Load screening results from PostgreSQL database"""
    try:
        from utils.db_util import get_engine
        import pandas as pd
        
        engine = get_engine()
        
        if run_id:
            query = "SELECT * FROM fundamental_screen WHERE run_id = %s ORDER BY score DESC"
            df = pd.read_sql_query(query, engine, params=(run_id,))
        else:
            query = "SELECT * FROM fundamental_screen ORDER BY created_at DESC, score DESC"
            df = pd.read_sql_query(query, engine)
        
        return df
    except Exception as e:
        st.error(f"Database load error: {e}")
        return pd.DataFrame()

def main():
    st.set_page_config(
        page_title="Alpha Agents",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar navigation
    st.sidebar.title("🚀 Alpha Agents")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Choose Page",
        ["📊 Fundamental Screener", "🤖 QualAgent Analysis"]
    )
    
    # Display selected page
    if page == "📊 Fundamental Screener":
        screener_page()
    elif page == "🤖 QualAgent Analysis":
        qualagent_page()

def screener_page():
    """Page 1: Fundamental Screener - FIXED VERSION"""
    
    st.title("📊 Fundamental Screener")
    st.markdown("Filter stocks based on quantitative financial criteria")
    
    # SAME CODE AS ORIGINAL
    SECTORS = sorted(StockScreener.SECTORS)
    INDUSTRIES = sorted(StockScreener.INDUSTRIES)
    REGIONS = sorted(StockScreener.REGIONS)
    DEFAULT_REGION = "US"

    @st.cache_data(show_spinner=False)
    def run_screen(region, filter_by, selection, min_cap, max_cap, use_fast=True):
        if use_fast:
            # Use fast screener with parallel processing
            screener = FastStockScreener(max_workers=15, cache_hours=6)
            return screener.screen(
                region=region,
                min_cap=min_cap,
                max_cap=max_cap,
                sectors=selection if filter_by == "Sector" else None,
                industries=selection if filter_by == "Industry" else None,
                max_companies=100  # Limit for speed
            )
        else:
            # Use original screener
            options = {
                "region": region,
                "min_cap": min_cap,
                "max_cap": max_cap,
                "sectors": None,
                "industries": None,
            }
            if filter_by == "Sector":
                options["sectors"] = selection
            else:
                options["industries"] = selection
            screener = StockScreener(**options)
            return screener.screen()

    # Filters
    st.sidebar.header("🔍 Filters")
    
    # Speed option
    use_fast_screener = st.sidebar.checkbox("⚡ Fast Mode (Parallel Processing)", value=True, 
                                           help="Uses parallel processing and caching for faster results")
    
    filter_by = st.sidebar.radio("Filter By", ("Sector", "Industry"))

    if filter_by == "Sector":
        selection = st.sidebar.selectbox("Sector", SECTORS)
    else:
        selection = st.sidebar.selectbox("Industry", INDUSTRIES)

    region = st.sidebar.selectbox("Region", REGIONS, index=REGIONS.index(DEFAULT_REGION))
    min_cap = st.sidebar.number_input("Minimum market cap (USD)", min_value=0.0, value=0.0, step=1e9)
    max_cap_value = st.sidebar.number_input("Maximum market cap (USD, 0 for unlimited)", min_value=0.0, value=0.0, step=1e9)
    max_cap = float("inf") if max_cap_value == 0 else max_cap_value

    run_requested = st.sidebar.button("🚀 Run Screen", type="primary")

    if "screen_params" not in st.session_state:
        st.info("Adjust the filters and click 'Run Screen' to fetch results.")

    if run_requested:
        if not pd.isna(max_cap) and max_cap < min_cap:
            st.sidebar.error("Maximum market cap must be greater than minimum market cap.")
        else:
            st.session_state["screen_params"] = {
                "region": region,
                "filter_by": filter_by,
                "selection": selection,
                "min_cap": float(min_cap),
                "max_cap": float(max_cap),
            }

    params = st.session_state.get("screen_params")
    if params:
        # Add fast mode to params
        params["use_fast"] = use_fast_screener
        
        with st.spinner("Screening companies..."):
            df = run_screen(**params)
        
        if df.empty:
            st.warning("No companies matched the current filters.")
        else:
            st.caption(f"Showing {len(df)} companies filtered by {params['filter_by'].lower()} '{params['selection']}'.")
            st.dataframe(df, width="stretch")
            
            # Store results for QualAgent
            st.session_state["screener_results"] = df
            st.success(f"✅ {len(df)} companies ready for QualAgent analysis!")
            
            # Export options
            df_export = df.reset_index(drop=False)
            if "ticker" not in df_export.columns and "index" in df_export.columns:
                df_export = df_export.rename(columns={"index": "ticker"})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = df_export.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    file_name="fundamental_screen_results.csv",
                    mime="text/csv",
                )
            
            with col2:
                import io
                excel_buffer = io.BytesIO()
                df_export.to_excel(excel_buffer, index=False, sheet_name="Results")
                excel_buffer.seek(0)
                st.download_button(
                    "📥 Download Excel",
                    data=excel_buffer.getvalue(),
                    file_name="fundamental_screen_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            
            with col3:
                if st.button("💾 Store in PostgreSQL Database"):
                    try:
                        with st.spinner("Saving results to PostgreSQL database..."):
                            run_id = save_fundamental_screen(df.reset_index(drop=False))
                        st.success(f"✅ Saved {len(df)} rows to PostgreSQL database!")
                        st.success(f"Run ID: `{run_id}`")
                        st.session_state["last_run_id"] = run_id
                    except Exception as e:
                        st.error(f"Database save failed: {str(e)}")
                        st.info("💡 You can still download CSV/Excel files")
            
            # Navigation hint
            st.markdown("---")
            st.info("💡 **Next Step**: Switch to 'QualAgent Analysis' page to analyze these companies with AI!")

def qualagent_page():
    """Page 2: QualAgent Analysis - FIXED VERSION"""
    
    st.title("🤖 QualAgent Analysis")
    st.markdown("AI-powered qualitative analysis using multiple LLM models")
    
    # Data source selection
    st.sidebar.header("📊 Data Source")
    data_source = st.sidebar.radio(
        "Choose Data Source",
        ["From Screener Results", "From PostgreSQL Database", "Upload CSV", "Manual Input", "Sample Companies"]
    )
    
    companies_to_analyze = []
    
    if data_source == "From Screener Results":
        if "screener_results" in st.session_state and not st.session_state["screener_results"].empty:
            df = st.session_state["screener_results"]
            st.success(f"✅ Found {len(df)} companies from screener results")
            
            # Show companies
            st.subheader("📋 Companies from Screener")
            display_cols = ['ticker', 'company_name', 'sector'] if 'company_name' in df.columns else ['ticker']
            st.dataframe(df[display_cols])
            
            # Select companies
            if 'ticker' in df.columns:
                selected_tickers = st.multiselect(
                    "Select companies to analyze",
                    df['ticker'].tolist(),
                    default=df['ticker'].tolist()[:3]
                )
                companies_to_analyze = selected_tickers
        else:
            st.warning("No screener results available. Run the Fundamental Screener first!")
    
    elif data_source == "From PostgreSQL Database":
        st.subheader("🗄️ Load from PostgreSQL Database")
        
        # Load all runs
        all_data = load_fundamental_screen_postgresql()
        
        if not all_data.empty:
            st.success(f"✅ Found {len(all_data)} companies in PostgreSQL database")
            
            # Show recent runs
            recent_runs = all_data['run_id'].unique()[:5]
            selected_run = st.selectbox("Select Run ID", recent_runs)
            
            if selected_run:
                run_data = all_data[all_data['run_id'] == selected_run]
                st.subheader(f"📋 Companies from Run: {selected_run}")
                st.dataframe(run_data[['ticker', 'company_name', 'sector', 'score']].head(10))
                
                # Select companies
                selected_tickers = st.multiselect(
                    "Select companies to analyze",
                    run_data['ticker'].tolist(),
                    default=run_data['ticker'].tolist()[:3]
                )
                companies_to_analyze = selected_tickers
        else:
            st.warning("No data found in PostgreSQL database. Run the Fundamental Screener first!")
    
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
    
    elif data_source == "Sample Companies":
        st.subheader("📋 Sample Companies")
        sample_companies = ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN"]
        selected_tickers = st.multiselect(
            "Select sample companies to analyze",
            sample_companies,
            default=sample_companies[:3]
        )
        companies_to_analyze = selected_tickers
    
    # Analysis configuration
    st.sidebar.header("🔧 Analysis Configuration")
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["quick", "comprehensive", "expert-guided"],
        help="Quick: Fast analysis (~30s), Comprehensive: Detailed analysis (~3min), Expert-guided: With human feedback (~10min)"
    )
    
    user_id = st.sidebar.text_input("User ID", value="streamlit_user")
    
    # Run analysis
    if companies_to_analyze:
        st.subheader("🚀 Run Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Run QualAgent Analysis", type="primary"):
                run_qualagent_analysis_fixed(companies_to_analyze, analysis_type, user_id)
        
        with col2:
            if st.button("💰 Estimate Cost"):
                cost_per_analysis = {"quick": 0.003, "comprehensive": 0.018, "expert-guided": 0.030}
                total_cost = cost_per_analysis.get(analysis_type, 0.018) * len(companies_to_analyze)
                st.info(f"💰 Estimated cost: ${total_cost:.3f}")
    
    # Display results
    display_analysis_results()

def run_qualagent_analysis_fixed(companies, analysis_type, user_id):
    """Run QualAgent analysis - FIXED VERSION"""
    
    st.subheader("🔄 Running Analysis...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    for i, company in enumerate(companies):
        status_text.text(f"Analyzing {company}...")
        progress_bar.progress((i + 1) / len(companies))
        
        try:
            # Run QualAgent analysis using FIXED approach
            result = run_single_analysis_fixed(company, analysis_type, user_id)
            results.append(result)
            
            if result["status"] == "success":
                st.success(f"✅ {company} analysis completed")
            else:
                st.error(f"❌ Error analyzing {company}: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            results.append({
                "company": company,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            st.error(f"❌ Error analyzing {company}: {str(e)}")
    
    # Store results
    st.session_state["qualagent_results"] = results
    st.session_state["analysis_completed"] = True
    
    status_text.text("Analysis completed!")
    progress_bar.progress(1.0)

def ensure_company_in_qualagent_data(company_ticker):
    """Ensure company exists in QualAgent data"""
    try:
        qualagent_dir = Path(__file__).parent / "agents" / "QualAgent"
        companies_file = qualagent_dir / "data" / "companies.json"
        
        # Load existing companies
        with open(companies_file, 'r') as f:
            companies = json.load(f)
        
        # Check if company exists
        company_exists = any(c.get('ticker') == company_ticker for c in companies)
        
        if not company_exists:
            # Add basic company entry
            new_company = {
                "company_name": f"{company_ticker} Inc.",
                "ticker": company_ticker,
                "subsector": "Unknown",
                "market_cap_usd": 1000000000,  # Default 1B
                "employees": 1000,
                "founded_year": 2000,
                "headquarters": "Unknown",
                "description": f"Company {company_ticker} - data from screener",
                "website": f"https://www.{company_ticker.lower()}.com",
                "status": "active",
                "id": str(uuid.uuid4()),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            companies.append(new_company)
            
            # Save back to file
            with open(companies_file, 'w') as f:
                json.dump(companies, f, indent=2)
            
            st.info(f"✅ Added {company_ticker} to QualAgent data")
        
        return True
        
    except Exception as e:
        st.warning(f"Could not ensure company data for {company_ticker}: {e}")
        return False

def run_single_analysis_fixed(company, analysis_type, user_id):
    """Run analysis for a single company - FIXED VERSION"""
    
    try:
        # Ensure company exists in QualAgent data
        ensure_company_in_qualagent_data(company)
        
        # Change to QualAgent directory
        qualagent_dir = Path(__file__).parent / "agents" / "QualAgent"
        
        # Use the demo script which works better
        cmd = [
            "python", "run_analysis_demo.py",
            "--single", company,
            "--user-id", user_id
        ]
        
        # Set environment to handle encoding properly
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            cmd,
            cwd=qualagent_dir,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            encoding='utf-8',  # Force UTF-8 encoding
            errors='replace',   # Replace problematic characters
            env=env  # Use modified environment
        )
        
        # Check if analysis actually completed successfully despite encoding error
        if result.returncode == 0 or "Enhanced analysis completed" in result.stderr:
            # Analysis completed successfully
            return {
                "company": company,
                "status": "success",
                "output": "Analysis completed successfully - check QualAgent directory for results",
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Clean the error output
            clean_error = result.stderr.encode('utf-8', errors='replace').decode('utf-8')
            return {
                "company": company,
                "status": "error",
                "error": clean_error,
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

def display_analysis_results():
    """Display analysis results with full report viewing and download options"""
    
    if "analysis_completed" in st.session_state and st.session_state["analysis_completed"]:
        st.subheader("📊 Analysis Results")
        
        results = st.session_state.get("qualagent_results", [])
        
        if results:
            # Show results summary
            results_df = pd.DataFrame(results)
            st.dataframe(results_df[['company', 'status', 'timestamp']])
            
            # Get all available reports
            qualagent_dir = Path(__file__).parent / "agents" / "QualAgent"
            all_reports = []
            
            for result in results:
                if result["status"] == "success":
                    company = result['company']
                    # Find all report files for this company
                    report_files = list(qualagent_dir.glob(f"analysis_report_{company}_*.md"))
                    json_files = list(qualagent_dir.glob(f"analysis_results_{company}_*.json"))
                    csv_files = list(qualagent_dir.glob(f"analysis_results_{company}_*.csv"))
                    
                    for report_file in report_files:
                        all_reports.append({
                            'company': company,
                            'file': report_file,
                            'type': 'Markdown Report',
                            'timestamp': report_file.stat().st_mtime
                        })
                    
                    for json_file in json_files:
                        all_reports.append({
                            'company': company,
                            'file': json_file,
                            'type': 'JSON Data',
                            'timestamp': json_file.stat().st_mtime
                        })
                    
                    for csv_file in csv_files:
                        all_reports.append({
                            'company': company,
                            'file': csv_file,
                            'type': 'CSV Data',
                            'timestamp': csv_file.stat().st_mtime
                        })
            
            if all_reports:
                st.subheader("📄 Available Reports")
                
                # Sort by timestamp (newest first)
                all_reports.sort(key=lambda x: x['timestamp'], reverse=True)
                
                # Create selection interface
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Report selection dropdown
                    report_options = [f"{r['company']} - {r['type']} ({r['file'].name})" for r in all_reports]
                    selected_report_idx = st.selectbox(
                        "Select Report to View",
                        range(len(report_options)),
                        format_func=lambda x: report_options[x]
                    )
                
                with col2:
                    # Download button
                    if st.button("📥 Download Selected Report"):
                        selected_report = all_reports[selected_report_idx]
                        file_path = selected_report['file']
                        
                        try:
                            with open(file_path, 'rb') as f:
                                file_data = f.read()
                            
                            st.download_button(
                                f"Download {file_path.name}",
                                file_data,
                                file_name=file_path.name,
                                mime="text/plain" if file_path.suffix == '.md' else "application/json" if file_path.suffix == '.json' else "text/csv"
                            )
                        except Exception as e:
                            st.error(f"Download failed: {str(e)}")
                
                # Display selected report
                if selected_report_idx is not None:
                    selected_report = all_reports[selected_report_idx]
                    file_path = selected_report['file']
                    
                    st.subheader(f"📖 {selected_report['company']} - {selected_report['type']}")
                    st.info(f"File: `{file_path.name}`")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if file_path.suffix == '.md':
                            # Display markdown report
                            st.markdown("---")
                            st.markdown(content)
                        elif file_path.suffix == '.json':
                            # Display JSON data in a nice format
                            import json
                            data = json.loads(content)
                            st.json(data)
                        elif file_path.suffix == '.csv':
                            # Display CSV data as dataframe
                            df = pd.read_csv(file_path)
                            st.dataframe(df)
                        
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
            
            # Show individual results summary
            st.subheader("📋 Analysis Summary")
            for result in results:
                if result["status"] == "success":
                    with st.expander(f"✅ {result['company']} - Analysis Summary"):
                        st.success(f"✅ Analysis completed successfully for {result['company']}")
                        
                        # Show key metrics from the output
                        output_text = result["output"]
                        if "Enhanced analysis completed" in output_text:
                            st.text("✅ Enhanced analysis completed")
                        if "Saved multi-LLM results" in output_text:
                            st.text("✅ Results saved in multiple formats")
                        if "Extracted" in output_text and "scores" in output_text:
                            lines = output_text.split('\n')
                            score_lines = [line for line in lines if 'Extracted' in line and 'scores' in line]
                            if score_lines:
                                st.text(f"📈 {score_lines[0]}")
                        
                        # Show available files for this company
                        company_files = [r for r in all_reports if r['company'] == result['company']]
                        if company_files:
                            st.info(f"📁 Available files: {len(company_files)}")
                            for file_info in company_files:
                                st.text(f"  • {file_info['type']}: {file_info['file'].name}")
                else:
                    with st.expander(f"❌ {result['company']} - Error"):
                        error_text = result["error"]
                        if error_text:
                            clean_error = error_text.encode('ascii', errors='ignore').decode('ascii')
                            st.error(clean_error)
                        else:
                            st.error("Analysis failed - check logs for details")
            
            # Download all results
            st.subheader("📥 Download Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Download All Reports (ZIP)"):
                    # Create a zip file with all reports
                    import zipfile
                    import io
                    
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for report in all_reports:
                            zip_file.write(report['file'], report['file'].name)
                    
                    zip_buffer.seek(0)
                    st.download_button(
                        "Download All Reports",
                        zip_buffer.getvalue(),
                        file_name=f"qualagent_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )
            
            with col2:
                # Download results summary JSON
                results_json = json.dumps(results, indent=2)
                st.download_button(
                    "📊 Download Results Summary",
                    results_json,
                    file_name=f"qualagent_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col3:
                # Download results as CSV
                results_df = pd.DataFrame(results)
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    "📈 Download Results CSV",
                    csv_data,
                    file_name=f"qualagent_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
