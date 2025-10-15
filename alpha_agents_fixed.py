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
from datetime import datetime
import datetime as dt

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agents', 'QualAgent'))

# Import original modules
from utils.stock_screener import StockScreener
from utils.fast_stock_screener import FastStockScreener
from utils.db_util import save_fundamental_screen

def load_fundamental_screen_postgresql(run_id=None, limit=None, industry_filter=None, sector_filter=None, min_score=None):
    """Load screening results from PostgreSQL database with enhanced filtering"""
    try:
        from utils.db_util import get_engine
        from sqlalchemy import text
        import pandas as pd
        
        engine = get_engine()
        
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
        from utils.db_util import get_engine
        from sqlalchemy import text
        
        engine = get_engine()
        
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

def main():
    st.set_page_config(
        page_title="Alpha Agents",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Clean navigation
    st.sidebar.title("🚀 Alpha Agents")
    st.sidebar.markdown("**AI-Powered Stock Analysis Platform**")
    st.sidebar.markdown("---")
    
    # Simple page selection
    page = st.sidebar.radio(
        "Navigate",
        ["📊 Screener", "🤖 QualAgent"],
        help="Choose between Fundamental Screening and AI Analysis"
    )
    
    # Display selected page
    if page == "📊 Screener":
        screener_page()
    elif page == "🤖 QualAgent":
        qualagent_page()

def screener_page():
    """Fundamental Stock Screener"""
    
    st.title("📊 Stock Screener")
    st.markdown("**Quantitative Analysis** - Filter stocks based on financial metrics")
    
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
    use_fast_screener = st.sidebar.checkbox("⚡ Fast Mode", value=True, 
                                           help="Parallel processing for faster results")
    
    filter_by = st.sidebar.radio("Filter By", ("Sector", "Industry"))

    if filter_by == "Sector":
        selection = st.sidebar.selectbox("Sector", SECTORS)
    else:
        selection = st.sidebar.selectbox("Industry", INDUSTRIES)

    region = st.sidebar.selectbox("Region", REGIONS, index=REGIONS.index(DEFAULT_REGION))
    min_cap = st.sidebar.number_input("Min Market Cap (USD)", min_value=0.0, value=0.0, step=1e9)
    max_cap_value = st.sidebar.number_input("Max Market Cap (USD, 0=unlimited)", min_value=0.0, value=0.0, step=1e9)
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
            
            # Store results for QualAgent with proper ticker handling
            df_for_qualagent = df.copy()
            
            # Ensure ticker column exists
            if "ticker" not in df_for_qualagent.columns:
                if df_for_qualagent.index.name == 'ticker':
                    df_for_qualagent = df_for_qualagent.reset_index()
                elif "index" in df_for_qualagent.columns:
                    df_for_qualagent = df_for_qualagent.rename(columns={"index": "ticker"})
                else:
                    # Create ticker column from index
                    df_for_qualagent = df_for_qualagent.reset_index()
                    if df_for_qualagent.index.name:
                        df_for_qualagent['ticker'] = df_for_qualagent.index
                    else:
                        df_for_qualagent['ticker'] = df_for_qualagent.iloc[:, 0]  # Use first column as ticker
            
            st.session_state["screener_results"] = df_for_qualagent
            st.success(f"✅ {len(df_for_qualagent)} companies ready for QualAgent analysis!")
            
            # Show companies that will be analyzed
            if 'ticker' in df_for_qualagent.columns:
                st.subheader("🎯 Companies Ready for Analysis")
                display_cols = ['ticker']
                if 'company_name' in df_for_qualagent.columns:
                    display_cols.append('company_name')
                elif 'name' in df_for_qualagent.columns:
                    display_cols.append('name')
                if 'sector' in df_for_qualagent.columns:
                    display_cols.append('sector')
                if 'score' in df_for_qualagent.columns:
                    display_cols.append('score')
                
                st.dataframe(df_for_qualagent[display_cols].head(10), width='content')
                st.info(f"💡 Go to 'QualAgent Analysis' page to analyze these {len(df_for_qualagent)} companies!")
            
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
            st.info("💡 **Next Step**: Switch to '🤖 QualAgent' page to analyze these companies with AI!")

def qualagent_page():
    """AI-Powered Qualitative Analysis"""
    
    st.title("🤖 QualAgent")
    st.markdown("**AI Analysis** - Multi-LLM qualitative analysis and ranking")
    
    # Database status check
    try:
        from utils.db_util import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        st.sidebar.success("✅ Database Connected")
    except Exception as e:
        st.sidebar.error("❌ Database Disconnected")
        st.sidebar.info("Check .env file")
    
    # Data source selection
    st.sidebar.header("📊 Data Source")
    data_source = st.sidebar.radio(
        "Load Companies",
        ["From Screener", "From Database", "Upload CSV", "Manual Input", "Sample"]
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
                st.dataframe(df[display_cols], width='content')
            else:
                st.dataframe(df.head(), width='content')
            
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
            from utils.db_util import get_engine
            from sqlalchemy import text
            
            engine = get_engine()
            
            # Test connection with better error handling
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test"))
                test_value = result.fetchone()[0]
            
            st.success("✅ Database connection successful!")
            
            # Test if fundamental_screen table exists and has data
            try:
                test_query = "SELECT COUNT(*) as count FROM fundamental_screen LIMIT 1"
                test_result = pd.read_sql_query(text(test_query), engine)
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
            
            # Query mode selection - simplified
            query_mode = st.radio(
                "Query Mode",
                ["🎯 Direct Company Selection", "📊 Pre-built Queries"],
                horizontal=True
            )
            
        except Exception as e:
            st.error(f"❌ Database connection failed: {str(e)}")
            st.info("💡 Please check your DATABASE_URL in .env file")
            st.stop()
        
        if query_mode == "🎯 Direct Company Selection":
            st.subheader("🎯 Direct Company Selection from Database")
            
            # Initialize row_limit
            row_limit = 200
            
            # Load all companies from database - use simple query for Direct Selection
            with st.spinner("Loading companies from database..."):
                try:
                    # Simple query without parameters for Direct Company Selection
                    simple_query = f"SELECT * FROM fundamental_screen ORDER BY created_at DESC, score DESC LIMIT {row_limit}"
                    all_companies = pd.read_sql_query(text(simple_query), engine)
                except Exception as e:
                    st.error(f"Direct query failed: {e}")
                    # Fallback to the function
                    all_companies = load_fundamental_screen_postgresql(limit=row_limit)
            
            if not all_companies.empty:
                st.success(f"✅ Loaded {len(all_companies)} companies from database")
                
                # Show companies immediately
                st.subheader("📊 Available Companies")
                
                # Display columns with better formatting
                display_cols = []
                if 'ticker' in all_companies.columns:
                    display_cols.append('ticker')
                if 'company_name' in all_companies.columns:
                    display_cols.append('company_name')
                if 'sector' in all_companies.columns:
                    display_cols.append('sector')
                if 'industry' in all_companies.columns:
                    display_cols.append('industry')
                if 'score' in all_companies.columns:
                    display_cols.append('score')
                if 'market_cap' in all_companies.columns:
                    display_cols.append('market_cap')
                
                if display_cols:
                    # Format the dataframe for better display
                    display_df = all_companies[display_cols].copy()
                    if 'score' in display_df.columns:
                        display_df['score'] = display_df['score'].round(2)
                    if 'market_cap' in display_df.columns:
                        display_df['market_cap'] = display_df['market_cap'].apply(lambda x: f"${x/1e9:.1f}B" if pd.notna(x) and x > 0 else "N/A")
                    
                    st.dataframe(display_df, width='stretch')
                else:
                    st.dataframe(all_companies, width='content')
                
                # Direct company selection
                if 'ticker' in all_companies.columns:
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
                    
                    # Handle selection buttons
                    if select_all:
                        st.session_state['direct_selected_tickers'] = all_companies['ticker'].tolist()
                    elif select_top5:
                        st.session_state['direct_selected_tickers'] = all_companies['ticker'].tolist()[:5]
                    elif select_top10:
                        st.session_state['direct_selected_tickers'] = all_companies['ticker'].tolist()[:10]
                    elif select_none:
                        st.session_state['direct_selected_tickers'] = []
                    
                    # Multiselect with better default
                    default_selection = st.session_state.get('direct_selected_tickers', all_companies['ticker'].tolist()[:3])
                    selected_tickers = st.multiselect(
                        "Select companies to analyze",
                        all_companies['ticker'].tolist(),
                        default=default_selection,
                        key="direct_company_selection",
                        help="Choose companies for QualAgent analysis"
                    )
                    companies_to_analyze = selected_tickers
                    
                    # Show selection summary with company details
                    if companies_to_analyze:
                        st.success(f"✅ Selected {len(companies_to_analyze)} companies for analysis")
                        
                        # Show selected companies with their details
                        selected_df = all_companies[all_companies['ticker'].isin(companies_to_analyze)]
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
                    else:
                        st.warning("⚠️ No companies selected")
                else:
                    st.error("❌ No 'ticker' column found in database")
            else:
                st.warning("⚠️ No companies found in database")
                st.info("💡 Run the Fundamental Screener first to populate the database")
        
        elif query_mode == "📊 Pre-built Queries":
            st.subheader("📊 Smart Database Query Interface")
            
            # Load filter options from database
            with st.spinner("Loading filter options..."):
                filter_options = get_database_filters()
            
            # Create filter interface - simplified for non-technical users
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Filter Options")
                
                # Row limit - simplified
                row_limit = st.number_input("Max rows to load", min_value=10, max_value=1000, value=100, step=10, help="Enter 0 for no limit")
                if row_limit == 0:
                    row_limit = None
                
                # Industry dropdown
                if filter_options['industries']:
                    industry_filter = st.selectbox(
                        "Filter by Industry",
                        ["All Industries"] + filter_options['industries'],
                        help="Select a specific industry to filter companies"
                    )
                    industry_filter = None if industry_filter == "All Industries" else industry_filter
                else:
                    industry_filter = st.text_input("Filter by industry (manual)", placeholder="e.g., Technology")
                
                # Sector dropdown
                if filter_options['sectors']:
                    sector_filter = st.selectbox(
                        "Filter by Sector",
                        ["All Sectors"] + filter_options['sectors'],
                        help="Select a specific sector to filter companies"
                    )
                    sector_filter = None if sector_filter == "All Sectors" else sector_filter
                else:
                    sector_filter = st.text_input("Filter by sector (manual)", placeholder="e.g., Technology")
            
            with col2:
                st.subheader("📊 Score & Run Filters")
                
                # Score filter
                min_score = st.number_input(
                    "Minimum Score", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=0.0, 
                    step=1.0,
                    help="Filter companies with score above this value"
                )
                
                # Run ID filter
                if filter_options['runs']:
                    selected_run = st.selectbox(
                        "Filter by Run ID",
                        ["All Runs"] + filter_options['runs'],
                        help="Select a specific screening run"
                    )
                    run_filter = None if selected_run == "All Runs" else selected_run
                else:
                    run_filter = None
                
                # Additional filters
                st.subheader("🔍 Additional Options")
                show_only_top_scorers = st.checkbox("Show only top 20% scorers", help="Filter to companies in top 20% by score")
            
            # Execute query button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Execute Smart Query", type="primary", width='stretch'):
                    try:
                        # Load data with filters
                        filtered_data = load_fundamental_screen_postgresql(
                            run_id=run_filter,
                            limit=row_limit,
                            industry_filter=industry_filter,
                            sector_filter=sector_filter,
                            min_score=min_score if min_score > 0 else None
                        )
                        
                        if not filtered_data.empty:
                            # Apply top scorers filter if selected
                            if show_only_top_scorers and 'score' in filtered_data.columns:
                                score_threshold = filtered_data['score'].quantile(0.8)
                                filtered_data = filtered_data[filtered_data['score'] >= score_threshold]
                                st.info(f"📈 Showing top 20% scorers (score >= {score_threshold:.1f})")
                            
                            st.success(f"✅ Query executed successfully! Found {len(filtered_data)} companies")
                            
                            # Show results with enhanced display
                            st.subheader("📊 Query Results")
                            
                            # Display columns with better formatting
                            display_cols = []
                            if 'ticker' in filtered_data.columns:
                                display_cols.append('ticker')
                            if 'company_name' in filtered_data.columns:
                                display_cols.append('company_name')
                            if 'sector' in filtered_data.columns:
                                display_cols.append('sector')
                            if 'industry' in filtered_data.columns:
                                display_cols.append('industry')
                            if 'score' in filtered_data.columns:
                                display_cols.append('score')
                            if 'market_cap' in filtered_data.columns:
                                display_cols.append('market_cap')
                            
                            if display_cols:
                                # Format the dataframe for better display
                                display_df = filtered_data[display_cols].copy()
                                if 'score' in display_df.columns:
                                    display_df['score'] = display_df['score'].round(2)
                                if 'market_cap' in display_df.columns:
                                    display_df['market_cap'] = display_df['market_cap'].apply(lambda x: f"${x/1e9:.1f}B" if pd.notna(x) and x > 0 else "N/A")
                                
                                st.dataframe(display_df, width='stretch')
                            else:
                                st.dataframe(filtered_data, width='content')
                            
                            # Store results for company selection
                            st.session_state["db_query_results"] = filtered_data
                            
                            # Company selection interface
                            if 'ticker' in filtered_data.columns:
                                st.subheader("🎯 Select Companies for QualAgent Analysis")
                                
                                # Company search for database results
                                db_search_term = st.text_input("🔍 Search companies (type to filter)", placeholder="e.g., type 'S' to find companies starting with S", key="db_company_search")
                                
                                # Filter companies based on search
                                if db_search_term:
                                    db_filtered_companies = [c for c in filtered_data['ticker'].tolist() if c.lower().startswith(db_search_term.lower())]
                                    st.info(f"Found {len(db_filtered_companies)} companies starting with '{db_search_term}'")
                                else:
                                    db_filtered_companies = filtered_data['ticker'].tolist()
                                
                                # Quick selection buttons
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    select_all = st.button("✅ Select All", key="db_select_all")
                                with col2:
                                    select_top = st.button("🏆 Select Top 5", key="db_select_top")
                                with col3:
                                    select_top10 = st.button("🥇 Select Top 10", key="db_select_top10")
                                with col4:
                                    select_none = st.button("❌ Select None", key="db_select_none")
                                
                                # Handle selection buttons
                                if select_all:
                                    st.session_state['db_selected_tickers'] = db_filtered_companies
                                elif select_top:
                                    st.session_state['db_selected_tickers'] = db_filtered_companies[:5]
                                elif select_top10:
                                    st.session_state['db_selected_tickers'] = db_filtered_companies[:10]
                                elif select_none:
                                    st.session_state['db_selected_tickers'] = []
                                
                                # Multiselect with search
                                default_selection = st.session_state.get('db_selected_tickers', db_filtered_companies[:3])
                                selected_tickers = st.multiselect(
                                    "Select companies to analyze",
                                    db_filtered_companies,
                                    default=default_selection,
                                    key="db_company_selection",
                                    help="Choose companies for QualAgent analysis"
                                )
                                companies_to_analyze = selected_tickers
                                
                                # Show selection summary with company details
                                if companies_to_analyze:
                                    st.success(f"✅ Selected {len(companies_to_analyze)} companies for analysis")
                                    
                                    # Show selected companies with their details
                                    selected_df = filtered_data[filtered_data['ticker'].isin(companies_to_analyze)]
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
                                else:
                                    st.warning("⚠️ No companies selected")
                            else:
                                st.error("❌ No 'ticker' column found in query results")
                        else:
                            st.warning("⚠️ No companies found matching your criteria")
                            st.info("💡 Try adjusting your filters or check if data exists in the database")
                            
                    except Exception as e:
                        st.error(f"❌ Query failed: {str(e)}")
                        st.info("💡 Try adjusting your filters or check the database connection")
        
    
    elif data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.subheader("📋 Uploaded Data")
            st.dataframe(df.head(), width='content')
            
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
    
    elif data_source == "Sample":
        st.subheader("📋 Sample Companies")
        sample_companies = ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN"]
        selected_tickers = st.multiselect(
            "Select sample companies to analyze",
            sample_companies,
            default=sample_companies[:3]
        )
        companies_to_analyze = selected_tickers
    
    # Analysis configuration
    st.sidebar.header("🔧 Analysis")
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["quick", "comprehensive", "expert-guided"],
        help="Quick: ~30s, Comprehensive: ~3min, Expert-guided: ~10min"
    )
    
    # Show analysis type recommendations
    if analysis_type == "expert-guided":
        st.sidebar.warning("⚠️ Expert-guided analysis can take 10+ minutes per company and may timeout")
        st.sidebar.info("💡 Recommended: Use 'quick' analysis for faster results")
    elif analysis_type == "comprehensive":
        st.sidebar.info("💡 Comprehensive analysis takes ~3 minutes per company")
    else:
        st.sidebar.success("✅ Quick analysis recommended for best performance")
    
    user_id = st.sidebar.text_input("User ID", value="streamlit_user")
    
    # Debug: Show companies to analyze
    st.sidebar.write("---")
    st.sidebar.subheader("🔍 Debug Info")
    st.sidebar.write(f"Companies to analyze: {len(companies_to_analyze)}")
    if companies_to_analyze:
        st.sidebar.write(f"Selected: {companies_to_analyze[:3]}{'...' if len(companies_to_analyze) > 3 else ''}")
    
    # Run analysis
    if companies_to_analyze:
        st.subheader("🚀 Run Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Run QualAgent Analysis", type="primary"):
                st.write(f"Starting analysis for: {companies_to_analyze}")
                run_qualagent_analysis_fixed(companies_to_analyze, analysis_type, user_id)
        
        with col2:
            if st.button("💰 Estimate Cost"):
                cost_per_analysis = {"quick": 0.003, "comprehensive": 0.018, "expert-guided": 0.030}
                total_cost = cost_per_analysis.get(analysis_type, 0.018) * len(companies_to_analyze)
                st.info(f"💰 Estimated cost: ${total_cost:.3f}")
    else:
        st.warning("⚠️ No companies selected for analysis")
        st.info("💡 Please select companies from the data source above")
    
    # Display results
    display_analysis_results()

def run_qualagent_analysis_fixed(companies, analysis_type, user_id):
    """Run QualAgent analysis - FIXED VERSION"""
    
    st.subheader("🔄 Running Analysis...")
    
    # Show analysis type and estimated time
    analysis_times = {
        "quick": "~30 seconds per company",
        "comprehensive": "~3 minutes per company", 
        "expert-guided": "~10 minutes per company"
    }
    
    st.info(f"📊 Analysis Type: **{analysis_type}** - Estimated time: {analysis_times.get(analysis_type, '~3 minutes')} per company")
    st.info(f"⏱️ Total estimated time: {len(companies)} companies × {analysis_times.get(analysis_type, '~3 minutes')}")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    for i, company in enumerate(companies):
        status_text.text(f"🔄 Analyzing {company}... (Company {i+1}/{len(companies)})")
        progress_bar.progress((i + 1) / len(companies))
        
        try:
            # Run QualAgent analysis using FIXED approach
            result = run_single_analysis_fixed(company, analysis_type, user_id)
            results.append(result)
            
            if result["status"] == "success":
                st.success(f"✅ {company} analysis completed successfully")
            elif result["status"] == "timeout":
                st.warning(f"⏰ {company} analysis timed out - try 'quick' analysis type")
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
    
    # Show final summary
    successful = len([r for r in results if r["status"] == "success"])
    failed = len([r for r in results if r["status"] != "success"])
    
    status_text.text("Analysis completed!")
    progress_bar.progress(1.0)
    
    st.success(f"📊 Analysis Summary: {successful} successful, {failed} failed")
    
    if failed > 0:
        st.info("💡 For faster results, try using 'quick' analysis type or analyze fewer companies at once")

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
            "--models", "llama-3-70b", "mixtral-8x7b", "qwen2-72b", "deepseek-coder-33b"
        ]
        
        # Set environment to handle encoding properly
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            cmd,
            cwd=qualagent_dir,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout (increased from 5 minutes)
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
            "error": f"Analysis timed out after 10 minutes for {company}. Try using 'quick' analysis type for faster results.",
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
    """Display analysis results with enhanced LLM model rankings and individual company reports"""
    
    # Initialize variables
    companies_with_reports = []
    all_reports = []
    
    if "analysis_completed" in st.session_state and st.session_state["analysis_completed"]:
        st.subheader("📊 QualAgent Analysis Results")
        
        results = st.session_state.get("qualagent_results", [])
        
        if results:
            # Show results summary
            results_df = pd.DataFrame(results)
            st.dataframe(results_df[['company', 'status', 'timestamp']], width='content')
            
            # Get all available reports and analyze LLM performance
            qualagent_dir = Path(__file__).parent / "agents" / "QualAgent"
            
            all_reports = []
            llm_performance = {}
            company_scores = {}  # Store scores for ranking
            
            st.write(f"🔍 DEBUG: Processing {len(results)} results")
            
            for result in results:
                if result["status"] == "success":
                    company = result['company']
                    st.write(f"🔍 DEBUG: Processing company: {company}")
                    
                    # Find all report files for this company - use correct patterns
                    report_files = list(qualagent_dir.glob(f"analysis_report_{company}_*.md"))
                    # JSON files use timestamp format, so we need to search all recent JSON files
                    json_files = list(qualagent_dir.glob("analysis_results_*.json"))
                    csv_files = list(qualagent_dir.glob("analysis_results_*.csv"))
                    
                    # Debug: Show what files were found
                    st.write(f"🔍 Debug: For {company} found {len(report_files)} MD, {len(json_files)} JSON, {len(csv_files)} CSV files")
                    
                    for report_file in report_files:
                        all_reports.append({
                            'company': company,
                            'file': report_file,
                            'type': 'Markdown Report',
                            'timestamp': report_file.stat().st_mtime
                        })
                    
                    st.write(f"🔍 DEBUG: Processing {len(json_files)} JSON files for {company}")
                    
                    for json_file in json_files:
                        st.write(f"🔍 DEBUG: Checking JSON file: {json_file.name}")
                        # Check if this JSON file contains data for our company
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            st.write(f"🔍 DEBUG: Loaded JSON data from {json_file.name}")
                            st.write(f"🔍 DEBUG: JSON keys: {list(data.keys())}")
                            
                            # Check if this file contains our company
                            company_found = False
                            if 'analyses' in data:
                                st.write(f"🔍 DEBUG: Found {len(data['analyses'])} analyses in {json_file.name}")
                                for analysis in data['analyses']:
                                    ticker = analysis.get('company', {}).get('ticker')
                                    st.write(f"🔍 DEBUG: Checking ticker: {ticker} against {company}")
                                    if ticker == company:
                                        company_found = True
                                        st.write(f"🔍 Debug: Found {company} in {json_file.name}")
                                        break
                            else:
                                st.write(f"🔍 DEBUG: No 'analyses' key in {json_file.name}")
                            
                            if company_found:
                                all_reports.append({
                                    'company': company,
                                    'file': json_file,
                                    'type': 'JSON Data',
                                    'timestamp': json_file.stat().st_mtime
                                })
                                
                                # Analyze LLM performance from JSON files
                                st.write(f"🔍 DEBUG: Checking detailed_results in {json_file.name}")
                                
                                # Initialize variables at the beginning
                                company_scores[company] = {}
                                total_score = 0
                                score_count = 0
                                
                                # Check if this JSON file contains analysis for the current company
                                if 'analyses' in data and len(data['analyses']) > 0:
                                    analysis = data['analyses'][0]  # Get the first analysis
                                    if 'detailed_results' in analysis and 'llm_analyses' in analysis['detailed_results']:
                                        st.write(f"🔍 Debug: Processing LLM analyses for {company}")
                                        st.write(f"🔍 DEBUG: Found {len(analysis['detailed_results']['llm_analyses'])} LLM analyses")
                                    
                                        for llm_analysis in analysis['detailed_results']['llm_analyses']:
                                            model_name = llm_analysis.get('llm_model', 'unknown')
                                            st.write(f"🔍 DEBUG: Processing LLM model: {model_name}")
                                        
                                            if model_name not in llm_performance:
                                                llm_performance[model_name] = {
                                                    'total_analyses': 0,
                                                    'successful_analyses': 0,
                                                    'avg_score': 0,
                                                    'companies': [],
                                                    'individual_scores': {}
                                                }
                                            
                                            llm_performance[model_name]['total_analyses'] += 1
                                            
                                            # Extract scores from parsed_output or raw_output
                                            scores = []
                                            st.write(f"🔍 DEBUG: Extracting scores for {model_name}")
                                            
                                            # Try parsed_output first
                                            if 'parsed_output' in llm_analysis and 'dimensions' in llm_analysis['parsed_output']:
                                                dimensions = llm_analysis['parsed_output']['dimensions']
                                                st.write(f"🔍 DEBUG: Found dimensions in parsed_output: {type(dimensions)}")
                                                if isinstance(dimensions, dict):
                                                    scores = [dim.get('score', 0) for dim in dimensions.values() if isinstance(dim, dict) and 'score' in dim]
                                                    st.write(f"🔍 DEBUG: Extracted {len(scores)} scores from parsed_output dimensions")
                                            
                                            # If no scores in parsed_output, try to extract from raw_output
                                            if not scores and 'raw_output' in llm_analysis:
                                                st.write(f"🔍 DEBUG: No scores in parsed_output, trying raw_output")
                                                raw_output = llm_analysis['raw_output']
                                                # Look for dimensions with scores in the raw output
                                                import re
                                                score_pattern = r'"score":\s*(\d+)'
                                                score_matches = re.findall(score_pattern, raw_output)
                                                if score_matches:
                                                    scores = [int(score) for score in score_matches]
                                                    st.write(f"🔍 DEBUG: Extracted {len(scores)} scores from raw_output using regex")
                                                else:
                                                    st.write(f"🔍 DEBUG: No scores found in raw_output using regex")
                                            else:
                                                st.write(f"🔍 DEBUG: No raw_output available")
                                            
                                            if scores:
                                                st.write(f"🔍 Debug: Found {len(scores)} scores for {company} - {model_name}: {scores}")
                                                llm_performance[model_name]['successful_analyses'] += 1
                                                llm_performance[model_name]['companies'].append(company)
                                                
                                                llm_avg_score = sum(scores) / len(scores) if scores else 0
                                                
                                                llm_performance[model_name]['individual_scores'][company] = llm_avg_score
                                                company_scores[company][model_name] = llm_avg_score
                                                total_score += llm_avg_score
                                                score_count += 1
                                            else:
                                                st.write(f"🔍 Debug: No scores found for {company} - {model_name}")
                                    
                                    # Calculate overall company score
                                    if score_count > 0:
                                        company_scores[company]['overall_avg'] = total_score / score_count
                                    else:
                                        st.write(f"🔍 Debug: No scores calculated for {company}")
                                else:
                                    st.write(f"🔍 DEBUG: No detailed_results.llm_analyses found in {json_file.name}")
                                    # Show what keys are available
                                    if 'analyses' in data and len(data['analyses']) > 0:
                                        analysis = data['analyses'][0]
                                        if 'detailed_results' in analysis:
                                            st.write(f"🔍 DEBUG: detailed_results keys: {list(analysis['detailed_results'].keys())}")
                                        else:
                                            st.write(f"🔍 DEBUG: No detailed_results in analysis")
                                            st.write(f"🔍 DEBUG: Analysis keys: {list(analysis.keys())}")
                                    else:
                                        st.write(f"🔍 DEBUG: No analyses found in {json_file.name}")
                                        st.write(f"🔍 DEBUG: Available keys: {list(data.keys())}")
                                    
                        except Exception as e:
                            st.write(f"⚠️ Error reading {json_file}: {e}")
                            continue
                    
                    for csv_file in csv_files:
                        all_reports.append({
                            'company': company,
                            'file': csv_file,
                            'type': 'CSV Data',
                            'timestamp': csv_file.stat().st_mtime
                        })
            
            # Debug: Show what we collected
            st.write(f"🔍 Debug: llm_performance keys: {list(llm_performance.keys())}")
            st.write(f"🔍 Debug: company_scores keys: {list(company_scores.keys())}")
            st.write(f"🔍 Debug: all_reports count: {len(all_reports)}")
            
            # Display Comprehensive LLM Ranking Table
            if llm_performance and company_scores:
                st.subheader("🏆 LLM Ranking & Company Scores")
                
                # Create comprehensive ranking table
                companies_list = list(company_scores.keys())
                models_list = list(llm_performance.keys())
                
                # Create ranking dataframe
                ranking_data = []
                for company in companies_list:
                    row = {'Company': company}
                    
                    # Add individual LLM scores
                    for model in models_list:
                        score = company_scores[company].get(model, 0)
                        row[f'{model}'] = round(score, 2) if score > 0 else 'N/A'
                    
                    # Add overall average
                    overall_avg = company_scores[company].get('overall_avg', 0)
                    row['Overall Score'] = round(overall_avg, 2) if overall_avg > 0 else 'N/A'
                    
                    ranking_data.append(row)
                
                # Sort by overall average
                ranking_df = pd.DataFrame(ranking_data)
                ranking_df = ranking_df.sort_values('Overall Score', ascending=False, na_position='last')
                
                # Display ranking table with better formatting
                st.dataframe(ranking_df, width='stretch')
                
                # Show table info
                st.success(f"📊 {len(ranking_df)} companies ranked by overall score")
                
                # Company search and selection
                st.subheader("🔍 Select Companies for Detailed Analysis")
                
                # Company search
                search_term = st.text_input("🔍 Search companies (type to filter)", placeholder="e.g., type 'S' to find companies starting with S")
                
                # Filter companies based on search
                if search_term:
                    filtered_companies = [c for c in companies_list if c.lower().startswith(search_term.lower())]
                    st.info(f"Found {len(filtered_companies)} companies starting with '{search_term}'")
                else:
                    filtered_companies = companies_list
                
                # Company selection with search
                selected_companies = st.multiselect(
                    "Select companies to view reports",
                    filtered_companies,
                    default=filtered_companies[:3] if len(filtered_companies) >= 3 else filtered_companies,
                    key="company_selection_for_reports"
                )
                
                if selected_companies:
                    st.success(f"✅ Selected {len(selected_companies)} companies: {', '.join(selected_companies)}")
                    
                    # Show selected companies' scores
                    selected_df = ranking_df[ranking_df['Company'].isin(selected_companies)]
                    st.subheader("📊 Selected Companies Scores")
                    st.dataframe(selected_df, width='stretch')
                    
                    # Report selection by company and LLM
                    st.subheader("📄 Select Reports to View")
                    
                    # Get reports for selected companies
                    selected_reports = [r for r in all_reports if r['company'] in selected_companies]
                    
                    if selected_reports:
                        # Company selection for reports
                        report_company = st.selectbox(
                            "Select Company",
                            selected_companies,
                            key="report_company_selection",
                            help="Choose which company's reports to view"
                        )
                        
                        # Get reports for selected company
                        company_reports = [r for r in selected_reports if r['company'] == report_company]
                        
                        if company_reports:
                            st.info(f"📊 {len(company_reports)} reports available for {report_company}")
                            
                            # Report type selection
                            report_types = list(set([r['type'] for r in company_reports]))
                            
                            for report_type in report_types:
                                type_reports = [r for r in company_reports if r['type'] == report_type]
                                
                                with st.expander(f"📋 {report_type} ({len(type_reports)} files)"):
                                    for report in type_reports:
                                        col1, col2, col3 = st.columns([3, 1, 1])
                                        
                                        with col1:
                                            st.write(f"**{report['file'].name}**")
                                            st.caption(f"Modified: {datetime.fromtimestamp(report['timestamp']).strftime('%Y-%m-%d %H:%M')}")
                                        
                                        with col2:
                                            # View button
                                            if st.button(f"👁️ View", key=f"view_report_{report['file'].name}"):
                                                st.session_state[f"viewing_report_{report['file'].name}"] = True
                                        
                                        with col3:
                                            # Download button
                                            try:
                                                with open(report['file'], 'rb') as f:
                                                    file_data = f.read()
                                                
                                                st.download_button(
                                                    "📥 Download",
                                                    file_data,
                                                    file_name=report['file'].name,
                                                    mime="text/plain" if report['file'].suffix == '.md' else "application/json" if report['file'].suffix == '.json' else "text/csv",
                                                    key=f"download_report_{report['file'].name}"
                                                )
                                            except Exception as e:
                                                st.error(f"Download failed: {str(e)}")
                                        
                                        # Display content if viewing
                                        if st.session_state.get(f"viewing_report_{report['file'].name}", False):
                                            st.markdown("---")
                                            st.subheader(f"📖 Viewing: {report['file'].name}")
                                            
                                            try:
                                                with open(report['file'], 'r', encoding='utf-8') as f:
                                                    content = f.read()
                                                
                                                if report['file'].suffix == '.md':
                                                    st.markdown("### Report Content:")
                                                    st.markdown(content)
                                                elif report['file'].suffix == '.json':
                                                    st.markdown("### JSON Data:")
                                                    data = json.loads(content)
                                                    st.json(data)
                                                elif report['file'].suffix == '.csv':
                                                    st.markdown("### CSV Data:")
                                                    df = pd.read_csv(report['file'])
                                                    st.dataframe(df, width='stretch')
                                                else:
                                                    st.markdown("### File Content:")
                                                    st.text(content)
                                                
                                                # Download button for viewed report
                                                st.markdown("---")
                                                col1, col2 = st.columns([1, 1])
                                                
                                                with col1:
                                                    try:
                                                        with open(report['file'], 'rb') as f:
                                                            file_data = f.read()
                                                        
                                                        st.download_button(
                                                            f"📥 Download {report['file'].name}",
                                                            file_data,
                                                            file_name=report['file'].name,
                                                            mime="text/plain" if report['file'].suffix == '.md' else "application/json" if report['file'].suffix == '.json' else "text/csv",
                                                            key=f"download_viewed_report_{report['file'].name}"
                                                        )
                                                    except Exception as download_error:
                                                        st.error(f"Download failed: {str(download_error)}")
                                                
                                                with col2:
                                                    if st.button(f"❌ Close Viewer", key=f"close_report_{report['file'].name}"):
                                                        st.session_state[f"viewing_report_{report['file'].name}"] = False
                                                        st.rerun()
                                                
                                            except Exception as e:
                                                st.error(f"Error reading file: {str(e)}")
                                                st.info("💡 Try downloading the file instead")
                                        
                                        st.markdown("---")
                        else:
                            st.warning(f"No reports found for {report_company}")
                    else:
                        st.warning("No reports found for selected companies")
                
                # Top 3 Companies Selection
                st.subheader("🥇 Top 3 Companies Selection")
                top_3_companies = ranking_df.head(3)['Company'].tolist()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🏆 Select Top 3 Companies", type="primary"):
                        st.session_state["top_3_tickers"] = top_3_companies
                        st.success(f"✅ Selected top 3 companies: {', '.join(top_3_companies)}")
                
                with col2:
                    if st.button("📊 Show Top 3 Details"):
                        st.session_state["show_top_3_details"] = True
                
                with col3:
                    if st.button("📄 View Top 3 Reports"):
                        st.session_state["view_top_3_reports"] = True
                
                # Show top 3 details if requested
                if st.session_state.get("show_top_3_details", False):
                    st.subheader("📊 Top 3 Companies Details")
                    top_3_df = ranking_df.head(3)
                    st.dataframe(top_3_df, width='stretch')
                    
                    if st.button("❌ Close Details"):
                        st.session_state["show_top_3_details"] = False
                        st.rerun()
                
                # Show top 3 reports if requested
                if st.session_state.get("view_top_3_reports", False):
                    st.subheader("📄 Top 3 Companies Reports")
                    
                    # Get all reports for top 3 companies
                    top3_reports = [r for r in all_reports if r['company'] in top_3_companies]
                    
                    if top3_reports:
                        # Company selection for reports
                        selected_company = st.selectbox(
                            "Select Company to View Reports",
                            top_3_companies,
                            key="top3_company_selection",
                            help="Choose which company's reports to view"
                        )
                        
                        # Get reports for selected company
                        company_reports = [r for r in top3_reports if r['company'] == selected_company]
                        
                        if company_reports:
                            st.info(f"📊 {len(company_reports)} reports available for {selected_company}")
                            
                            # Show reports with timestamps
                            for report in company_reports:
                                # Format timestamp
                                timestamp = datetime.fromtimestamp(report['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                                
                                with st.expander(f"📋 {report['file'].name} - {timestamp}"):
                                    col1, col2 = st.columns([3, 1])
                                    
                                    with col1:
                                        st.write(f"**File:** {report['file'].name}")
                                        st.write(f"**Type:** {report['type']}")
                                        st.write(f"**Modified:** {timestamp}")
                                    
                                    with col2:
                                        # View button
                                        if st.button(f"👁️ View", key=f"view_top3_{report['file'].name}"):
                                            st.session_state[f"viewing_top3_{report['file'].name}"] = True
                                    
                                    # Display content if viewing
                                    if st.session_state.get(f"viewing_top3_{report['file'].name}", False):
                                        st.markdown("---")
                                        st.subheader(f"📖 Viewing: {report['file'].name}")
                                        
                                        try:
                                            with open(report['file'], 'r', encoding='utf-8') as f:
                                                content = f.read()
                                            
                                            if report['file'].suffix == '.md':
                                                st.markdown("### Report Content:")
                                                st.markdown(content)
                                            elif report['file'].suffix == '.json':
                                                st.markdown("### JSON Data:")
                                                data = json.loads(content)
                                                st.json(data)
                                            elif report['file'].suffix == '.csv':
                                                st.markdown("### CSV Data:")
                                                df = pd.read_csv(report['file'])
                                                st.dataframe(df, width='stretch')
                                            else:
                                                st.markdown("### File Content:")
                                                st.text(content)
                                            
                                            # Download button for viewed report
                                            st.markdown("---")
                                            col1, col2 = st.columns([1, 1])
                                            
                                            with col1:
                                                try:
                                                    with open(report['file'], 'rb') as f:
                                                        file_data = f.read()
                                                    
                                                    st.download_button(
                                                        f"📥 Download {report['file'].name}",
                                                        file_data,
                                                        file_name=report['file'].name,
                                                        mime="text/plain" if report['file'].suffix == '.md' else "application/json" if report['file'].suffix == '.json' else "text/csv",
                                                        key=f"download_viewed_{report['file'].name}"
                                                    )
                                                except Exception as download_error:
                                                    st.error(f"Download failed: {str(download_error)}")
                                            
                                            with col2:
                                                if st.button(f"❌ Close Viewer", key=f"close_top3_{report['file'].name}"):
                                                    st.session_state[f"viewing_top3_{report['file'].name}"] = False
                                                    st.rerun()
                                            
                                        except Exception as e:
                                            st.error(f"Error reading file: {str(e)}")
                                            st.info("💡 Try downloading the file instead")
                        else:
                            st.warning(f"No reports found for {selected_company}")
                    else:
                        st.warning("No reports found for top 3 companies")
                    
                    if st.button("❌ Close Reports"):
                        st.session_state["view_top_3_reports"] = False
                        st.rerun()
                
                # LLM Performance Summary
                st.subheader("📊 LLM Performance Summary")
                performance_data = []
                for model, stats in llm_performance.items():
                    # Calculate average score from individual scores
                    individual_scores = list(stats['individual_scores'].values())
                    avg_score = sum(individual_scores) / len(individual_scores) if individual_scores else 0
                    
                    performance_data.append({
                        'LLM Model': model,
                        'Average Score': round(avg_score, 2),
                        'Companies Analyzed': len(stats['companies']),
                        'Success Rate': f"{(stats['successful_analyses'] / stats['total_analyses'] * 100):.1f}%" if stats['total_analyses'] > 0 else "0%"
                    })
                
                performance_df = pd.DataFrame(performance_data)
                st.dataframe(performance_df, width='stretch')
                
            else:
                st.warning("⚠️ Could not parse LLM performance data from JSON files")
                st.info("💡 This might be due to different JSON structure. Showing basic results...")
                
                # Show basic results even if LLM parsing failed
                if all_reports:
                    st.subheader("📄 Available Reports")
                    for report in all_reports:
                        st.write(f"📋 {report['company']} - {report['type']} - {report['file'].name}")
                
                # Show basic company list
                if results:
                    st.subheader("📊 Analysis Results Summary")
                    for result in results:
                        if result["status"] == "success":
                            st.success(f"✅ {result['company']} - Analysis completed")
                        else:
                            st.error(f"❌ {result['company']} - {result.get('error', 'Unknown error')}")
                
                st.warning("No analysis results available. Run QualAgent analysis first!")
                
        else:
            st.info("No analysis results available. Run QualAgent analysis first!")

if __name__ == "__main__":
    main()