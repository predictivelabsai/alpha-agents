import os
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
import uuid

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
DDL_PATH = REPO_ROOT / "sql" / "create_table.sql"


def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable must be set for PostgreSQL connection")
    return create_engine(db_url)


def ensure_table_exists() -> None:
    engine = get_engine()
    # Use PostgreSQL DDL
    sql = DDL_PATH.read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.execute(text(sql))


def save_fundamental_screen(df: pd.DataFrame, run_id: Optional[str] = None) -> str:
    engine = get_engine()
    ensure_table_exists()
    if run_id is None:
        run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    df_to_save = df.copy()
    # Normalize expected columns
    if "ticker" not in df_to_save.columns:
        for candidate in ("index", "Index", "Unnamed: 0", "symbol", "Symbol"):
            if candidate in df_to_save.columns:
                df_to_save = df_to_save.rename(columns={candidate: "ticker"})
                break
    # Ensure run_id column present
    if "run_id" not in df_to_save.columns:
        df_to_save.insert(0, "run_id", run_id)
    # Keep only known schema columns
    allowed_cols = [
        "run_id",
        "ticker",
        "company_name",
        "sector",
        "industry",
        "market_cap",
        "gross_profit_margin_ttm",
        "operating_profit_margin_ttm",
        "net_profit_margin_ttm",
        "ebit_margin_ttm",
        "ebitda_margin_ttm",
        "roa_ttm",
        "roe_ttm",
        "roic_ttm",
        "total_revenue_4y_cagr",
        "net_income_4y_cagr",
        "operating_cash_flow_4y_cagr",
        "total_revenue_4y_consistency",
        "net_income_4y_consistency",
        "operating_cash_flow_4y_consistency",
        "current_ratio",
        "debt_to_ebitda_ratio",
        "debt_servicing_ratio",
        "score",
    ]
    existing = [c for c in allowed_cols if c in df_to_save.columns]
    df_to_save = df_to_save[existing]
    # Convert numpy types to Python native types to avoid PostgreSQL schema errors
    import numpy as np
    
    for col in df_to_save.columns:
        if df_to_save[col].dtype.name.startswith('float'):
            # Convert numpy float to Python float
            df_to_save[col] = df_to_save[col].apply(lambda x: float(x) if pd.notnull(x) else None)
        elif df_to_save[col].dtype.name.startswith('int'):
            # Convert numpy int to Python int
            df_to_save[col] = df_to_save[col].apply(lambda x: int(x) if pd.notnull(x) else None)
        elif 'object' in str(df_to_save[col].dtype):
            # Handle object columns that might contain numpy types
            df_to_save[col] = df_to_save[col].apply(lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x)
    
    # Final cleanup - ensure no numpy types remain
    df_to_save = df_to_save.where(pd.notnull(df_to_save), None)
    
    # Insert
    df_to_save.to_sql(
        "fundamental_screen",
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=500,
    )
    return run_id


def save_qualitative_analysis(analysis_data, run_id=None, company_ticker=None):
    """Save qualitative analysis results to PostgreSQL database"""
    try:
        import json
        import time
        import numpy as np
        
        engine = get_engine()
        
        # Generate run_id if not provided
        if not run_id:
            run_id = f"qual_analysis_{int(time.time())}"
        
        # Extract data from the JSON structure
        metadata = analysis_data.get('metadata', {})
        company = analysis_data.get('company', {})
        composite_score = analysis_data.get('composite_score', {})
        consensus_scores = analysis_data.get('consensus_scores', {})
        individual_model_results = analysis_data.get('individual_model_results', {})
        execution_metadata = analysis_data.get('execution_metadata', {})
        
        # Debug: Print structure to understand the data
        print(f"DEBUG: analysis_data keys: {list(analysis_data.keys())}")
        print(f"DEBUG: company data: {company}")
        print(f"DEBUG: composite_score: {composite_score}")
        print(f"DEBUG: consensus_scores type: {type(consensus_scores)}")
        if isinstance(consensus_scores, dict):
            print(f"DEBUG: consensus_scores keys: {list(consensus_scores.keys())}")
        else:
            print(f"DEBUG: consensus_scores value: {consensus_scores}")
        
        # Use provided ticker or extract from data
        if company_ticker:
            # Use the explicitly provided ticker
            print(f"DEBUG: Using provided ticker: {company_ticker}")
            company = {'ticker': company_ticker, 'company_name': company_ticker}
        else:
            # Handle different data structures - check if this is a batch result
            if 'output' in analysis_data and 'status' in analysis_data:
                # This is a batch result, extract company info differently
                output_text = analysis_data.get('output', '')
                print(f"DEBUG: Batch output detected: {output_text[:200]}...")
                
                # Try to extract company ticker from output - look for the specific company
                import re
                
                # First try to find the specific company being analyzed
                # Look for patterns like "Analyzing TTD" or "CHECK TTD completed"
                ticker_patterns = [
                    r'Analyzing (\w+)',
                    r'CHECK (\w+) completed',
                    r'(\w+) completed:',
                    r'Company: (\w+)',
                    r'Ticker: (\w+)'
                ]
                
                company_ticker = ''
                for pattern in ticker_patterns:
                    matches = re.findall(pattern, output_text)
                    if matches:
                        # Use the last match (most recent analysis)
                        company_ticker = matches[-1]
                        print(f"DEBUG: Extracted ticker from pattern '{pattern}': {company_ticker}")
                        break
                
                if not company_ticker:
                    # Fallback: try to extract from run_id or other sources
                    if 'company_ticker' in analysis_data:
                        company_ticker = analysis_data['company_ticker']
                        print(f"DEBUG: Using ticker from analysis_data: {company_ticker}")
                    else:
                        company_ticker = ''
                        print("DEBUG: Could not extract ticker from output")
                
                # Set basic company info
                company = {'ticker': company_ticker, 'company_name': company_ticker}
                composite_score = {'score': None, 'confidence': None, 'components_count': 0}
                consensus_scores = {}
                individual_model_results = {}
                execution_metadata = {}
            else:
                # Regular analysis data
                company_ticker = company.get('ticker', '')
                print(f"DEBUG: Regular analysis data, ticker: {company_ticker}")
        
        # Prepare the data for insertion
        insert_data = {
            'run_id': run_id,
            'company_ticker': company_ticker,
            'analysis_timestamp': metadata.get('timestamp', int(time.time())),
            'analysis_type': metadata.get('analysis_type', 'multi_llm'),
            'models_used': metadata.get('models_used', []),
            'successful_models': execution_metadata.get('successful_models', []),
            'failed_models': execution_metadata.get('failed_models', []),
            'consensus_method': execution_metadata.get('consensus_method', ''),
            'execution_time_seconds': analysis_data.get('total_time_seconds', 0),
            'total_cost_usd': analysis_data.get('total_cost_usd', 0),
            'concurrent_execution': execution_metadata.get('concurrent_execution', False),
            
            # Company information
            'company_name': company.get('company_name', ''),
            'company_subsector': company.get('subsector', ''),
            'company_market_cap_usd': company.get('market_cap_usd'),
            'company_employees': company.get('employees'),
            'company_founded_year': company.get('founded_year'),
            'company_headquarters': company.get('headquarters', ''),
            'company_description': company.get('description', ''),
            'company_website': company.get('website', ''),
            'company_status': company.get('status', ''),
            'company_id': company.get('id', ''),
            'company_created_at': company.get('created_at'),
            'company_updated_at': company.get('updated_at'),
            
            # Composite scores
            'composite_score': composite_score.get('score'),
            'composite_confidence': composite_score.get('confidence'),
            'composite_components_count': composite_score.get('components_count'),
            
            # Best model
            'best_model': analysis_data.get('best_model', ''),
            'total_weight_used': execution_metadata.get('scoring_metadata', {}).get('total_weight_used'),
            'components_used': execution_metadata.get('scoring_metadata', {}).get('components_used'),
            
            # Full JSON data - keep as-is with default=str for any non-serializable values
            'full_analysis_data': json.dumps(analysis_data, default=str),
            'individual_model_results': json.dumps(individual_model_results, default=str),
            'execution_metadata': json.dumps(execution_metadata, default=str)
        }
        
        # Initialize all component scores with default values
        component_fields = [
            'moat_brand_monopoly', 'moat_barriers_to_entry', 'moat_economies_of_scale',
            'moat_network_effects', 'moat_switching_costs', 'competitive_differentiation',
            'technology_moats', 'market_timing', 'management_quality', 'key_growth_drivers',
            'transformation_potential', 'platform_expansion', 'major_risk_factors', 'red_flags'
        ]
        
        for component in component_fields:
            insert_data[f'{component}_score'] = None
            insert_data[f'{component}_confidence'] = None
            insert_data[f'{component}_justification'] = ''
        
        # Extract consensus scores
        if isinstance(consensus_scores, dict) and consensus_scores:
            for score_key, score_data in consensus_scores.items():
                if isinstance(score_data, dict):
                    # Map score keys to database columns
                    if score_key == 'moat_brand_monopoly':
                        insert_data['moat_brand_monopoly_score'] = score_data.get('score')
                        insert_data['moat_brand_monopoly_confidence'] = score_data.get('confidence')
                        insert_data['moat_brand_monopoly_justification'] = score_data.get('justification', '')
                    elif score_key == 'moat_barriers_to_entry':
                        insert_data['moat_barriers_to_entry_score'] = score_data.get('score')
                        insert_data['moat_barriers_to_entry_confidence'] = score_data.get('confidence')
                        insert_data['moat_barriers_to_entry_justification'] = score_data.get('justification', '')
                    elif score_key == 'moat_economies_of_scale':
                        insert_data['moat_economies_of_scale_score'] = score_data.get('score')
                        insert_data['moat_economies_of_scale_confidence'] = score_data.get('confidence')
                        insert_data['moat_economies_of_scale_justification'] = score_data.get('justification', '')
                    elif score_key == 'moat_network_effects':
                        insert_data['moat_network_effects_score'] = score_data.get('score')
                        insert_data['moat_network_effects_confidence'] = score_data.get('confidence')
                        insert_data['moat_network_effects_justification'] = score_data.get('justification', '')
                    elif score_key == 'moat_switching_costs':
                        insert_data['moat_switching_costs_score'] = score_data.get('score')
                        insert_data['moat_switching_costs_confidence'] = score_data.get('confidence')
                        insert_data['moat_switching_costs_justification'] = score_data.get('justification', '')
                    elif score_key == 'competitive_differentiation':
                        insert_data['competitive_differentiation_score'] = score_data.get('score')
                        insert_data['competitive_differentiation_confidence'] = score_data.get('confidence')
                        insert_data['competitive_differentiation_justification'] = score_data.get('justification', '')
                    elif score_key == 'technology_moats':
                        insert_data['technology_moats_score'] = score_data.get('score')
                        insert_data['technology_moats_confidence'] = score_data.get('confidence')
                        insert_data['technology_moats_justification'] = score_data.get('justification', '')
                    elif score_key == 'market_timing':
                        insert_data['market_timing_score'] = score_data.get('score')
                        insert_data['market_timing_confidence'] = score_data.get('confidence')
                        insert_data['market_timing_justification'] = score_data.get('justification', '')
                    elif score_key == 'management_quality':
                        insert_data['management_quality_score'] = score_data.get('score')
                        insert_data['management_quality_confidence'] = score_data.get('confidence')
                        insert_data['management_quality_justification'] = score_data.get('justification', '')
                    elif score_key == 'key_growth_drivers':
                        insert_data['key_growth_drivers_score'] = score_data.get('score')
                        insert_data['key_growth_drivers_confidence'] = score_data.get('confidence')
                        insert_data['key_growth_drivers_justification'] = score_data.get('justification', '')
                    elif score_key == 'transformation_potential':
                        insert_data['transformation_potential_score'] = score_data.get('score')
                        insert_data['transformation_potential_confidence'] = score_data.get('confidence')
                        insert_data['transformation_potential_justification'] = score_data.get('justification', '')
                    elif score_key == 'platform_expansion':
                        insert_data['platform_expansion_score'] = score_data.get('score')
                        insert_data['platform_expansion_confidence'] = score_data.get('confidence')
                        insert_data['platform_expansion_justification'] = score_data.get('justification', '')
                    elif score_key == 'major_risk_factors':
                        insert_data['major_risk_factors_score'] = score_data.get('score')
                        insert_data['major_risk_factors_confidence'] = score_data.get('confidence')
                        insert_data['major_risk_factors_justification'] = score_data.get('justification', '')
                    elif score_key == 'red_flags':
                        insert_data['red_flags_score'] = score_data.get('score')
                        insert_data['red_flags_confidence'] = score_data.get('confidence')
                        insert_data['red_flags_justification'] = score_data.get('justification', '')
        else:
            print(f"DEBUG: consensus_scores is empty or not a dict: {consensus_scores}")
        
        # Convert numpy types to Python native types
        for key, value in insert_data.items():
            if hasattr(value, 'dtype'):  # numpy array
                if value.dtype.name.startswith('float'):
                    insert_data[key] = float(value) if pd.notnull(value) else None
                elif value.dtype.name.startswith('int'):
                    insert_data[key] = int(value) if pd.notnull(value) else None
            elif isinstance(value, (np.floating, np.integer)):
                insert_data[key] = value.item() if pd.notnull(value) else None
        
        # Insert into database
        insert_query = """
        INSERT INTO qualitative_analysis (
            run_id, company_ticker, analysis_timestamp, analysis_type, models_used, 
            successful_models, failed_models, consensus_method, execution_time_seconds, 
            total_cost_usd, concurrent_execution, company_name, company_subsector, 
            company_market_cap_usd, company_employees, company_founded_year, 
            company_headquarters, company_description, company_website, company_status, 
            company_id, company_created_at, company_updated_at, composite_score, 
            composite_confidence, composite_components_count, moat_brand_monopoly_score, 
            moat_brand_monopoly_confidence, moat_brand_monopoly_justification, 
            moat_barriers_to_entry_score, moat_barriers_to_entry_confidence, 
            moat_barriers_to_entry_justification, moat_economies_of_scale_score, 
            moat_economies_of_scale_confidence, moat_economies_of_scale_justification, 
            moat_network_effects_score, moat_network_effects_confidence, 
            moat_network_effects_justification, moat_switching_costs_score, 
            moat_switching_costs_confidence, moat_switching_costs_justification, 
            competitive_differentiation_score, competitive_differentiation_confidence, 
            competitive_differentiation_justification, technology_moats_score, 
            technology_moats_confidence, technology_moats_justification, 
            market_timing_score, market_timing_confidence, market_timing_justification, 
            management_quality_score, management_quality_confidence, 
            management_quality_justification, key_growth_drivers_score, 
            key_growth_drivers_confidence, key_growth_drivers_justification, 
            transformation_potential_score, transformation_potential_confidence, 
            transformation_potential_justification, platform_expansion_score, 
            platform_expansion_confidence, platform_expansion_justification, 
            major_risk_factors_score, major_risk_factors_confidence, 
            major_risk_factors_justification, red_flags_score, red_flags_confidence, 
            red_flags_justification, best_model, total_weight_used, components_used, 
            full_analysis_data, individual_model_results, execution_metadata
        ) VALUES (
            :run_id, :company_ticker, :analysis_timestamp, :analysis_type, :models_used, 
            :successful_models, :failed_models, :consensus_method, :execution_time_seconds, 
            :total_cost_usd, :concurrent_execution, :company_name, :company_subsector, 
            :company_market_cap_usd, :company_employees, :company_founded_year, 
            :company_headquarters, :company_description, :company_website, :company_status, 
            :company_id, :company_created_at, :company_updated_at, :composite_score, 
            :composite_confidence, :composite_components_count, :moat_brand_monopoly_score, 
            :moat_brand_monopoly_confidence, :moat_brand_monopoly_justification, 
            :moat_barriers_to_entry_score, :moat_barriers_to_entry_confidence, 
            :moat_barriers_to_entry_justification, :moat_economies_of_scale_score, 
            :moat_economies_of_scale_confidence, :moat_economies_of_scale_justification, 
            :moat_network_effects_score, :moat_network_effects_confidence, 
            :moat_network_effects_justification, :moat_switching_costs_score, 
            :moat_switching_costs_confidence, :moat_switching_costs_justification, 
            :competitive_differentiation_score, :competitive_differentiation_confidence, 
            :competitive_differentiation_justification, :technology_moats_score, 
            :technology_moats_confidence, :technology_moats_justification, 
            :market_timing_score, :market_timing_confidence, :market_timing_justification, 
            :management_quality_score, :management_quality_confidence, 
            :management_quality_justification, :key_growth_drivers_score, 
            :key_growth_drivers_confidence, :key_growth_drivers_justification, 
            :transformation_potential_score, :transformation_potential_confidence, 
            :transformation_potential_justification, :platform_expansion_score, 
            :platform_expansion_confidence, :platform_expansion_justification, 
            :major_risk_factors_score, :major_risk_factors_confidence, 
            :major_risk_factors_justification, :red_flags_score, :red_flags_confidence, 
            :red_flags_justification, :best_model, :total_weight_used, :components_used, 
            :full_analysis_data, :individual_model_results, :execution_metadata
        )
        """
        
        with engine.begin() as conn:
            conn.execute(text(insert_query), insert_data)
        
        return True, f"Successfully saved qualitative analysis for {company.get('ticker', 'Unknown')}"
        
    except Exception as e:
        return False, f"Error saving qualitative analysis: {str(e)}"


def load_qualitative_analysis(ticker=None, run_id=None, limit=100):
    """Load qualitative analysis results from database"""
    try:
        engine = get_engine()
        
        # Build query
        query = "SELECT * FROM qualitative_analysis WHERE 1=1"
        params = {}
        
        if ticker:
            query += " AND company_ticker = :ticker"
            params['ticker'] = ticker
        
        if run_id:
            query += " AND run_id = :run_id"
            params['run_id'] = run_id
        
        query += " ORDER BY analysis_timestamp DESC, created_at DESC"
        
        if limit:
            query += " LIMIT :limit"
            params['limit'] = limit
        
        # Execute query using SQLAlchemy connection
        with engine.connect() as conn:
            if params:
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))
            
            # Convert result to DataFrame
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        return df
        
    except Exception as e:
        print(f"Error loading qualitative analysis: {e}")
        return pd.DataFrame()
