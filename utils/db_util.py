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
