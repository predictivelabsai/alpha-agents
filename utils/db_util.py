import os
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
import uuid

import pandas as pd
from sqlalchemy import create_engine, text

REPO_ROOT = Path(__file__).resolve().parents[1]
DDL_PATH = REPO_ROOT / "sql" / "create_table.sql"


def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        db_file = REPO_ROOT / "db" / "alpha_agents.db"
        db_file.parent.mkdir(exist_ok=True)
        db_url = f"sqlite:///{db_file}"
    return create_engine(db_url)


def ensure_table_exists() -> None:
    engine = get_engine()
    # Robust dialect detection to avoid running PostgreSQL DDL on SQLite
    dialect = (getattr(engine.url, "get_backend_name", lambda: engine.dialect.name)()).lower()
    if "sqlite" in dialect:
        sql = (
            "CREATE TABLE IF NOT EXISTS fundamental_screen (\n"
            "    id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
            "    run_id TEXT NOT NULL,\n"
            "    ticker TEXT NOT NULL,\n"
            "    company_name TEXT,\n"
            "    sector TEXT,\n"
            "    industry TEXT,\n"
            "    market_cap REAL,\n"
            "    gross_profit_margin_ttm REAL,\n"
            "    operating_profit_margin_ttm REAL,\n"
            "    net_profit_margin_ttm REAL,\n"
            "    ebit_margin_ttm REAL,\n"
            "    ebitda_margin_ttm REAL,\n"
            "    roa_ttm REAL,\n"
            "    roe_ttm REAL,\n"
            "    roic_ttm REAL,\n"
            "    total_revenue_4y_cagr REAL,\n"
            "    net_income_4y_cagr REAL,\n"
            "    operating_cash_flow_4y_cagr REAL,\n"
            "    total_revenue_4y_consistency REAL,\n"
            "    net_income_4y_consistency REAL,\n"
            "    operating_cash_flow_4y_consistency REAL,\n"
            "    current_ratio REAL,\n"
            "    debt_to_ebitda_ratio REAL,\n"
            "    debt_servicing_ratio REAL,\n"
            "    score REAL,\n"
            "    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,\n"
            "    UNIQUE (run_id, ticker)\n"
            ");"
        )
    else:
        # Default to PostgreSQL DDL
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
