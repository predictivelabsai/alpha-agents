import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.stock_screener import StockScreener  # type: ignore


DDL_PATH = REPO_ROOT / "sql" / "create_table.sql"

def get_database_url() -> str:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        # Fallback to local sqlite file for convenience
        db_file = REPO_ROOT / "db" / "alpha_agents.db"
        db_file.parent.mkdir(exist_ok=True)
        db_url = f"sqlite:///{db_file}"
    return db_url


def create_table(engine) -> None:
    sql = DDL_PATH.read_text(encoding="utf-8")
    with engine.begin() as conn:
        # Some engines require splitting; text() handles entire DDL for single statement
        conn.execute(text(sql))


def build_sample_dataframe(run_id: str) -> pd.DataFrame:
    screener = StockScreener(region="US", industries="Semiconductors")
    df = screener.screen().reset_index(drop=False)
    # Ensure columns that match DDL exist even if missing
    required_cols = [
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
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.Series([None] * len(df))
    df["run_id"] = run_id
    return df[required_cols]


def insert_dataframe(engine, df: pd.DataFrame, if_exists: str = "append") -> None:
    df.to_sql("fundamental_screen", con=engine, if_exists=if_exists, index=False)


def main(insert: bool = True) -> None:
    engine = create_engine(get_database_url())
    create_table(engine)
    if insert:
        try:
            run_id = os.getenv("RUN_ID") or f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
            df = build_sample_dataframe(run_id)
            insert_dataframe(engine, df)
            print(f"Inserted {len(df)} rows into fundamental_screen with run_id={run_id}")
        except FileNotFoundError:
            print(
                "Universe file missing at utils/universe/us.csv; table created but no data inserted."
            )


if __name__ == "__main__":
    main(insert=True)
