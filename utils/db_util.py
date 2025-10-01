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
    sql = DDL_PATH.read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.execute(text(sql))


def save_fundamental_screen(df: pd.DataFrame, run_id: Optional[str] = None) -> str:
    engine = get_engine()
    ensure_table_exists()
    if run_id is None:
        run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    df_to_save = df.copy()
    if "run_id" not in df_to_save.columns:
        df_to_save.insert(0, "run_id", run_id)
    df_to_save.to_sql("fundamental_screen", con=engine, if_exists="append", index=False)
    return run_id
