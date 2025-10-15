import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
import uuid

import pandas as pd
from sqlalchemy import text
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.db_util import save_fundamental_screen, get_engine, ensure_table_exists  # type: ignore


def main() -> None:
    out_dir = REPO_ROOT / "test-data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "db_util_test.json"

    ensure_table_exists()

    # Create a unique run_id for this test
    run_id = f"test-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

    # Read CSV export from the screener
    csv_path = REPO_ROOT / "test-data" / "fundamental_screen_results.csv"
    if not csv_path.exists():
        result = {
            "status": "error",
            "error": f"CSV not found: {csv_path}",
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
        return

    df = pd.read_csv(csv_path)
    # Try to recover ticker if stored as first unnamed column
    if "ticker" not in df.columns:
        for candidate in ["index", "Index", "Unnamed: 0", "symbol", "Symbol"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "ticker"})
                break
    if "ticker" not in df.columns:
        result = {
            "status": "error",
            "error": "CSV missing required 'ticker' column",
            "columns": list(df.columns),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
        return

    result = {"status": "ok", "run_id": run_id}

    try:
        saved_run_id = save_fundamental_screen(df, run_id=run_id)
        engine = get_engine()
        with engine.begin() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM fundamental_screen WHERE run_id = :rid"),
                {"rid": saved_run_id},
            ).scalar()
        result.update({"inserted": int(count), "source_csv": str(csv_path.relative_to(REPO_ROOT))})
    except Exception as e:
        result = {"status": "error", "error": str(e)}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
