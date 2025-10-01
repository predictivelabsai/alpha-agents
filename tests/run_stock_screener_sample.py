import json
import sys
from pathlib import Path

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.stock_screener import StockScreener  # type: ignore


def main() -> None:
    out_dir = Path(REPO_ROOT / "test-data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "stock_screener_sample.json"

    payload = {}
    try:
        screener = StockScreener(
            region="US",
            industries="Semiconductors",
            min_cap=0,
            max_cap=float("inf"),
        )
        df = screener.screen()
        df = df.reset_index(drop=False)
        payload = {
            "status": "ok",
            "industry": "Semiconductors",
            "count": int(len(df)),
            "top": df.head(10).to_dict(orient="records"),
        }
    except FileNotFoundError as e:
        payload = {
            "status": "missing_universe",
            "message": (
                "Stock universe CSV not found. Generate it first at utils/universe/us.csv."
            ),
            "error": str(e),
        }
    except Exception as e:
        payload = {
            "status": "error",
            "error": str(e),
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
