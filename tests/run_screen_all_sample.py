import json
import sys
from pathlib import Path
from datetime import datetime

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.fundamental_agent import FundamentalAgent  # type: ignore


SAMPLE_INDUSTRIES = [
    "Semiconductors",
    "Software - Application",
]


def main() -> None:
    out_dir = Path(REPO_ROOT / "test-data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "screen_all_sample.json"

    results = {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z", "industries": []}

    agent = FundamentalAgent()

    for industry in SAMPLE_INDUSTRIES:
        try:
            df = agent.screen_sector(
                region="US",
                sectors=None,
                industries=industry,
                min_cap=0,
                max_cap=float("inf"),
                verbose=False,
            )
            df = df if df is not None else []
            top = []
            if hasattr(df, "head"):
                top = df.head(5).to_dict(orient="records")
                count = int(len(df))
            else:
                count = 0
            results["industries"].append({
                "industry": industry,
                "count": count,
                "top": top,
            })
        except Exception as e:
            results["industries"].append({
                "industry": industry,
                "status": "error",
                "error": str(e),
            })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
