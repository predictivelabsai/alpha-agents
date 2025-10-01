import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import glob
import pandas as pd

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]


def find_latest_csv(pattern: str) -> Path | None:
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return Path(matches[-1])


def main() -> None:
    out_dir = REPO_ROOT / "test-data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "fundamental_agent_cli_sample.json"

    cmd = [
        sys.executable,
        "-m",
        "agents.fundamental_agent",
        "US",
        "--industry",
        "Semiconductors",
    ]

    result = {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "command": " ".join(cmd),
    }

    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        result["status"] = "cli_error"
        result["returncode"] = e.returncode
        result["stderr"] = e.stderr.decode("utf-8", errors="ignore")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return

    csv_dir = REPO_ROOT / "test-data" / "screener"
    csv_dir.mkdir(exist_ok=True)

    latest_csv = find_latest_csv(str(csv_dir / "semiconductors_*.csv"))
    if latest_csv and latest_csv.exists():
        try:
            df = pd.read_csv(latest_csv)
            result["csv_path"] = str(latest_csv.relative_to(REPO_ROOT))
            result["count"] = int(len(df))
            result["top"] = df.head(5).to_dict(orient="records")
        except Exception as e:
            result["status"] = "parse_error"
            result["error"] = str(e)
    else:
        result["status"] = "missing_output"
        result["message"] = "CLI did not produce the expected CSV file."

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
