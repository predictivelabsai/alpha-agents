import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    # Ensure repo root on path and outputs dir exists
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    out_dir = repo_root / "test-results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "openrouter_test.json"

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        result = {
            "status": "error",
            "error": "OPENROUTER_API_KEY not set in environment or .env",
        }
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    # Popular open-source models on OpenRouter
    models = [
        # DeepSeek
        "deepseek/deepseek-chat",
        # Kimi (Moonshot)
        "moonshotai/moonshot-v1-8k",
        # Qwen
        "qwen/qwen-2.5-7b-instruct",
        # Llama
        "meta-llama/llama-3.1-8b-instruct",
    ]

    run_id = f"openrouter-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    prompt = "Say 'Hello, world!' and identify your model in one short sentence."
    results: list[dict] = []

    for model in models:
        entry: dict = {"model": model}
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a concise assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=64,
                temperature=0.2,
            )
            text = resp.choices[0].message.content if resp.choices else ""
            entry.update({"status": "ok", "output": text})
        except Exception as e:
            entry.update({"status": "error", "error": str(e)})
        results.append(entry)

    summary = {
        "status": "ok",
        "run_id": run_id,
        "prompt": prompt,
        "results": results,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


