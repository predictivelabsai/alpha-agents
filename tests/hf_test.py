import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


def call_hf_inference(model: str, prompt: str, hf_key: str) -> dict:
    try:
        client = InferenceClient(model=model, token=hf_key)
        text = client.text_generation(
            prompt=prompt,
            max_new_tokens=128,
            temperature=0.2,
        )
        return {"status": "ok", "output": text}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main() -> None:
    # Ensure repo root on path and outputs dir exists
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    out_dir = repo_root / "test-results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "hf_test.json"

    # Load environment variables
    load_dotenv()
    hf_key = os.getenv("HF_KEY")
    if not hf_key:
        result = {"status": "error", "error": "HF_KEY not set in environment or .env"}
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    # Popular open models on HF (ensure they have inference enabled on the hub)
    models = [
        "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]

    run_id = f"hf-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    prompt = "### Instruction:\nSay 'Hello, world!' and identify your model in one short sentence.\n### Response:\n"
    results: list[dict] = []

    for model in models:
        entry = {"model": model}
        entry.update(call_hf_inference(model, prompt, hf_key))
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


