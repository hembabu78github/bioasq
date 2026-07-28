from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "stage0"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def serialise_model(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        raw = model.model_dump()
    elif hasattr(model, "dict"):
        raw = model.dict()
    else:
        raw = {"repr": repr(model)}
    return raw


def main() -> int:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key or api_key == "PASTE_YOUR_KEY_HERE":
        print("ERROR: GROQ_API_KEY is missing from .env.")
        return 2

    client = Groq(api_key=api_key)

    try:
        response = client.models.list()
    except Exception as exc:
        error_report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        out = OUTPUT_DIR / "groq_models.json"
        out.write_text(json.dumps(error_report, indent=2), encoding="utf-8")
        print(f"Groq model query failed. Details saved to: {out}")
        return 1

    models = [serialise_model(item) for item in response.data]
    models.sort(key=lambda item: str(item.get("id", "")))

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "success": True,
        "model_count": len(models),
        "models": models,
        "note": (
            "This file records models visible to the account at the time of the query. "
            "A later experiment configuration must record the exact selected model."
        ),
    }

    out = OUTPUT_DIR / "groq_models.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Groq connection successful. {len(models)} model(s) found.")
    for item in models:
        print(f"- {item.get('id', '<unknown id>')}")
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
