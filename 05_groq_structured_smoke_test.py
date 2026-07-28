from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq

ROOT = Path(__file__).resolve().parent
PILOT_PATH = ROOT / "data" / "processed" / "pilot" / "bioasq11_pilot_80.json"
OUTPUT_DIR = ROOT / "outputs" / "stage1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "openai/gpt-oss-20b"
SEED = 20260725
MAX_COMPLETION_TOKENS = 450


def evidence_for(q: dict[str, Any], limit: int = 4) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for index, snippet in enumerate(q.get("snippets", []), start=1):
        if not isinstance(snippet, dict):
            continue
        text = snippet.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        items.append({
            "evidence_id": f"E{index}",
            "document": str(snippet.get("document", "")),
            "text": text.strip(),
        })
        if len(items) >= limit:
            break
    return items


def messages_for(q: dict[str, Any], evidence: list[dict[str, str]]) -> list[dict[str, str]]:
    evidence_text = "\n\n".join(
        f"[{item['evidence_id']}] {item['text']}" for item in evidence
    )
    return [
        {
            "role": "system",
            "content": (
                "Use only the supplied biomedical evidence. Return one valid JSON object "
                "and no markdown. Include answer (string or list), supported (boolean), "
                "supporting_evidence_ids (list of strings), and brief_rationale (string). "
                "When evidence is insufficient, set supported to false and answer to "
                "\"INSUFFICIENT_EVIDENCE\"."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question type: {q.get('type')}\n"
                f"Question: {q.get('body')}\n\n"
                f"Evidence:\n{evidence_text}"
            ),
        },
    ]


def parse_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(text)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(value, dict):
        return None, "Top-level JSON value is not an object."
    return value, None


def main() -> int:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "PASTE_YOUR_KEY_HERE":
        print("ERROR: GROQ_API_KEY is missing from .env.")
        return 2
    if not PILOT_PATH.exists():
        print(f"ERROR: Missing pilot file: {PILOT_PATH}")
        return 1

    questions = json.loads(PILOT_PATH.read_text(encoding="utf-8")).get("questions", [])
    selected: list[dict[str, Any]] = []
    seen_types: set[str] = set()

    for q in questions:
        qtype = str(q.get("type", "missing"))
        if qtype not in seen_types:
            selected.append(q)
            seen_types.add(qtype)

    client = Groq(api_key=api_key)
    result_path = OUTPUT_DIR / "groq_smoke_test_results.jsonl"
    records: list[dict[str, Any]] = []

    with result_path.open("w", encoding="utf-8") as handle:
        for q in selected:
            evidence = evidence_for(q)
            messages = messages_for(q, evidence)
            started = time.perf_counter()
            raw = ""
            parsed = None
            error = None
            usage = None

            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0,
                    seed=SEED,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content or ""
                parsed, error = parse_object(raw)
                if getattr(response, "usage", None) is not None:
                    usage = (
                        response.usage.model_dump()
                        if hasattr(response.usage, "model_dump")
                        else {"repr": repr(response.usage)}
                    )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            record = {
                "run_utc": datetime.now(timezone.utc).isoformat(),
                "question_id": q.get("id"),
                "question_type": q.get("type"),
                "question": q.get("body"),
                "model": MODEL,
                "seed": SEED,
                "temperature": 0,
                "max_completion_tokens": MAX_COMPLETION_TOKENS,
                "evidence": evidence,
                "messages": messages,
                "raw_response": raw,
                "parsed_response": parsed,
                "valid_json_object": parsed is not None,
                "error": error,
                "latency_ms": latency_ms,
                "usage": usage,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"{str(q.get('type')):<8} "
                f"JSON={record['valid_json_object']} "
                f"latency={latency_ms:.0f} ms"
            )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "question_count": len(records),
        "valid_json_count": sum(r["valid_json_object"] for r in records),
        "error_count": sum(bool(r["error"]) for r in records),
        "mean_latency_ms": (
            round(sum(r["latency_ms"] for r in records) / len(records), 2)
            if records else None
        ),
        "question_types": [r["question_type"] for r in records],
        "result_relative_path": str(result_path.relative_to(ROOT)),
        "status": (
            "pass"
            if records and all(r["valid_json_object"] and not r["error"] for r in records)
            else "review_required"
        ),
    }

    summary_path = OUTPUT_DIR / "groq_smoke_test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSummary: {summary_path}")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
