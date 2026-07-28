from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from json_repair import repair_json

ROOT = Path(__file__).resolve().parent
MODEL = "openai/gpt-oss-20b"
SEED = 20260725
TEMPERATURE = 0
ALLOWED_LABELS = {"supported", "contradicted", "insufficient_evidence"}
CALL_SPACING_SECONDS = 2.0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_json_object(raw: str) -> tuple[dict[str, Any], bool]:
    try:
        parsed = json.loads(raw)
        repaired = False
    except Exception:
        parsed = repair_json(raw, return_objects=True)
        repaired = True
    if not isinstance(parsed, dict):
        raise ValueError("Top-level response is not a JSON object.")
    return parsed, repaired


def failed_generation_from_exception(exc: Exception) -> str | None:
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    error = body.get("error", body)
    if not isinstance(error, dict):
        return None
    value = error.get("failed_generation")
    return value if isinstance(value, str) and value.strip() else None


def retry_after_seconds(exc: Exception) -> float | None:
    text = str(exc)
    match = re.search(
        r"try again in\s*(?:(\d+(?:\.\d+)?)m)?\s*(?:(\d+(?:\.\d+)?)s)?",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return float(match.group(1) or 0) * 60 + float(match.group(2) or 0)
    if "rate limit" in text.lower() or "error code: 429" in text.lower():
        return 60.0
    return None


def call_json(
    client: Groq,
    messages: list[dict[str, str]],
    call_name: str,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Call Groq for one JSON object.

    Execution policy:
    1. Try structured JSON mode up to three non-rate failures.
    2. Recover a non-empty failed_generation payload when Groq supplies one.
    3. If structured validation repeatedly fails, retry in plain-text mode
       with an explicit JSON-only instruction and parse/repair the response.
    4. Rate-limit waits do not consume normal retry attempts.
    """
    max_rate_wait = float(
        os.getenv("STAGE7_MAX_RATE_WAIT_SECONDS", "1800")
    )
    max_rate_events = int(
        os.getenv("STAGE7_MAX_RATE_WAIT_EVENTS", "20")
    )
    rate_events = 0
    attempt = 0
    last_error = ""

    def wait_for_rate_limit(exc: Exception, mode: str) -> bool:
        nonlocal rate_events
        wait = retry_after_seconds(exc)
        if wait is None:
            return False
        rate_events += 1
        if rate_events > max_rate_events:
            raise RuntimeError(
                f"{call_name} exceeded rate-limit wait-event limit."
            ) from exc
        if wait > max_rate_wait:
            raise RuntimeError(
                f"{call_name} requires {wait:.1f}s wait, above configured "
                f"maximum {max_rate_wait:.1f}s."
            ) from exc
        wait += 5.0
        print(
            f"{call_name}: Groq rate limit reached in {mode}; waiting "
            f"{wait:.1f}s, then resuming."
        )
        time.sleep(wait)
        return True

    structured_failures = 0
    while structured_failures < 3:
        attempt += 1
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                seed=SEED,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            latency_ms = round(
                (time.perf_counter() - started) * 1000, 2
            )
            raw = response.choices[0].message.content or ""
            if not raw.strip():
                raise ValueError("Structured JSON mode returned an empty response.")
            parsed, repaired = parse_json_object(raw)
            usage = (
                response.usage.model_dump()
                if getattr(response, "usage", None) is not None
                and hasattr(response.usage, "model_dump")
                else None
            )
            time.sleep(CALL_SPACING_SECONDS)
            return parsed, {
                "call_name": call_name,
                "model": MODEL,
                "attempt": attempt,
                "response_mode": "json_object",
                "latency_ms": latency_ms,
                "usage": usage,
                "json_repaired": repaired,
                "raw_response": raw,
                "rate_limit_wait_events": rate_events,
            }
        except Exception as exc:
            failed = failed_generation_from_exception(exc)
            if failed:
                try:
                    parsed, repaired = parse_json_object(failed)
                    time.sleep(CALL_SPACING_SECONDS)
                    return parsed, {
                        "call_name": call_name,
                        "model": MODEL,
                        "attempt": attempt,
                        "response_mode": "failed_generation_repair",
                        "latency_ms": round(
                            (time.perf_counter() - started) * 1000, 2
                        ),
                        "usage": None,
                        "json_repaired": repaired,
                        "raw_response": failed,
                        "original_error": f"{type(exc).__name__}: {exc}",
                        "rate_limit_wait_events": rate_events,
                    }
                except Exception as repair_exc:
                    last_error = (
                        f"{type(exc).__name__}: {exc}; failed_generation "
                        f"repair failed: {type(repair_exc).__name__}: "
                        f"{repair_exc}"
                    )

            if wait_for_rate_limit(exc, "structured JSON mode"):
                continue

            structured_failures += 1
            last_error = f"{type(exc).__name__}: {exc}"
            if structured_failures < 3:
                time.sleep(min(5 * structured_failures, 15))

    fallback_messages = [dict(message) for message in messages]
    fallback_messages[0] = {
        "role": "system",
        "content": (
            fallback_messages[0]["content"]
            + "\n\nIMPORTANT OUTPUT RULE: Return one compact valid JSON object "
              "only. Do not use markdown fences, comments, explanations, "
              "trailing commas, NaN, or text outside the JSON object."
        ),
    }

    fallback_failures = 0
    while fallback_failures < 3:
        attempt += 1
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=fallback_messages,
                temperature=TEMPERATURE,
                seed=SEED,
                max_completion_tokens=max_tokens,
            )
            latency_ms = round(
                (time.perf_counter() - started) * 1000, 2
            )
            raw = response.choices[0].message.content or ""
            if not raw.strip():
                raise ValueError("Plain-text fallback returned an empty response.")
            parsed, repaired = parse_json_object(raw)
            usage = (
                response.usage.model_dump()
                if getattr(response, "usage", None) is not None
                and hasattr(response.usage, "model_dump")
                else None
            )
            time.sleep(CALL_SPACING_SECONDS)
            return parsed, {
                "call_name": call_name,
                "model": MODEL,
                "attempt": attempt,
                "response_mode": "plain_text_json_fallback",
                "latency_ms": latency_ms,
                "usage": usage,
                "json_repaired": repaired,
                "raw_response": raw,
                "structured_mode_last_error": last_error,
                "rate_limit_wait_events": rate_events,
            }
        except Exception as exc:
            if wait_for_rate_limit(exc, "plain-text fallback"):
                continue

            fallback_failures += 1
            last_error = f"{type(exc).__name__}: {exc}"
            if fallback_failures < 3:
                time.sleep(min(10 * fallback_failures, 30))

    raise RuntimeError(
        f"{call_name} failed in structured and fallback modes: {last_error}"
    )


def groq_client() -> Groq:
    load_dotenv(ROOT / ".env")
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Set it in the current PowerShell "
            "session or in D:\\Prog\\JMS_RAG\\.env."
        )
    return Groq(api_key=key)


def format_evidence(evidence: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{row['evidence_id']}] {row['text']}" for row in evidence
    )


def frozen_hardened_verifier_messages(
    question: str,
    evidence: str,
    graph: str,
    claims: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Act as a strict claim-level evidence verifier. Use only the displayed "
                "retrieved evidence and evidence-derived graph. Return JSON with one "
                "`verifications` list. Every input claim_id must appear exactly once.\n\n"
                "Allowed status values:\n"
                "- supported\n"
                "- contradicted\n"
                "- insufficient_evidence\n\n"
                "A claim is supported only when every material component is directly "
                "supported. Separately inspect:\n"
                "- named entities and entity identity;\n"
                "- relation direction;\n"
                "- negation;\n"
                "- date, timing and sequence;\n"
                "- numbers, units and quantities;\n"
                "- population, disease and intervention scope;\n"
                "- comparative or superlative language;\n"
                "- regulatory status;\n"
                "- commercial or market availability;\n"
                "- association versus causation;\n"
                "- investigated versus effective;\n"
                "- treatment versus cure.\n\n"
                "Do not use plausible background knowledge. Do not make inferential "
                "bridges. In particular:\n"
                "- FDA approval does not prove market availability;\n"
                "- being studied does not prove effectiveness;\n"
                "- association does not prove causation;\n"
                "- evidence from one year does not prove a different stated year;\n"
                "- support for a broad relation does not support a stronger superlative.\n\n"
                "Use contradicted only when the displayed evidence directly supports the "
                "opposite or makes the claim incompatible. Use insufficient_evidence when "
                "a material qualifier is absent, ambiguous or only inferable.\n\n"
                "Each verification item must contain:\n"
                "- claim_id\n"
                "- status\n"
                "- evidence_ids\n"
                "- graph_edge_ids\n"
                "- unsupported_or_contradicted_span\n"
                "- material_qualifiers_checked\n"
                "- brief_rationale\n\n"
                "Keep each rationale to 20 words or fewer and keep "
                "material_qualifiers_checked as a compact list."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\nEvidence:\n{evidence}\n\n"
                f"Graph:\n{graph}\n\nClaims:\n"
                f"{json.dumps(claims, ensure_ascii=False)}"
            ),
        },
    ]


def verification_items(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    items = parsed.get("verifications")
    if not isinstance(items, list):
        items = parsed.get("claims")
    return items if isinstance(items, list) else []


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    return label if label in ALLOWED_LABELS else ""


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None
