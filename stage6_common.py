from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from json_repair import repair_json

ROOT = Path(__file__).resolve().parent
BASELINE_MODEL = "openai/gpt-oss-20b"
ALLOWED_LABELS = {"supported", "contradicted", "insufficient_evidence"}
SEED = 20260725
TEMPERATURE = 0
MAX_TOKENS = 2400
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def final_human_label(row: dict[str, str]) -> str:
    for key in (
        "adjudicated_label",
        "final_human_label",
        "annotator_A_label",
        "annotator_1_label",
    ):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


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
    """
    Extract Groq's retry window from messages such as:
    'Please try again in 9m25.92s' or 'try again in 42.5s'.
    """
    text = str(exc)
    match = re.search(
        r"try again in\s*(?:(\d+(?:\.\d+)?)m)?\s*(?:(\d+(?:\.\d+)?)s)?",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        minutes = float(match.group(1) or 0)
        seconds = float(match.group(2) or 0)
        return minutes * 60 + seconds
    if "rate limit" in text.lower() or "error code: 429" in text.lower():
        return 60.0
    return None


def call_json(
    client: Groq,
    model: str,
    messages: list[dict[str, str]],
    call_name: str,
    max_tokens: int = MAX_TOKENS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    JSON call with:
    - malformed-JSON recovery;
    - structured-mode fallback;
    - automatic waiting for Groq 429 retry windows.

    Rate-limit waits do not consume the normal three non-rate retry attempts.
    """
    last_error = None
    max_rate_wait = float(
        os.getenv("STAGE6_MAX_RATE_WAIT_SECONDS", "1800")
    )
    max_rate_wait_events = int(
        os.getenv("STAGE6_MAX_RATE_WAIT_EVENTS", "20")
    )
    rate_wait_events = 0
    attempt = 0

    structured_non_rate_failures = 0
    while structured_non_rate_failures < 3:
        attempt += 1
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                seed=SEED,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            raw = response.choices[0].message.content or ""
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
                "model": model,
                "attempt": attempt,
                "response_mode": "json_object",
                "json_repaired": repaired,
                "latency_ms": latency_ms,
                "usage": usage,
                "raw_response": raw,
                "rate_limit_wait_events": rate_wait_events,
            }
        except Exception as exc:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)

            failed = failed_generation_from_exception(exc)
            if failed:
                try:
                    parsed, repaired = parse_json_object(failed)
                    time.sleep(CALL_SPACING_SECONDS)
                    return parsed, {
                        "call_name": call_name,
                        "model": model,
                        "attempt": attempt,
                        "response_mode": "failed_generation_repair",
                        "json_repaired": repaired,
                        "latency_ms": latency_ms,
                        "usage": None,
                        "raw_response": failed,
                        "original_error": f"{type(exc).__name__}: {exc}",
                        "rate_limit_wait_events": rate_wait_events,
                    }
                except Exception as repair_exc:
                    last_error = (
                        f"{type(exc).__name__}: {exc}; repair failed: "
                        f"{type(repair_exc).__name__}: {repair_exc}"
                    )

            retry_seconds = retry_after_seconds(exc)
            if retry_seconds is not None:
                rate_wait_events += 1
                if rate_wait_events > max_rate_wait_events:
                    raise RuntimeError(
                        f"{call_name} exceeded {max_rate_wait_events} "
                        "rate-limit wait events."
                    ) from exc
                if retry_seconds > max_rate_wait:
                    raise RuntimeError(
                        f"{call_name} requires a {retry_seconds:.1f}s wait, "
                        f"which exceeds STAGE6_MAX_RATE_WAIT_SECONDS="
                        f"{max_rate_wait:.1f}."
                    ) from exc
                wait_seconds = retry_seconds + 5.0
                print(
                    f"{call_name}: Groq rate limit reached; waiting "
                    f"{wait_seconds:.1f}s, then resuming automatically."
                )
                time.sleep(wait_seconds)
                continue

            structured_non_rate_failures += 1
            last_error = f"{type(exc).__name__}: {exc}"
            if structured_non_rate_failures < 3:
                time.sleep(min(5 * structured_non_rate_failures, 15))

    fallback_messages = list(messages)
    fallback_messages[0] = {
        "role": "system",
        "content": (
            messages[0]["content"]
            + "\nReturn exactly one JSON object. Do not use markdown fences, "
              "comments or prose outside the JSON."
        ),
    }

    fallback_non_rate_failures = 0
    while fallback_non_rate_failures < 3:
        attempt += 1
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=fallback_messages,
                temperature=TEMPERATURE,
                max_completion_tokens=max_tokens,
            )
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            raw = response.choices[0].message.content or ""
            if not raw.strip():
                raise ValueError("Fallback returned an empty response.")
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
                "model": model,
                "attempt": attempt,
                "response_mode": "plain_text_json_fallback",
                "json_repaired": repaired,
                "latency_ms": latency_ms,
                "usage": usage,
                "raw_response": raw,
                "structured_mode_last_error": last_error,
                "rate_limit_wait_events": rate_wait_events,
            }
        except Exception as exc:
            retry_seconds = retry_after_seconds(exc)
            if retry_seconds is not None:
                rate_wait_events += 1
                if rate_wait_events > max_rate_wait_events:
                    raise RuntimeError(
                        f"{call_name} exceeded {max_rate_wait_events} "
                        "rate-limit wait events."
                    ) from exc
                if retry_seconds > max_rate_wait:
                    raise RuntimeError(
                        f"{call_name} requires a {retry_seconds:.1f}s wait, "
                        f"which exceeds STAGE6_MAX_RATE_WAIT_SECONDS="
                        f"{max_rate_wait:.1f}."
                    ) from exc
                wait_seconds = retry_seconds + 5.0
                print(
                    f"{call_name}: Groq rate limit reached during fallback; "
                    f"waiting {wait_seconds:.1f}s."
                )
                time.sleep(wait_seconds)
                continue

            fallback_non_rate_failures += 1
            last_error = f"{type(exc).__name__}: {exc}"
            if fallback_non_rate_failures < 3:
                time.sleep(min(10 * fallback_non_rate_failures, 30))

    raise RuntimeError(f"{call_name} failed after retries: {last_error}")


def groq_client() -> Groq:
    load_dotenv(ROOT / ".env")
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key == "PASTE_YOUR_KEY_HERE":
        raise RuntimeError("GROQ_API_KEY is missing from .env.")
    return Groq(api_key=key)


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    return label if label in ALLOWED_LABELS else "insufficient_evidence"


def cohen_kappa(actual: list[str], predicted: list[str]) -> float | None:
    if not actual or len(actual) != len(predicted):
        return None
    n = len(actual)
    observed = sum(a == b for a, b in zip(actual, predicted)) / n
    labels = sorted(ALLOWED_LABELS)
    actual_counts = Counter(actual)
    predicted_counts = Counter(predicted)
    expected = sum(
        (actual_counts[label] / n) * (predicted_counts[label] / n)
        for label in labels
    )
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def metrics_from_pairs(
    actual: list[str],
    predicted: list[str],
) -> dict[str, Any]:
    labels = ["supported", "contradicted", "insufficient_evidence"]
    matrix = {
        a: {p: 0 for p in labels}
        for a in labels
    }
    for a, p in zip(actual, predicted):
        matrix[a][p] += 1

    per_class = {}
    f1_observed = []
    weighted_f1_num = 0.0
    weighted_f1_den = 0
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[a][label] for a in labels if a != label)
        fn = sum(matrix[label][p] for p in labels if p != label)
        support = sum(matrix[label].values())
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        if precision is None or recall is None or precision + recall == 0:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class[label] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if support and f1 is not None:
            f1_observed.append(f1)
            weighted_f1_num += support * f1
            weighted_f1_den += support

    correct = sum(a == p for a, p in zip(actual, predicted))
    supported_predictions = sum(p == "supported" for p in predicted)
    false_supports = sum(
        p == "supported" and a != "supported"
        for a, p in zip(actual, predicted)
    )
    return {
        "claim_count": len(actual),
        "accuracy": correct / len(actual) if actual else None,
        "cohen_kappa": cohen_kappa(actual, predicted),
        "macro_f1_observed_classes": (
            statistics.mean(f1_observed) if f1_observed else None
        ),
        "weighted_f1_observed_classes": (
            weighted_f1_num / weighted_f1_den if weighted_f1_den else None
        ),
        "false_support_count": false_supports,
        "false_support_rate_among_supported_predictions": (
            false_supports / supported_predictions
            if supported_predictions
            else None
        ),
        "per_class": per_class,
        "confusion_matrix_actual_rows_predicted_columns": matrix,
    }
