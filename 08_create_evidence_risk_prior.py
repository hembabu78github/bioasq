from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "bioasq11" / "training11b.json"
DERIVED_DIR = ROOT / "data" / "processed" / "stage2"
OUTPUT_DIR = ROOT / "outputs" / "stage2"
DERIVED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = DERIVED_DIR / "query_evidence_risk_prior.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "evidence_risk_summary.json"

RELATION_CUES = re.compile(
    r"\b(association|associated|relationship|relation|interact|interaction|"
    r"effect|affect|cause|causal|mechanism|role|pathway|regulate|inhibit|activate|"
    r"treat|treatment|therapy|diagnos|predict|risk|increase|decrease)\b",
    re.IGNORECASE,
)
COMPARISON_CUES = re.compile(
    r"\b(compare|difference|versus|vs\.?|better|worse|higher|lower|similar|"
    r"different|most|least)\b",
    re.IGNORECASE,
)
MULTICLAUSE_CUES = re.compile(r"\b(and|or|while|whereas|however|respectively)\b", re.IGNORECASE)
EXPLANATION_CUES = re.compile(r"^\s*(how|why|describe|explain|what are the mechanisms?)\b", re.IGNORECASE)


def classify(q: dict[str, Any]) -> dict[str, Any]:
    body = str(q.get("body", "")).strip()
    qtype = str(q.get("type", "missing")).lower().strip()
    word_count = len(body.split())

    base = {
        "yesno": 0,
        "factoid": 0,
        "list": 1,
        "summary": 2,
    }.get(qtype, 1)

    relation = bool(RELATION_CUES.search(body))
    comparison = bool(COMPARISON_CUES.search(body))
    multiclause = bool(MULTICLAUSE_CUES.search(body))
    explanation = bool(EXPLANATION_CUES.search(body))
    long_query = word_count >= 12

    score = base
    score += 1 if relation else 0
    score += 1 if comparison else 0
    score += 1 if explanation else 0
    score += 1 if long_query and multiclause else 0

    if score <= 1:
        label = "low"
    elif score <= 3:
        label = "medium"
    else:
        label = "high"

    return {
        "question_id": str(q.get("id", "")),
        "question_type": qtype,
        "word_count": word_count,
        "features": {
            "relation_or_mechanism_cue": relation,
            "comparison_cue": comparison,
            "multi_clause_cue": multiclause,
            "explanation_cue": explanation,
            "long_query": long_query,
        },
        "evidence_risk_prior_score": score,
        "evidence_risk_prior_label": label,
    }


def main() -> int:
    if not DATA_PATH.exists():
        print(f"ERROR: Missing source dataset: {DATA_PATH}")
        return 1

    questions = json.loads(DATA_PATH.read_text(encoding="utf-8")).get("questions", [])
    records = [classify(q) for q in questions if isinstance(q, dict)]

    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    labels = Counter(record["evidence_risk_prior_label"] for record in records)
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        by_type[record["question_type"]][record["evidence_risk_prior_label"]] += 1

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "definition": (
            "Transparent query-only evidence-risk prior. It estimates the likelihood that "
            "a minimal retrieval route will be insufficient. It is not a clinical harm score."
        ),
        "uses_gold_evidence_or_answers": False,
        "labels": dict(sorted(labels.items())),
        "labels_by_question_type": {
            qtype: dict(sorted(counter.items()))
            for qtype, counter in sorted(by_type.items())
        },
        "rule_status": (
            "Initial prior only. Route thresholds and predictive usefulness will be tuned "
            "and evaluated on development data before final freezing."
        ),
        "output_relative_path": str(OUTPUT_PATH.relative_to(ROOT)),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Risk-prior labels: {summary['labels']}")
    print(f"Summary saved: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
