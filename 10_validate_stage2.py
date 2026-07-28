from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed" / "stage2"
OUTPUT_DIR = ROOT / "outputs" / "stage2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = DATA_DIR / "candidate_snippets.jsonl"
GOLD_PATH = DATA_DIR / "gold_relevance.jsonl"
RISK_PATH = DATA_DIR / "query_evidence_risk_prior.jsonl"
SPLIT_MANIFEST = OUTPUT_DIR / "split_manifest.json"
BM25_SUMMARY = OUTPUT_DIR / "bm25_pilot_summary.json"
REPORT_PATH = OUTPUT_DIR / "stage2_validation_report.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    required = [CORPUS_PATH, GOLD_PATH, RISK_PATH, SPLIT_MANIFEST, BM25_SUMMARY]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("ERROR: Missing files:")
        for path in missing:
            print(f"- {path}")
        return 1

    corpus = read_jsonl(CORPUS_PATH)
    gold = read_jsonl(GOLD_PATH)
    risk = read_jsonl(RISK_PATH)
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    bm25 = json.loads(BM25_SUMMARY.read_text(encoding="utf-8"))

    corpus_ids = [row.get("snippet_id") for row in corpus]
    corpus_id_set = set(corpus_ids)
    question_ids_in_public_corpus = sum(
        1 for row in corpus if "question_id" in row or "question_ids" in row
    )
    duplicate_corpus_ids = len(corpus_ids) - len(corpus_id_set)

    missing_relevant_ids = 0
    relevance_question_ids: list[str] = []
    for row in gold:
        relevance_question_ids.append(str(row.get("question_id", "")))
        missing_relevant_ids += sum(
            1
            for snippet_id in row.get("relevant_snippet_ids", [])
            if snippet_id not in corpus_id_set
        )

    risk_question_ids = [str(row.get("question_id", "")) for row in risk]
    risk_gold_fields = sum(
        1
        for row in risk
        if any(
            field in row
            for field in (
                "exact_answer",
                "ideal_answer",
                "documents",
                "snippets",
                "gold_relevant_count",
            )
        )
    )

    split_counts = split.get("counts", {})
    total_split = sum(split_counts.values())
    split_overlap = sum(split.get("id_overlap_counts", {}).values())
    duplicate_body_leakage = split.get("duplicate_body_group_leakage_count", -1)

    checks = {
        "public_corpus_has_no_question_ids": question_ids_in_public_corpus == 0,
        "candidate_snippet_ids_unique": duplicate_corpus_ids == 0,
        "all_gold_relevant_ids_exist_in_candidate_corpus": missing_relevant_ids == 0,
        "risk_prior_contains_no_gold_fields": risk_gold_fields == 0,
        "risk_prior_covers_all_relevance_questions": set(risk_question_ids)
        == set(relevance_question_ids),
        "split_has_no_id_overlap": split_overlap == 0,
        "duplicate_question_bodies_do_not_cross_splits": duplicate_body_leakage == 0,
        "split_covers_all_questions": total_split == len(relevance_question_ids),
        "bm25_pilot_completed": bm25.get("pilot_question_count") == 80,
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "counts": {
            "candidate_snippets": len(corpus),
            "gold_relevance_questions": len(relevance_question_ids),
            "risk_prior_questions": len(risk_question_ids),
            "split_total": total_split,
            "public_corpus_question_id_fields": question_ids_in_public_corpus,
            "duplicate_candidate_ids": duplicate_corpus_ids,
            "missing_gold_snippet_ids": missing_relevant_ids,
            "risk_rows_with_gold_fields": risk_gold_fields,
        },
        "scientific_note": (
            "Passing this report validates structural leakage controls for the closed "
            "snippet-bank benchmark. It does not prove that the benchmark represents "
            "full PubMed retrieval or real clinical deployment."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Stage 2 validation status: {report['status']}")
    for name, passed in checks.items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Report saved: {REPORT_PATH}")
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
