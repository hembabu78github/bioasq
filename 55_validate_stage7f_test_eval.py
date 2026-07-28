from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "stage7f_test_eval"
TEST_IDS = ROOT / "data" / "processed" / "stage3" / "test_ids_v2.txt"

RESULTS = OUTPUT_DIR / "stage7f_test_results.jsonl"
SUMMARY = OUTPUT_DIR / "stage7f_test_summary.json"
RETRIEVAL = OUTPUT_DIR / "stage7f_test_retrieval_summary.json"
FINAL_MANIFEST = OUTPUT_DIR / "stage7f_test_final_manifest.json"
REPORT = OUTPUT_DIR / "stage7f_test_validation_report.json"

EXPECTED_TEST_IDS_SHA256 = "925e8029179c57a6f7c3bb6a1c120d90dd3c497a968c1b2a66fe65df5b17c0f3"
EXPECTED_COUNTS = {"total": 706, "factoid": 212, "list": 135, "summary": 169, "yesno": 190}


def load_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    required = [RESULTS, SUMMARY, RETRIEVAL, FINAL_MANIFEST, TEST_IDS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("ERROR: Missing Stage 7F outputs:")
        for path in missing:
            print("-", path)
        return 1

    rows = load_jsonl(RESULTS)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    retrieval = json.loads(RETRIEVAL.read_text(encoding="utf-8"))
    manifest = json.loads(FINAL_MANIFEST.read_text(encoding="utf-8"))

    type_counts = Counter(row["question_type"] for row in rows)
    allowed_evidence = {"E1", "E2", "E3", "E4", "E5"}
    score_fields = [
        "deployed_primary_score", "deployed_strict_score",
        "raw_primary_score", "raw_strict_score",
    ]

    citations_valid = all(
        set(item.get("evidence_ids", [])) <= allowed_evidence
        and not item.get("graph_edge_ids")
        for row in rows
        for item in row["verifications"]
    )
    disposition_valid = all(
        row["final_disposition"]
        == (
            "release"
            if (
                not row["generator_abstained"]
                and row["claims"]
                and row["verifications"]
                and all(
                    item["status"] == "supported"
                    for item in row["verifications"]
                )
            )
            else "abstain"
        )
        for row in rows
    )

    checks = {
        "sealed_test_id_hash_is_unchanged": (
            hashlib.sha256(TEST_IDS.read_bytes()).hexdigest()
            == EXPECTED_TEST_IDS_SHA256
        ),
        "seven_hundred_six_test_questions_present": (
            len(rows) == EXPECTED_COUNTS["total"]
        ),
        "test_question_type_counts_match_frozen_split": all(
            type_counts[qtype] == EXPECTED_COUNTS[qtype]
            for qtype in ("factoid", "list", "summary", "yesno")
        ),
        "only_frozen_bge_route_is_present": all(
            row["route_selected"] == "bge_text_only"
            for row in rows
        ),
        "five_unique_evidence_snippets_per_question": all(
            len(row["evidence_snippet_ids"]) == 5
            and len(set(row["evidence_snippet_ids"])) == 5
            for row in rows
        ),
        "all_answers_and_verifications_complete": all(
            bool(str(row["answer"]).strip())
            and len(row["claims"]) == len(row["verifications"])
            for row in rows
        ),
        "all_verifier_citations_are_displayed_E1_to_E5": citations_valid,
        "graph_is_never_supplied_downstream": all(
            row["graph_supplied_to_generator"] is False
            and row["graph_supplied_to_verifier"] is False
            for row in rows
        ),
        "final_disposition_matches_frozen_rule": disposition_valid,
        "no_non_supported_claim_is_released": (
            summary["unsupported_claims_released"] == 0
        ),
        "all_gold_scores_are_between_zero_and_one": all(
            0.0 <= float(row[field]) <= 1.0
            for row in rows for field in score_fields
        ),
        "retrieval_metrics_are_complete": (
            retrieval["overall"]["question_count"]
            == EXPECTED_COUNTS["total"]
            and set(retrieval["by_question_type"])
            == {"factoid", "list", "summary", "yesno"}
        ),
        "single_frozen_test_manifest_is_recorded": (
            manifest["one_time_sealed_test_evaluation"] is True
            and manifest["selected_route"] == "bge_text_only"
            and manifest["no_post_test_tuning"] is True
        ),
        "test_gold_access_is_recorded_only_at_finalization": (
            summary["test_gold_answers_accessed"] is True
            and summary["test_gold_relevance_accessed"] is True
        ),
    }

    status = "pass" if all(checks.values()) else "fail"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks": checks,
        "counts": {
            "questions": len(rows),
            "question_type_counts": dict(sorted(type_counts.items())),
            "releases": summary["release_count"],
            "abstentions": summary["abstention_count"],
        },
        "final_rule": (
            "The sealed test is complete. Do not retune or rerun based on "
            "these results. Proceed to manuscript analysis and reporting."
        ),
    }
    REPORT.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Stage 7F validation status: {status}")
    for name, passed in checks.items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Report: {REPORT}")
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
