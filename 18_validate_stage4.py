from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "stage4"
PRIVATE_DIR = ROOT / "data" / "processed" / "stage4"
REPORT_PATH = OUTPUT_DIR / "stage4_validation_report.json"

DEV_IDS = ROOT / "data" / "processed" / "stage3" / "dev_ids_v2.txt"
TEST_IDS = ROOT / "data" / "processed" / "stage3" / "test_ids_v2.txt"
DIAGNOSTICS_SUMMARY = OUTPUT_DIR / "dev_retrieval_diagnostics_summary.json"
DIAGNOSTICS_PRIVATE = PRIVATE_DIR / "dev_retrieval_diagnostics_private.jsonl"
PILOT_MANIFEST = OUTPUT_DIR / "graph_claim_pilot_manifest.json"
PILOT_RESULTS = OUTPUT_DIR / "graph_claim_pilot_results.jsonl"
PILOT_SUMMARY = OUTPUT_DIR / "graph_claim_pilot_summary.json"
EXAMPLES = OUTPUT_DIR / "graph_claim_examples.jsonl"
BY_TYPE = OUTPUT_DIR / "graph_claim_pilot_by_type.csv"

ALLOWED_STATUS = {"supported", "contradicted", "insufficient_evidence"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    required = [
        DEV_IDS,
        TEST_IDS,
        DIAGNOSTICS_SUMMARY,
        DIAGNOSTICS_PRIVATE,
        PILOT_MANIFEST,
        PILOT_RESULTS,
        PILOT_SUMMARY,
        EXAMPLES,
        BY_TYPE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("ERROR: Missing Stage 4 outputs:")
        for path in missing:
            print(f"- {path}")
        return 1

    dev_ids = {
        line.strip() for line in DEV_IDS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    test_ids = {
        line.strip() for line in TEST_IDS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    diagnostics_summary = json.loads(
        DIAGNOSTICS_SUMMARY.read_text(encoding="utf-8")
    )
    diagnostics = load_jsonl(DIAGNOSTICS_PRIVATE)
    manifest = json.loads(PILOT_MANIFEST.read_text(encoding="utf-8"))
    results = load_jsonl(PILOT_RESULTS)
    summary = json.loads(PILOT_SUMMARY.read_text(encoding="utf-8"))

    selected_ids = {q["question_id"] for q in manifest["selected_questions"]}
    result_ids = {r["question_id"] for r in results}

    invalid_status_count = 0
    missing_edge_provenance = 0
    missing_claim_verification = 0
    for record in results:
        valid_evidence_ids = {item["evidence_id"] for item in record["evidence"]}
        for edge in record["graph"]["relations"]:
            if not edge.get("evidence_ids") or not set(edge["evidence_ids"]).issubset(
                valid_evidence_ids
            ):
                missing_edge_provenance += 1
        for variant in ("text_only", "graph_assisted"):
            claim_ids = {
                claim["claim_id"] for claim in record["answers"][variant]["claims"]
            }
            verification_ids = {
                item["claim_id"]
                for item in record["verifications"][variant]["verifications"]
            }
            missing_claim_verification += len(claim_ids - verification_ids)
            invalid_status_count += sum(
                item["status"] not in ALLOWED_STATUS
                for item in record["verifications"][variant]["verifications"]
            )

    type_counts = Counter(q["question_type"] for q in manifest["selected_questions"])

    checks = {
        "development_diagnostics_cover_all_709_questions": len(diagnostics) == 709
        and diagnostics_summary.get("development_question_count") == 709,
        "diagnostics_report_no_test_access": diagnostics_summary.get(
            "test_questions_accessed"
        )
        is False,
        "pilot_has_12_questions": len(selected_ids) == 12,
        "pilot_has_three_questions_per_type": set(type_counts.values()) == {3}
        and len(type_counts) == 4,
        "all_pilot_questions_are_in_development": selected_ids.issubset(dev_ids),
        "no_pilot_question_is_in_test": selected_ids.isdisjoint(test_ids),
        "all_selected_questions_have_results": selected_ids == result_ids,
        "all_expected_groq_calls_completed": summary.get("completed_call_count") == 36,
        "all_graph_edges_have_evidence_provenance": missing_edge_provenance == 0,
        "every_claim_has_a_verification": missing_claim_verification == 0,
        "all_verification_statuses_valid": invalid_status_count == 0,
        "pilot_summary_states_manual_audit_is_required": "manual audit"
        in json.dumps(summary).lower(),
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "counts": {
            "development_diagnostics": len(diagnostics),
            "pilot_questions": len(results),
            "question_types": dict(sorted(type_counts.items())),
            "missing_edge_provenance": missing_edge_provenance,
            "missing_claim_verification": missing_claim_verification,
            "invalid_verification_statuses": invalid_status_count,
        },
        "scientific_note": (
            "Passing Stage 4 validates feasibility, provenance and schema integrity. It does "
            "not establish claim-verification accuracy or clinical validity. A blinded manual "
            "audit and larger development evaluation are required before the final test run."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Stage 4 validation status: {report['status']}")
    for name, passed in checks.items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Report saved: {REPORT_PATH}")
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
