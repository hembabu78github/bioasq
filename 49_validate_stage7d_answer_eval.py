from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

from stage7d_common import ALLOWED_LABELS, ROOT, load_jsonl

OUTPUT_DIR = ROOT / "outputs" / "stage7d_answer_eval"
RESULTS = OUTPUT_DIR / "stage7d_answer_eval_results.jsonl"
SUMMARY = OUTPUT_DIR / "stage7d_answer_eval_summary.json"
PAIRS = OUTPUT_DIR / "stage7d_answer_eval_paired_comparison.json"
MANIFEST = OUTPUT_DIR / "stage7d_answer_eval_manifest.json"
ROUTE_RESULTS = (
    ROOT / "outputs" / "stage7d_route_freeze"
    / "stage7d_route_results.jsonl"
)
REPORT = OUTPUT_DIR / "stage7d_answer_eval_validation_report.json"

ACCEPTED_ROUTE_SHA256 = "1227d2b07bdb4b5ced07b6b818495fc855678bea3a1266ba300fa490c17fa070"
ACCEPTED_GRAPH_ROUTES = ["S7Q-009", "S7Q-010", "S7Q-011"]


def main() -> int:
    missing = [
        str(path)
        for path in (
            RESULTS,
            SUMMARY,
            PAIRS,
            MANIFEST,
            ROUTE_RESULTS,
        )
        if not path.exists()
    ]
    if missing:
        print("ERROR: Missing Stage 7D-B final outputs:")
        for path in missing:
            print("-", path)
        return 1

    results = load_jsonl(RESULTS)
    summary = json.loads(
        SUMMARY.read_text(encoding="utf-8")
    )
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))
    manifest = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )
    route_rows = load_jsonl(ROUTE_RESULTS)
    route_by_stage = {
        row["stage7_id"]: row for row in route_rows
    }

    grouped = defaultdict(dict)
    for row in results:
        grouped[row["stage7_id"]][row["condition"]] = row

    graph_pairs = [
        row for row in results
        if row["condition"] == "risk_adaptive"
        and row["route_selected"] == "graph_selected_text_only"
    ]
    hybrid_adaptive_identical = []
    for stage7_id, conditions in grouped.items():
        hybrid = conditions.get("hybrid_text_only")
        adaptive = conditions.get("risk_adaptive")
        if hybrid and adaptive:
            hybrid_adaptive_identical.append(
                (
                    stage7_id,
                    hybrid["shared_job_id"]
                    == adaptive["shared_job_id"],
                    hybrid["evidence_snippet_ids"]
                    == adaptive["evidence_snippet_ids"],
                )
            )

    evidence_closed = True
    complete = True
    disposition_valid = True
    for row in results:
        claims = row.get("claims", [])
        verifications = row.get("verifications", [])
        if not str(row.get("answer", "")).strip():
            complete = False
        if len(claims) != len(verifications):
            complete = False
        if row.get("graph_supplied_to_generator"):
            evidence_closed = False
        if row.get("graph_supplied_to_verifier"):
            evidence_closed = False

        labels = []
        for item in verifications:
            labels.append(item.get("status"))
            if (
                item.get("status") not in ALLOWED_LABELS
                or not str(
                    item.get("brief_rationale", "")
                ).strip()
                or not isinstance(
                    item.get("material_qualifiers_checked"),
                    list,
                )
                or not item.get(
                    "material_qualifiers_checked"
                )
            ):
                complete = False
            if (
                not isinstance(item.get("evidence_ids"), list)
                or not set(item.get("evidence_ids", []))
                <= {"E1", "E2", "E3", "E4", "E5"}
                or item.get("graph_edge_ids")
            ):
                evidence_closed = False

        should_release = bool(
            not row["generator_abstained"]
            and claims
            and labels
            and all(label == "supported" for label in labels)
        )
        if (
            row["final_disposition"] == "release"
        ) != should_release:
            disposition_valid = False

    route_hash_ok = (
        hashlib.sha256(
            ROUTE_RESULTS.read_bytes()
        ).hexdigest()
        == ACCEPTED_ROUTE_SHA256
    )
    route_graph_ids = [
        row["stage7_id"]
        for row in route_rows
        if row["route_selected"] == "graph_selected_text_only"
    ]

    checks = {
        "accepted_route_hash_is_unchanged": route_hash_ok,
        "twenty_four_questions_present": len(grouped) == 24,
        "seventy_two_logical_arms_present": len(results) == 72,
        "twenty_four_arms_per_condition": all(
            sum(row["condition"] == condition for row in results) == 24
            for condition in (
                "bge_text_only",
                "hybrid_text_only",
                "risk_adaptive",
            )
        ),
        "three_frozen_graph_routes_preserved": (
            route_graph_ids == ACCEPTED_GRAPH_ROUTES
            and sorted(
                row["stage7_id"] for row in graph_pairs
            )
            == sorted(ACCEPTED_GRAPH_ROUTES)
        ),
        "non_graph_adaptive_arms_reuse_hybrid_jobs": all(
            same_job and same_evidence
            for stage7_id, same_job, same_evidence
            in hybrid_adaptive_identical
            if stage7_id not in ACCEPTED_GRAPH_ROUTES
        ),
        "graph_adaptive_arms_use_changed_jobs_and_evidence": all(
            not same_job and not same_evidence
            for stage7_id, same_job, same_evidence
            in hybrid_adaptive_identical
            if stage7_id in ACCEPTED_GRAPH_ROUTES
        ),
        "all_answers_and_verifications_complete": complete,
        "all_verifier_citations_are_displayed_E1_to_E5": (
            evidence_closed
        ),
        "graph_is_never_supplied_downstream": (
            evidence_closed
            and summary.get("graph_payload_supplied_count") == 0
        ),
        "final_disposition_matches_frozen_rule": (
            disposition_valid
        ),
        "no_non_supported_claim_is_released": (
            summary.get("unsupported_claims_released") == 0
        ),
        "paired_file_contains_twenty_four_questions": (
            pairs.get("question_count") == 24
        ),
        "logical_deduplication_is_recorded": (
            summary.get("unique_execution_job_count", 0)
            <= 72
            and summary.get("deduplicated_logical_arm_count", 0)
            == 72 - summary.get("unique_execution_job_count", 0)
        ),
        "sealed_test_not_accessed": (
            summary.get("sealed_test_accessed") is False
        ),
    }

    status = "pass" if all(checks.values()) else "fail"
    report = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": status,
        "checks": checks,
        "counts": {
            "questions": len(grouped),
            "logical_arms": len(results),
            "unique_execution_jobs": summary.get(
                "unique_execution_job_count"
            ),
            "graph_routes": len(graph_pairs),
            "final_releases": sum(
                row["final_disposition"] == "release"
                for row in results
            ),
            "final_abstentions": sum(
                row["final_disposition"] == "abstain"
                for row in results
            ),
        },
        "next_gate": (
            "Perform a stratified blinded human audit of graph-routed pairs, "
            "changed answers, releases, abstentions and verifier disagreements. "
            "Do not access the sealed test yet."
        ),
    }
    REPORT.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"Stage 7D-B final validation status: {status}")
    for name, passed in checks.items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Report: {REPORT}")
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
