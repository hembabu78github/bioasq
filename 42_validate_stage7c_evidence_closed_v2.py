from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone

from stage7_common import ALLOWED_LABELS, ROOT, load_jsonl

OUTPUT_DIR = ROOT / "outputs" / "stage7c_evidence_closed_v2"
SAMPLE = OUTPUT_DIR / "stage7c_evidence_closed_v2_sample.json"
RESULTS = OUTPUT_DIR / "stage7c_evidence_closed_v2_results.jsonl"
SUMMARY = OUTPUT_DIR / "stage7c_evidence_closed_v2_summary.json"
PAIRS = OUTPUT_DIR / "stage7c_evidence_closed_v2_paired_comparison.json"
REPORT = OUTPUT_DIR / "stage7c_evidence_closed_v2_validation_report.json"


def main() -> int:
    missing = [
        str(path)
        for path in (SAMPLE, RESULTS, SUMMARY, PAIRS)
        if not path.exists()
    ]
    if missing:
        print("ERROR: Missing Stage 7C V2 outputs:")
        for path in missing:
            print("-", path)
        return 1

    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    results = load_jsonl(RESULTS)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))

    arms = sample["arms"]
    expected = {arm["arm_id"] for arm in arms}
    received = {row["arm_id"] for row in results}
    allowed_evidence_ids = {"E1", "E2", "E3", "E4", "E5"}

    complete = True
    evidence_closed = True
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
                <= allowed_evidence_ids
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

    graph_rows = [
        row
        for row in results
        if row["route"] == "graph_selected_text_only"
    ]
    grouped = defaultdict(dict)
    for row in results:
        grouped[row["stage7_id"]][row["route"]] = row

    checks = {
        "six_questions_and_eight_arms": (
            len({arm["stage7_id"] for arm in arms}) == 6
            and len(arms) == 8
        ),
        "six_hybrid_results_reused": (
            sum(row["reuse_v1_result"] for row in results) == 6
        ),
        "two_graph_selected_text_answers_regenerated": (
            len(graph_rows) == 2
            and all(not row["reuse_v1_result"] for row in graph_rows)
        ),
        "all_expected_arms_completed": (
            expected == received and len(results) == len(arms)
        ),
        "graph_route_is_eligible_list_only": all(
            row["question_type"] == "list"
            and row["graph_route_eligible"]
            for row in graph_rows
        ),
        "graph_is_used_only_for_evidence_selection": (
            summary.get("graph_payload_supplied_count") == 0
            and evidence_closed
        ),
        "all_verifier_citations_are_displayed_E1_to_E5": (
            evidence_closed
        ),
        "all_answers_and_verifications_complete": complete,
        "final_disposition_matches_frozen_rule": (
            disposition_valid
        ),
        "no_non_supported_claim_is_released": (
            summary.get("unsupported_claims_released") == 0
        ),
        "paired_comparison_has_two_questions": (
            pairs.get("paired_question_count") == 2
        ),
        "sealed_test_not_accessed": (
            summary.get("sealed_test_accessed") is False
        ),
    }

    status = "pass" if all(checks.values()) else "fail"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks": checks,
        "counts": {
            "questions": len({row["stage7_id"] for row in results}),
            "answers": len(results),
            "new_generation_calls_expected": 2,
            "new_verifier_calls_expected": 2,
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
            "Only after evidence-closed results are reviewed should blinded "
            "human annotation begin."
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Stage 7C evidence-closed V2 validation status: {status}")
    for name, passed in checks.items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Report: {REPORT}")
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
