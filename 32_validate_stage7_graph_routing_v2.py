from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from stage7_common import ROOT, load_jsonl

OUTPUT_DIR = ROOT / "outputs" / "stage7_graph_routing_v2"
SAMPLE = OUTPUT_DIR / "stage7_graph_routing_v2_sample.json"
RESULTS = OUTPUT_DIR / "stage7_graph_routing_v2_results.jsonl"
SUMMARY = OUTPUT_DIR / "stage7_graph_routing_v2_summary.json"
REPORT = OUTPUT_DIR / "stage7_graph_routing_v2_validation_report.json"


def main() -> int:
    missing = [
        str(path)
        for path in (SAMPLE, RESULTS, SUMMARY)
        if not path.exists()
    ]
    if missing:
        print("ERROR: Missing Stage 7B outputs:")
        for path in missing:
            print("-", path)
        return 1

    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))["questions"]
    results = load_jsonl(RESULTS)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    type_counts = Counter(row["question_type"] for row in sample)
    uncertainty_counts = Counter(
        row["retrieval_uncertainty_label"] for row in sample
    )
    role_counts = Counter(row["structural_role"] for row in sample)

    valid_candidate_ids = all(
        set(row["graph_selected_top5"])
        <= set(row["hybrid_top5"])
        | {
            trace["snippet_id"] for trace in row["selection_trace"]
        }
        for row in results
    )
    five_unique_selected = all(
        len(row["graph_selected_top5"]) == 5
        and len(set(row["graph_selected_top5"])) == 5
        for row in results
    )
    controls_not_graph_routed = all(
        not row["graph_route_eligible"]
        and row["risk_route_selected"] != "graph_coverage"
        for row in results
        if row["structural_role"] == "direct_control"
    )
    insufficient_not_graph_routed = all(
        not row["graph_route_eligible"]
        for row in results
        if not row["graph_sufficient"]
    )
    eligible_changed = all(
        row["evidence_set_changed"]
        and row["novel_selected_count"] >= 1
        for row in results
        if row["graph_route_eligible"]
    )
    eligible_count = sum(
        row["graph_route_eligible"]
        for row in results
        if row["structural_role"] == "graph_stress"
    )

    checks = {
        "routing_stress_has_8_questions": len(sample) == 8,
        "two_questions_per_question_type": all(
            type_counts[qtype] == 2
            for qtype in ("factoid", "list", "summary", "yesno")
        ),
        "uncertainty_distribution_3_2_3": (
            uncertainty_counts["low"] == 3
            and uncertainty_counts["medium"] == 2
            and uncertainty_counts["high"] == 3
        ),
        "four_graph_stress_and_four_controls": (
            role_counts["graph_stress"] == 4
            and role_counts["direct_control"] == 4
        ),
        "eight_graph_outputs_present": len(results) == 8,
        "every_selector_returns_five_unique_snippets": five_unique_selected,
        "selected_ids_are_from_candidate_pool": valid_candidate_ids,
        "at_least_two_graph_stress_routes_are_eligible": eligible_count >= 2,
        "every_eligible_graph_route_changes_evidence": eligible_changed,
        "no_direct_control_is_graph_routed": controls_not_graph_routed,
        "graph_insufficient_cases_are_not_graph_routed": insufficient_not_graph_routed,
        "no_answer_or_verifier_calls_made": (
            summary.get("answer_generation_call_count") == 0
            and summary.get("verification_call_count") == 0
        ),
        "sealed_test_not_accessed": summary.get("sealed_test_accessed") is False,
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "counts": {
            "questions": len(sample),
            "graph_stress_route_eligible_count": eligible_count,
            "risk_route_counts": summary.get("risk_route_counts", {}),
        },
        "next_gate": (
            "Only after this routing-only gate passes should graph-conditioned "
            "answer generation be rerun."
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Stage 7 graph-routing V2 validation status: {report['status']}")
    for name, passed in checks.items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Report: {REPORT}")
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
