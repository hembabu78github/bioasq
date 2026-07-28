from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from stage7_common import ROOT, load_jsonl

OUTPUT_DIR = ROOT / "outputs" / "stage7_graph_scope_v3"
SAMPLE = OUTPUT_DIR / "stage7_graph_scope_v3_sample.json"
RESULTS = OUTPUT_DIR / "stage7_graph_scope_v3_results.jsonl"
SUMMARY = OUTPUT_DIR / "stage7_graph_scope_v3_summary.json"
REPORT = OUTPUT_DIR / "stage7_graph_scope_v3_validation_report.json"


def main() -> int:
    missing = [
        str(path)
        for path in (SAMPLE, RESULTS, SUMMARY)
        if not path.exists()
    ]
    if missing:
        print("ERROR: Missing Stage 7B V3 outputs:")
        for path in missing:
            print("-", path)
        return 1

    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))[
        "questions"
    ]
    results = load_jsonl(RESULTS)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    type_counts = Counter(row["question_type"] for row in sample)
    eligible = [
        row for row in results if row["graph_route_eligible"]
    ]

    checks = {
        "scope_gate_has_6_questions": len(sample) == 6,
        "three_list_and_three_summary_questions": (
            type_counts["list"] == 3
            and type_counts["summary"] == 3
        ),
        "six_graph_outputs_present": len(results) == 6,
        "every_selector_returns_five_unique_snippets": all(
            len(row["graph_selected_top5"]) == 5
            and len(set(row["graph_selected_top5"])) == 5
            for row in results
        ),
        "zero_coverage_novelty_never_selected": all(
            row["zero_coverage_novel_selected_count"] == 0
            for row in results
        ),
        "at_least_two_questions_have_graph_exclusive_value": (
            len(eligible) >= 2
        ),
        "every_eligible_route_has_useful_novel_evidence": all(
            row["useful_novel_selected_count"] >= 1
            and row["graph_exclusive_relevant_item_count"] >= 1
            and row["evidence_set_changed"]
            for row in eligible
        ),
        "every_eligible_route_has_two_answer_aspects": all(
            row["relevant_answer_aspect_count"] >= 2
            for row in eligible
        ),
        "no_factoid_or_yesno_questions_in_scope_gate": all(
            row["question_type"] in {"list", "summary"}
            for row in results
        ),
        "no_answer_or_verifier_calls_made": (
            summary.get("answer_generation_call_count") == 0
            and summary.get("verification_call_count") == 0
        ),
        "sealed_test_not_accessed": (
            summary.get("sealed_test_accessed") is False
        ),
    }

    status = "pass" if all(checks.values()) else "fail"
    architectural_decision = (
        "Retain GraphRAG as a selective list/summary route and proceed "
        "to a verifier-driven answer smoke."
        if status == "pass"
        else "Do not retain GraphRAG as a central contribution. Simplify "
        "to hybrid retrieval plus verification and selective abstention."
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks": checks,
        "counts": {
            "questions": len(results),
            "eligible_graph_routes": len(eligible),
            "eligible_question_ids": [
                row["stage7_id"] for row in eligible
            ],
        },
        "architectural_decision": architectural_decision,
        "scientific_note": (
            "This is a development technical gate, not a final performance estimate."
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Stage 7B V3 scope validation status: {status}")
    for name, passed in checks.items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Decision: {architectural_decision}")
    print(f"Report: {REPORT}")
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
