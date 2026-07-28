from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from stage7d_common import ROOT, load_jsonl

OUTPUT_DIR = ROOT / "outputs" / "stage7d_route_freeze"
SAMPLE = OUTPUT_DIR / "stage7d_route_sample.json"
RESULTS = OUTPUT_DIR / "stage7d_route_results.jsonl"
SUMMARY = OUTPUT_DIR / "stage7d_route_summary.json"
REPORT = OUTPUT_DIR / "stage7d_route_validation_report.json"


def main() -> int:
    missing = [
        str(path)
        for path in (SAMPLE, RESULTS, SUMMARY)
        if not path.exists()
    ]
    if missing:
        print("ERROR: Missing Stage 7D route outputs:")
        for path in missing:
            print("-", path)
        return 1

    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))[
        "questions"
    ]
    results = load_jsonl(RESULTS)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    type_counts = Counter(
        row["question_type"] for row in results
    )
    list_rows = [
        row for row in results
        if row["question_type"] == "list"
    ]
    graph_rows = [
        row for row in results
        if row["route_selected"]
        == "graph_selected_text_only"
    ]
    non_list_rows = [
        row for row in results
        if row["question_type"] != "list"
    ]

    checks = {
        "balanced_24_question_sample_present": (
            len(sample) == 24 and len(results) == 24
        ),
        "six_questions_per_type": all(
            type_counts[qtype] == 6
            for qtype in ("factoid", "list", "summary", "yesno")
        ),
        "six_list_questions_received_route_evaluation": (
            len(list_rows) == 6
            and all(row.get("graph") for row in list_rows)
        ),
        "all_non_list_questions_use_hybrid": all(
            row["route_selected"] == "hybrid_text_only"
            and not row["graph_route_eligible"]
            for row in non_list_rows
        ),
        "graph_route_is_list_only": all(
            row["question_type"] == "list"
            and row["graph_route_eligible"]
            for row in graph_rows
        ),
        "at_least_two_graph_routes_are_frozen": (
            len(graph_rows) >= 2
        ),
        "every_graph_route_changes_evidence": all(
            row["evidence_set_changed"]
            for row in graph_rows
        ),
        "every_graph_route_has_exclusive_value": all(
            row["graph_exclusive_relevant_item_count"] >= 1
            and row["useful_novel_selected_count"] >= 1
            for row in graph_rows
        ),
        "zero_coverage_novelty_never_creates_route": all(
            row["zero_coverage_novel_selected_count"] == 0
            for row in graph_rows
        ),
        "every_question_has_five_unique_selected_snippets": all(
            len(row["selected_evidence_snippet_ids"]) == 5
            and len(set(row["selected_evidence_snippet_ids"])) == 5
            for row in results
        ),
        "graph_is_marked_hidden_from_downstream": all(
            row["graph_hidden_from_downstream"] is True
            for row in results
        ),
        "no_answer_or_verifier_calls_made": (
            summary.get("answer_generation_call_count") == 0
            and summary.get("verification_call_count") == 0
        ),
        "no_graph_payload_supplied_downstream": (
            summary.get(
                "graph_payload_supplied_to_downstream_count"
            ) == 0
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
            "questions": len(results),
            "list_questions": len(list_rows),
            "graph_routes": len(graph_rows),
            "graph_route_stage7_ids": [
                row["stage7_id"] for row in graph_rows
            ],
            "route_counts": summary.get("route_counts", {}),
        },
        "next_gate": (
            "Upload the four route-freeze files for review. Do not start "
            "the 72-arm answer evaluation until the route manifest is accepted."
        ),
    }
    REPORT.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"Stage 7D route-freeze validation status: {status}")
    for name, passed in checks.items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Report: {REPORT}")
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
