from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "stage7e_gold_scoring"

RESULTS_PATH = (
    ROOT / "outputs" / "stage7d_answer_eval"
    / "stage7d_answer_eval_results.jsonl"
)
PER_QUESTION = OUTPUT_DIR / "stage7e_gold_per_question.jsonl"
SUMMARY = OUTPUT_DIR / "stage7e_gold_summary.json"
DECISION = OUTPUT_DIR / "stage7e_text_route_decision.json"
REPORT = OUTPUT_DIR / "stage7e_gold_validation_report.json"

ACCEPTED_RESULTS_SHA256 = "d28008c3cec4ef81bd6e2e98a0f7708f9c9606dd65c7607dd487e1ad8f972a66"
CONDITIONS = {"bge_text_only", "hybrid_text_only"}


def load_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    missing = [
        str(path)
        for path in (RESULTS_PATH, PER_QUESTION, SUMMARY, DECISION)
        if not path.exists()
    ]
    if missing:
        print("ERROR: Missing Stage 7E outputs:")
        for path in missing:
            print("-", path)
        return 1

    rows = load_jsonl(PER_QUESTION)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    decision = json.loads(DECISION.read_text(encoding="utf-8"))

    question_ids = {row["stage7_id"] for row in rows}
    type_counts = Counter(
        (row["condition"], row["question_type"]) for row in rows
    )

    score_fields = [
        "deployed_primary_score",
        "deployed_strict_score",
        "raw_primary_score",
        "raw_strict_score",
    ]
    score_range_ok = all(
        0.0 <= float(row[field]) <= 1.0
        for row in rows
        for field in score_fields
    )

    ci = summary["paired_comparison"]["bootstrap_95_ci"]

    checks = {
        "accepted_stage7d_results_hash_is_unchanged": (
            hashlib.sha256(RESULTS_PATH.read_bytes()).hexdigest()
            == ACCEPTED_RESULTS_SHA256
        ),
        "twenty_four_development_questions_present": (
            len(question_ids) == 24
        ),
        "forty_eight_text_only_arms_present": len(rows) == 48,
        "only_bge_and_hybrid_are_scored": (
            {row["condition"] for row in rows} == CONDITIONS
        ),
        "six_questions_per_type_per_condition": all(
            type_counts[(condition, question_type)] == 6
            for condition in CONDITIONS
            for question_type in (
                "factoid", "list", "summary", "yesno"
            )
        ),
        "all_gold_answers_found": all(
            row.get("gold_exact_answer") is not None
            or row.get("gold_ideal_answer") is not None
            for row in rows
        ),
        "all_scores_are_between_zero_and_one": score_range_ok,
        "four_type_balanced_composites_present": all(
            set(summary["conditions"][condition]["by_type"])
            == {"factoid", "list", "summary", "yesno"}
            for condition in CONDITIONS
        ),
        "paired_bootstrap_is_recorded": (
            len(ci) == 2
            and 0 <= summary["paired_comparison"][
                "bootstrap_hybrid_win_probability"
            ] <= 1
        ),
        "route_selection_uses_frozen_rule": (
            decision["selected_final_text_route"] in CONDITIONS
            and (
                (
                    ci[0] > 0
                    and decision["selected_final_text_route"]
                    == "hybrid_text_only"
                )
                or (
                    ci[1] < 0
                    and decision["selected_final_text_route"]
                    == "bge_text_only"
                )
                or (
                    ci[0] <= 0 <= ci[1]
                    and decision["selected_final_text_route"]
                    == "bge_text_only"
                )
            )
        ),
        "graph_route_remains_excluded": (
            decision["graph_route_status"]
            == "excluded_from_main_architecture"
        ),
        "sealed_test_not_accessed": (
            summary.get("sealed_test_accessed") is False
            and decision.get("sealed_test_accessed") is False
        ),
    }

    status = "pass" if all(checks.values()) else "fail"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks": checks,
        "counts": {
            "questions": len(question_ids),
            "logical_arms": len(rows),
            "selected_route": decision[
                "selected_final_text_route"
            ],
        },
        "next_gate": (
            "Upload the four Stage 7E outputs for review. Do not "
            "prepare or access the sealed test until the text-only "
            "route decision is accepted and hash-frozen."
        ),
    }
    REPORT.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"Stage 7E validation status: {status}")
    for name, passed in checks.items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Report: {REPORT}")
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
