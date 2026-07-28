from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from stage7_common import ALLOWED_LABELS, ROOT, load_jsonl

SAMPLE = ROOT / "outputs" / "stage7_sampling" / "stage7_smoke_sample.json"
RESULTS = ROOT / "outputs" / "stage7_smoke" / "stage7_smoke_results.jsonl"
SUMMARY = ROOT / "outputs" / "stage7_smoke" / "stage7_smoke_summary.json"
REPORT = ROOT / "outputs" / "stage7_smoke" / "stage7_smoke_validation_report.json"

CONDITIONS = {
    "bge_text_only",
    "hybrid_text_only",
    "graph_reranked",
    "risk_adaptive_agentic",
}


def main() -> int:
    missing = [
        str(path) for path in (SAMPLE, RESULTS, SUMMARY) if not path.exists()
    ]
    if missing:
        print("ERROR: Missing Stage 7 smoke outputs:")
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
    role_counts = Counter(row["graph_role"] for row in sample)

    by_question = defaultdict(list)
    for row in results:
        by_question[row["question_id"]].append(row)

    every_question_four_conditions = all(
        {row["condition"] for row in rows} == CONDITIONS
        and len(rows) == 4
        for rows in by_question.values()
    )

    verifier_complete = True
    answer_complete = True
    evidence_logged = True
    for row in results:
        if not str(row.get("answer", "")).strip():
            answer_complete = False
        if not row.get("evidence_snippet_ids"):
            evidence_logged = False
        claims = row.get("claims", [])
        verifications = row.get("verifications", [])
        if not row.get("abstain") and not claims:
            answer_complete = False
        if len(claims) != len(verifications):
            verifier_complete = False
        for item in verifications:
            if (
                item.get("status") not in ALLOWED_LABELS
                or not str(item.get("brief_rationale", "")).strip()
                or not isinstance(
                    item.get("material_qualifiers_checked"), list
                )
                or not item.get("material_qualifiers_checked")
            ):
                verifier_complete = False

    graph_role_changed = sum(
        row["condition"] == "graph_reranked"
        and row["graph_role"] == "graph_suitable_candidate"
        and bool(row.get("evidence_changed_vs_hybrid"))
        for row in results
    )

    checks = {
        "smoke_has_8_questions": len(sample) == 8,
        "two_questions_per_question_type": all(
            type_counts[qtype] == 2
            for qtype in ("factoid", "list", "summary", "yesno")
        ),
        "smoke_uncertainty_distribution_3_2_3": (
            uncertainty_counts["low"] == 3
            and uncertainty_counts["medium"] == 2
            and uncertainty_counts["high"] == 3
        ),
        "four_graph_and_four_control_questions": (
            role_counts["graph_suitable_candidate"] == 4
            and role_counts["non_graph_control_candidate"] == 4
        ),
        "32_condition_answers_present": len(results) == 32,
        "every_question_has_four_conditions": every_question_four_conditions,
        "all_answers_and_claims_complete": answer_complete,
        "all_verifier_outputs_semantically_complete": verifier_complete,
        "all_conditions_log_evidence_ids": evidence_logged,
        "graph_changes_evidence_in_at_least_two_graph_cases": (
            graph_role_changed >= 2
        ),
        "sealed_test_not_accessed": summary.get("sealed_test_accessed") is False,
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "counts": {
            "questions": len(sample),
            "answers": len(results),
            "question_type_counts": dict(sorted(type_counts.items())),
            "uncertainty_counts": dict(sorted(uncertainty_counts.items())),
            "graph_role_counts": dict(sorted(role_counts.items())),
            "graph_suitable_evidence_change_count": graph_role_changed,
        },
        "next_gate": (
            "Review smoke results before running the 24-question full "
            "development experiment."
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Stage 7 smoke validation status: {report['status']}")
    for name, passed in checks.items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Report: {REPORT}")
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
