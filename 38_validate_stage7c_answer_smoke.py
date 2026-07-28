from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

from stage7_common import ALLOWED_LABELS, ROOT, load_jsonl

OUTPUT_DIR = ROOT / "outputs" / "stage7c_selective_answer_smoke"
SAMPLE = OUTPUT_DIR / "stage7c_answer_smoke_sample.json"
RESULTS = OUTPUT_DIR / "stage7c_answer_smoke_results.jsonl"
SUMMARY = OUTPUT_DIR / "stage7c_answer_smoke_summary.json"
PAIRS = OUTPUT_DIR / "stage7c_paired_comparison.json"
REPORT = OUTPUT_DIR / "stage7c_answer_smoke_validation_report.json"


def main() -> int:
    missing = [
        str(path)
        for path in (SAMPLE, RESULTS, SUMMARY, PAIRS)
        if not path.exists()
    ]
    if missing:
        print("ERROR: Missing Stage 7C outputs:")
        for path in missing:
            print("-", path)
        return 1

    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    results = load_jsonl(RESULTS)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))

    arms = sample["arms"]
    expected_arm_ids = {arm["arm_id"] for arm in arms}
    result_arm_ids = {row["arm_id"] for row in results}

    graph_rows = [
        row for row in results if row["route"] == "selective_graph"
    ]
    hybrid_rows = [
        row for row in results if row["route"] == "hybrid_baseline"
    ]

    grouped = defaultdict(dict)
    for row in results:
        grouped[row["stage7_id"]][row["route"]] = row

    complete = True
    deterministic_disposition = True
    no_unsupported_release = True
    for row in results:
        if not str(row.get("answer", "")).strip():
            complete = False
        claims = row.get("claims", [])
        verifications = row.get("verifications", [])
        if not row.get("generator_abstained") and not claims:
            complete = False
        if len(claims) != len(verifications):
            complete = False
        for item in verifications:
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

        labels = [
            item["status"] for item in verifications
        ]
        should_release = bool(
            not row["generator_abstained"]
            and claims
            and labels
            and all(label == "supported" for label in labels)
        )
        actual_release = (
            row["final_disposition"] == "release"
        )
        if should_release != actual_release:
            deterministic_disposition = False
        if actual_release and any(
            label != "supported" for label in labels
        ):
            no_unsupported_release = False

    graph_pairs_changed = all(
        set(grouped[row["stage7_id"]]["hybrid_baseline"][
            "evidence_snippet_ids"
        ])
        != set(row["evidence_snippet_ids"])
        for row in graph_rows
    )

    checks = {
        "sample_has_6_questions": (
            len({arm["stage7_id"] for arm in arms}) == 6
        ),
        "sample_has_8_answer_arms": len(arms) == 8,
        "six_hybrid_and_two_graph_answers": (
            len(hybrid_rows) == 6 and len(graph_rows) == 2
        ),
        "all_expected_arms_completed": (
            expected_arm_ids == result_arm_ids
            and len(results) == len(arms)
        ),
        "graph_route_is_list_only": all(
            row["question_type"] == "list"
            and row["graph_route_eligible"]
            for row in graph_rows
        ),
        "no_summary_question_uses_graph": all(
            not (
                row["question_type"] == "summary"
                and row["route"] == "selective_graph"
            )
            for row in results
        ),
        "graph_pairs_use_changed_evidence": graph_pairs_changed,
        "all_answers_and_verifications_complete": complete,
        "final_disposition_matches_frozen_rule": (
            deterministic_disposition
        ),
        "no_non_supported_claim_is_released": (
            no_unsupported_release
            and summary.get("unsupported_claims_released") == 0
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
            "questions": len(
                {row["stage7_id"] for row in results}
            ),
            "answers": len(results),
            "hybrid_answers": len(hybrid_rows),
            "graph_answers": len(graph_rows),
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
            "Review the two paired graph-versus-hybrid answers and complete "
            "the blinded human claim audit before any efficacy claim."
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Stage 7C answer-smoke validation status: {status}")
    for name, passed in checks.items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Report: {REPORT}")
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
