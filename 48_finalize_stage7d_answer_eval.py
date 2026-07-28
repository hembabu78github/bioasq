from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage7d_common import ROOT, load_checkpoint, write_jsonl

OUTPUT_DIR = ROOT / "outputs" / "stage7d_answer_eval"
LOGICAL_ARMS_JSON = OUTPUT_DIR / "stage7d_logical_arms.json"
JOBS_JSON = OUTPUT_DIR / "stage7d_execution_jobs.json"
MANIFEST = OUTPUT_DIR / "stage7d_answer_eval_manifest.json"
COMPLETED_JOB_DIR = (
    OUTPUT_DIR / "checkpoints" / "completed_jobs"
)

RESULTS_JSONL = OUTPUT_DIR / "stage7d_answer_eval_results.jsonl"
RESULTS_CSV = OUTPUT_DIR / "stage7d_answer_eval_results.csv"
SUMMARY = OUTPUT_DIR / "stage7d_answer_eval_summary.json"
PAIRS = OUTPUT_DIR / "stage7d_answer_eval_paired_comparison.json"


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fields = [
        "arm_id",
        "stage7_id",
        "question_type",
        "condition",
        "route_selected",
        "shared_job_id",
        "shared_job_logical_arm_count",
        "generator_abstained",
        "claim_count",
        "supported_claim_count",
        "contradicted_claim_count",
        "insufficient_claim_count",
        "final_disposition",
        "answer",
        "evidence_snippet_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            counts = Counter(
                item["status"] for item in row["verifications"]
            )
            writer.writerow(
                {
                    "arm_id": row["arm_id"],
                    "stage7_id": row["stage7_id"],
                    "question_type": row["question_type"],
                    "condition": row["condition"],
                    "route_selected": row["route_selected"],
                    "shared_job_id": row["shared_job_id"],
                    "shared_job_logical_arm_count": row[
                        "shared_job_logical_arm_count"
                    ],
                    "generator_abstained": row[
                        "generator_abstained"
                    ],
                    "claim_count": len(row["claims"]),
                    "supported_claim_count": counts["supported"],
                    "contradicted_claim_count": counts["contradicted"],
                    "insufficient_claim_count": counts[
                        "insufficient_evidence"
                    ],
                    "final_disposition": row[
                        "final_disposition"
                    ],
                    "answer": row["answer"],
                    "evidence_snippet_ids": json.dumps(
                        row["evidence_snippet_ids"]
                    ),
                }
            )


def main() -> int:
    missing = [
        str(path)
        for path in (LOGICAL_ARMS_JSON, JOBS_JSON, MANIFEST)
        if not path.exists()
    ]
    if missing:
        print("ERROR: Missing Stage 7D-B preparation outputs:")
        for path in missing:
            print("-", path)
        return 1

    arms = json.loads(
        LOGICAL_ARMS_JSON.read_text(encoding="utf-8")
    )["arms"]
    jobs = json.loads(
        JOBS_JSON.read_text(encoding="utf-8")
    )["jobs"]
    manifest = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    completed_by_job = {}
    missing_jobs = []
    for job in jobs:
        path = COMPLETED_JOB_DIR / f"{job['job_id']}.json"
        result = load_checkpoint(path)
        if not result:
            missing_jobs.append(job["job_id"])
            continue
        if (
            result.get("job_id") != job["job_id"]
            or result.get("evidence_snippet_ids")
            != job["evidence_snippet_ids"]
        ):
            raise RuntimeError(
                f"Completed job mismatch for {job['job_id']}."
            )
        completed_by_job[job["job_id"]] = result

    if missing_jobs:
        print(
            f"ERROR: {len(missing_jobs)} execution jobs remain incomplete."
        )
        print("First missing jobs:", ", ".join(missing_jobs[:10]))
        return 2

    logical_results = []
    for arm in arms:
        job_result = completed_by_job[arm["job_id"]]
        logical_results.append(
            {
                "expanded_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                **arm,
                "shared_job_id": arm["job_id"],
                "shared_job_logical_arm_count": job_result[
                    "logical_arm_count"
                ],
                "model": job_result["model"],
                "evidence": job_result["evidence"],
                "graph_supplied_to_generator": False,
                "graph_supplied_to_verifier": False,
                "generator_abstained": job_result[
                    "generator_abstained"
                ],
                "generator_abstention_reason": job_result[
                    "generator_abstention_reason"
                ],
                "answer": job_result["answer"],
                "claims": job_result["claims"],
                "verifications": job_result["verifications"],
                "final_disposition": job_result[
                    "final_disposition"
                ],
                "disposition_reason": job_result[
                    "disposition_reason"
                ],
                "final_answer": job_result["final_answer"],
                "answer_call": job_result.get("answer_call"),
                "verification_call": job_result.get(
                    "verification_call"
                ),
            }
        )

    logical_results.sort(
        key=lambda row: (
            row["stage7_id"],
            row["condition_order"],
        )
    )
    write_jsonl(RESULTS_JSONL, logical_results)
    write_csv(RESULTS_CSV, logical_results)

    by_condition = {}
    for condition in (
        "bge_text_only",
        "hybrid_text_only",
        "risk_adaptive",
    ):
        rows = [
            row for row in logical_results
            if row["condition"] == condition
        ]
        labels = [
            item["status"]
            for row in rows
            for item in row["verifications"]
        ]
        by_condition[condition] = {
            "answer_count": len(rows),
            "claim_count": len(labels),
            "verifier_label_counts": dict(
                sorted(Counter(labels).items())
            ),
            "final_release_count": sum(
                row["final_disposition"] == "release"
                for row in rows
            ),
            "final_abstention_count": sum(
                row["final_disposition"] == "abstain"
                for row in rows
            ),
            "mean_claim_count": (
                statistics.mean(
                    len(row["claims"]) for row in rows
                )
                if rows
                else 0
            ),
        }

    grouped = defaultdict(dict)
    for row in logical_results:
        grouped[row["stage7_id"]][row["condition"]] = row

    paired_rows = []
    for stage7_id, conditions in sorted(grouped.items()):
        bge = conditions["bge_text_only"]
        hybrid = conditions["hybrid_text_only"]
        adaptive = conditions["risk_adaptive"]
        paired_rows.append(
            {
                "stage7_id": stage7_id,
                "question_type": hybrid["question_type"],
                "question": hybrid["question"],
                "adaptive_route_selected": adaptive[
                    "route_selected"
                ],
                "bge_hybrid_same_job": (
                    bge["shared_job_id"]
                    == hybrid["shared_job_id"]
                ),
                "hybrid_adaptive_same_job": (
                    hybrid["shared_job_id"]
                    == adaptive["shared_job_id"]
                ),
                "bge_final_disposition": bge[
                    "final_disposition"
                ],
                "hybrid_final_disposition": hybrid[
                    "final_disposition"
                ],
                "adaptive_final_disposition": adaptive[
                    "final_disposition"
                ],
                "bge_answer": bge["answer"],
                "hybrid_answer": hybrid["answer"],
                "adaptive_answer": adaptive["answer"],
                "hybrid_adaptive_answer_changed": (
                    hybrid["answer"] != adaptive["answer"]
                ),
                "hybrid_adaptive_evidence_changed": (
                    hybrid["evidence_snippet_ids"]
                    != adaptive["evidence_snippet_ids"]
                ),
            }
        )

    PAIRS.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "question_count": len(paired_rows),
                "graph_routed_pair_count": sum(
                    row["adaptive_route_selected"]
                    == "graph_selected_text_only"
                    for row in paired_rows
                ),
                "pairs": paired_rows,
                "scientific_note": (
                    "Automated same-model verifier comparisons are descriptive "
                    "only. A stratified blinded human audit is required."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    unique_answer_calls = sum(
        bool(row.get("answer_call"))
        for row in completed_by_job.values()
    )
    unique_verifier_calls = sum(
        bool(row.get("verification_call"))
        for row in completed_by_job.values()
    )

    summary = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "development_only": True,
        "sealed_test_accessed": False,
        "question_count": len(grouped),
        "logical_answer_count": len(logical_results),
        "unique_execution_job_count": len(completed_by_job),
        "deduplicated_logical_arm_count": (
            len(logical_results) - len(completed_by_job)
        ),
        "unique_answer_call_record_count": unique_answer_calls,
        "unique_verifier_call_record_count": unique_verifier_calls,
        "by_condition": by_condition,
        "graph_routed_stage7_ids": manifest[
            "graph_route_stage7_ids"
        ],
        "graph_payload_supplied_count": 0,
        "unsupported_claims_released": sum(
            row["final_disposition"] == "release"
            and any(
                item["status"] != "supported"
                for item in row["verifications"]
            )
            for row in logical_results
        ),
        "final_disposition_policy": (
            "Release only when every atomic claim is verifier-supported; "
            "otherwise abstain."
        ),
        "scientific_note": (
            "Stage 7D-B is a development evaluation. Automated verifier "
            "outputs are not human gold and do not justify an efficacy claim."
        ),
    }
    SUMMARY.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Final logical results: {RESULTS_JSONL}")
    print(f"Summary: {SUMMARY}")
    print(f"Paired comparison: {PAIRS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
