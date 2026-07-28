from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage7d_common import MODEL, ROOT, load_jsonl

ROUTE_RESULTS = (
    ROOT / "outputs" / "stage7d_route_freeze"
    / "stage7d_route_results.jsonl"
)
RANKINGS = (
    ROOT / "data" / "processed" / "stage4"
    / "dev_retrieval_rankings_private.jsonl"
)
CORPUS = (
    ROOT / "data" / "processed" / "stage2"
    / "candidate_snippets.jsonl"
)

OUTPUT_DIR = ROOT / "outputs" / "stage7d_answer_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOGICAL_ARMS_JSON = OUTPUT_DIR / "stage7d_logical_arms.json"
LOGICAL_ARMS_CSV = OUTPUT_DIR / "stage7d_logical_arms.csv"
JOBS_JSON = OUTPUT_DIR / "stage7d_execution_jobs.json"
JOBS_CSV = OUTPUT_DIR / "stage7d_execution_jobs.csv"
BATCH_PLAN = OUTPUT_DIR / "stage7d_batch_plan.json"
MANIFEST = OUTPUT_DIR / "stage7d_answer_eval_manifest.json"

ACCEPTED_ROUTE_SHA256 = "1227d2b07bdb4b5ced07b6b818495fc855678bea3a1266ba300fa490c17fa070"
ACCEPTED_GRAPH_ROUTES = ["S7Q-009", "S7Q-010", "S7Q-011"]
BATCH_SIZE = 6
PROMPT_VERSION = "stage7d_evidence_closed_v1"


def top_unique(
    ranked_ids: list[str],
    corpus_ids: set[str],
    count: int = 5,
) -> list[str]:
    output = []
    for snippet_id in ranked_ids:
        if snippet_id in corpus_ids and snippet_id not in output:
            output.append(snippet_id)
        if len(output) >= count:
            break
    if len(output) != count:
        raise RuntimeError(
            f"Expected {count} valid unique snippets, found {len(output)}."
        )
    return output


def job_signature(
    question_id: str,
    question_type: str,
    question: str,
    evidence_ids: list[str],
) -> str:
    payload = {
        "question_id": question_id,
        "question_type": question_type,
        "question": question,
        "evidence_snippet_ids": evidence_ids,
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        json.dumps(row[field], ensure_ascii=False)
                        if isinstance(row.get(field), list)
                        else row.get(field)
                    )
                    for field in fields
                }
            )


def main() -> int:
    missing = [
        str(path)
        for path in (ROUTE_RESULTS, RANKINGS, CORPUS)
        if not path.exists()
    ]
    if missing:
        print("ERROR: Missing Stage 7D-B inputs:")
        for path in missing:
            print("-", path)
        return 1

    actual_route_hash = hashlib.sha256(
        ROUTE_RESULTS.read_bytes()
    ).hexdigest()
    if actual_route_hash != ACCEPTED_ROUTE_SHA256:
        raise RuntimeError(
            "The local route manifest does not match the accepted hash. "
            "Do not regenerate or alter Stage 7D-A routes."
        )

    routes = load_jsonl(ROUTE_RESULTS)
    routes.sort(key=lambda row: row["stage7_id"])
    if len(routes) != 24:
        raise RuntimeError(
            f"Expected 24 frozen route rows, found {len(routes)}."
        )

    graph_route_ids = [
        row["stage7_id"]
        for row in routes
        if row["route_selected"] == "graph_selected_text_only"
    ]
    if graph_route_ids != ACCEPTED_GRAPH_ROUTES:
        raise RuntimeError(
            f"Graph route IDs changed: {graph_route_ids}."
        )

    rankings = {
        row["question_id"]: row for row in load_jsonl(RANKINGS)
    }
    corpus_rows = load_jsonl(CORPUS)
    corpus_ids = {row["snippet_id"] for row in corpus_rows}

    logical_arms = []
    jobs_by_signature: dict[str, dict[str, Any]] = {}
    condition_order = {
        "bge_text_only": 1,
        "hybrid_text_only": 2,
        "risk_adaptive": 3,
    }

    for route in routes:
        qid = route["question_id"]
        ranking = rankings.get(qid)
        if not ranking:
            raise RuntimeError(f"Missing retrieval rankings for {qid}.")

        bge_top5 = top_unique(
            ranking["bge_ranked_ids"], corpus_ids
        )
        hybrid_top5 = top_unique(
            ranking["hybrid_bge_ranked_ids"], corpus_ids
        )
        if hybrid_top5 != route["hybrid_top5"]:
            raise RuntimeError(
                f"Hybrid evidence changed for {route['stage7_id']}."
            )

        adaptive_top5 = route[
            "selected_evidence_snippet_ids"
        ]
        if len(adaptive_top5) != 5 or len(set(adaptive_top5)) != 5:
            raise RuntimeError(
                f"Invalid adaptive evidence for {route['stage7_id']}."
            )
        if not set(adaptive_top5) <= corpus_ids:
            raise RuntimeError(
                f"Unknown adaptive evidence ID for {route['stage7_id']}."
            )

        arm_specs = [
            (
                "bge_text_only",
                bge_top5,
                "bge_text_only",
            ),
            (
                "hybrid_text_only",
                hybrid_top5,
                "hybrid_text_only",
            ),
            (
                "risk_adaptive",
                adaptive_top5,
                route["route_selected"],
            ),
        ]

        for condition, evidence_ids, route_selected in arm_specs:
            signature = job_signature(
                qid,
                route["question_type"],
                route["question"],
                evidence_ids,
            )
            job_id = f"S7DJ-{signature[:16].upper()}"
            arm_id = (
                f"S7D-{route['stage7_id']}-"
                f"{condition.upper().replace('_TEXT_ONLY', '').replace('_', '')}"
            )

            arm = {
                "arm_id": arm_id,
                "stage7_id": route["stage7_id"],
                "question_id": qid,
                "question_type": route["question_type"],
                "question": route["question"],
                "condition": condition,
                "condition_order": condition_order[condition],
                "route_selected": route_selected,
                "graph_route_eligible": route[
                    "graph_route_eligible"
                ],
                "retrieval_uncertainty_label": route[
                    "retrieval_uncertainty_label"
                ],
                "evidence_snippet_ids": evidence_ids,
                "job_id": job_id,
                "job_signature_sha256": signature,
                "graph_payload_supplied": False,
            }
            logical_arms.append(arm)

            if signature not in jobs_by_signature:
                jobs_by_signature[signature] = {
                    "job_id": job_id,
                    "job_signature_sha256": signature,
                    "stage7_id": route["stage7_id"],
                    "question_id": qid,
                    "question_type": route["question_type"],
                    "question": route["question"],
                    "evidence_snippet_ids": evidence_ids,
                    "model": MODEL,
                    "prompt_version": PROMPT_VERSION,
                    "logical_arm_ids": [],
                    "conditions": [],
                }
            jobs_by_signature[signature]["logical_arm_ids"].append(
                arm_id
            )
            jobs_by_signature[signature]["conditions"].append(
                condition
            )

    logical_arms.sort(
        key=lambda row: (
            row["stage7_id"],
            row["condition_order"],
        )
    )
    jobs = list(jobs_by_signature.values())
    stage_order = {
        row["stage7_id"]: index
        for index, row in enumerate(routes, start=1)
    }
    jobs.sort(
        key=lambda row: (
            stage_order[row["stage7_id"]],
            min(
                condition_order[condition]
                for condition in row["conditions"]
            ),
            row["job_id"],
        )
    )

    for index, job in enumerate(jobs, start=1):
        job["job_order"] = index
        job["batch_index"] = math.ceil(index / BATCH_SIZE)
        job["logical_arm_count"] = len(job["logical_arm_ids"])

    batch_count = math.ceil(len(jobs) / BATCH_SIZE)
    batches = []
    for batch_index in range(1, batch_count + 1):
        batch_jobs = [
            job for job in jobs
            if job["batch_index"] == batch_index
        ]
        batches.append(
            {
                "batch_index": batch_index,
                "job_count": len(batch_jobs),
                "job_ids": [job["job_id"] for job in batch_jobs],
                "stage7_ids": sorted(
                    {job["stage7_id"] for job in batch_jobs}
                ),
                "maximum_generation_calls": len(batch_jobs),
                "maximum_verifier_calls": len(batch_jobs),
            }
        )

    LOGICAL_ARMS_JSON.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "logical_arm_count": len(logical_arms),
                "arms": logical_arms,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_csv(
        LOGICAL_ARMS_CSV,
        logical_arms,
        [
            "arm_id",
            "stage7_id",
            "question_type",
            "condition",
            "route_selected",
            "graph_route_eligible",
            "job_id",
            "evidence_snippet_ids",
        ],
    )

    JOBS_JSON.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "unique_job_count": len(jobs),
                "jobs": jobs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_csv(
        JOBS_CSV,
        jobs,
        [
            "job_order",
            "batch_index",
            "job_id",
            "stage7_id",
            "question_type",
            "conditions",
            "logical_arm_count",
            "logical_arm_ids",
            "evidence_snippet_ids",
        ],
    )

    BATCH_PLAN.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "batch_size": BATCH_SIZE,
                "batch_count": batch_count,
                "batches": batches,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    condition_counts = Counter(
        row["condition"] for row in logical_arms
    )
    reused_logical_arms = len(logical_arms) - len(jobs)
    manifest = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "development_only": True,
        "sealed_test_accessed": False,
        "accepted_route_sha256": ACCEPTED_ROUTE_SHA256,
        "logical_question_count": 24,
        "logical_arm_count": len(logical_arms),
        "condition_counts": dict(
            sorted(condition_counts.items())
        ),
        "unique_execution_job_count": len(jobs),
        "deduplicated_logical_arm_count": reused_logical_arms,
        "maximum_generation_calls": len(jobs),
        "maximum_verifier_calls": len(jobs),
        "batch_size": BATCH_SIZE,
        "batch_count": batch_count,
        "graph_route_stage7_ids": graph_route_ids,
        "graph_payload_supplied_count": 0,
        "prompt_version": PROMPT_VERSION,
        "model": MODEL,
        "scientific_note": (
            "Seventy-two logical arms are retained. Exact question-and-evidence "
            "duplicates share one deterministic execution job, avoiding "
            "unnecessary API calls without changing the comparison."
        ),
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("Stage 7D-B answer batches prepared.")
    print(f"- Logical arms: {len(logical_arms)}")
    print(f"- Unique execution jobs: {len(jobs)}")
    print(f"- Deduplicated logical arms: {reused_logical_arms}")
    print(f"- Batches: {batch_count}")
    print(f"- Jobs per batch: at most {BATCH_SIZE}")
    print(f"- Manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
