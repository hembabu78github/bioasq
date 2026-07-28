from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage7f_common import ALLOWED_LABELS, load_checkpoint, load_jsonl
from stage7f_scoring import score_answer

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "stage7f_test_eval"
PRIVATE_DIR = ROOT / "data" / "processed" / "stage7f"

JOBS_JSON = OUTPUT_DIR / "stage7f_test_jobs.json"
MANIFEST = OUTPUT_DIR / "stage7f_test_manifest.json"
COMPLETED_DIR = OUTPUT_DIR / "checkpoints" / "completed_jobs"
RANKINGS = PRIVATE_DIR / "test_bge_rankings_private.jsonl"

DATA = ROOT / "data" / "raw" / "bioasq11" / "training11b.json"
TEST_IDS = ROOT / "data" / "processed" / "stage3" / "test_ids_v2.txt"
GOLD_RELEVANCE = (
    ROOT / "data" / "processed" / "stage2" / "gold_relevance.jsonl"
)

RESULTS_JSONL = OUTPUT_DIR / "stage7f_test_results.jsonl"
RESULTS_CSV = OUTPUT_DIR / "stage7f_test_results.csv"
SUMMARY = OUTPUT_DIR / "stage7f_test_summary.json"
RETRIEVAL_SUMMARY = OUTPUT_DIR / "stage7f_test_retrieval_summary.json"
FINAL_MANIFEST = OUTPUT_DIR / "stage7f_test_final_manifest.json"

EXPECTED_TEST_IDS_SHA256 = "925e8029179c57a6f7c3bb6a1c120d90dd3c497a968c1b2a66fe65df5b17c0f3"
EXPECTED_STAGE7E_DECISION_SHA256 = "6a6d420c3c250c5861bbdfb8b9634b59a36114ac3cf8cde3f98d95a104ff7fe8"
EXPECTED_COUNTS = {"total": 706, "factoid": 212, "list": 135, "summary": 169, "yesno": 190}
TYPE_ORDER = ("factoid", "list", "summary", "yesno")
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20260726


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def retrieval_metrics(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    first = None
    for rank, snippet_id in enumerate(ranked[:20], start=1):
        if snippet_id in relevant:
            first = rank
            break
    return {
        "hit_at_1": float(any(item in relevant for item in ranked[:1])),
        "hit_at_5": float(any(item in relevant for item in ranked[:5])),
        "hit_at_10": float(any(item in relevant for item in ranked[:10])),
        "hit_at_20": float(any(item in relevant for item in ranked[:20])),
        "reciprocal_rank_at_20": 0.0 if first is None else 1.0 / first,
    }


def stratified_bootstrap(rows: list[dict[str, Any]]) -> tuple[float, float]:
    grouped = {
        qtype: [row["deployed_primary_score"] for row in rows if row["question_type"] == qtype]
        for qtype in TYPE_ORDER
    }
    rng = random.Random(BOOTSTRAP_SEED)
    composites = []
    for _ in range(BOOTSTRAP_SAMPLES):
        type_means = []
        for qtype in TYPE_ORDER:
            values = grouped[qtype]
            sample = [values[rng.randrange(len(values))] for _ in values]
            type_means.append(mean(sample))
        composites.append(mean(type_means))
    composites.sort()
    return (
        composites[int(0.025 * BOOTSTRAP_SAMPLES)],
        composites[int(0.975 * BOOTSTRAP_SAMPLES)],
    )


def valid_completed(value: dict[str, Any], job: dict[str, Any]) -> bool:
    if not value:
        return False
    if value.get("job_id") != job["job_id"]:
        return False
    if value.get("evidence_snippet_ids") != job["evidence_snippet_ids"]:
        return False
    if value.get("graph_supplied_to_generator") is not False:
        return False
    if value.get("graph_supplied_to_verifier") is not False:
        return False
    claims = value.get("claims", [])
    verifications = value.get("verifications", [])
    if len(claims) != len(verifications):
        return False
    allowed = {"E1", "E2", "E3", "E4", "E5"}
    for item in verifications:
        if item.get("status") not in ALLOWED_LABELS:
            return False
        if not str(item.get("brief_rationale", "")).strip():
            return False
        if not set(item.get("evidence_ids", [])) <= allowed:
            return False
        if item.get("graph_edge_ids"):
            return False
    expected_release = (
        not value.get("generator_abstained")
        and bool(claims)
        and bool(verifications)
        and all(item["status"] == "supported" for item in verifications)
    )
    return (
        value.get("final_disposition")
        == ("release" if expected_release else "abstain")
    )


def main() -> int:
    required = [
        JOBS_JSON, MANIFEST, RANKINGS, DATA,
        TEST_IDS, GOLD_RELEVANCE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("ERROR: Missing Stage 7F finalization inputs:")
        for path in missing:
            print("-", path)
        return 1

    if sha256(TEST_IDS) != EXPECTED_TEST_IDS_SHA256:
        raise RuntimeError("The sealed test ID hash changed.")

    jobs_payload = json.loads(JOBS_JSON.read_text(encoding="utf-8"))
    jobs = jobs_payload["jobs"]
    if len(jobs) != EXPECTED_COUNTS["total"]:
        raise RuntimeError("Unexpected Stage 7F job count.")

    completed = []
    missing_jobs = []
    for job in jobs:
        path = COMPLETED_DIR / f"{job['job_id']}.json"
        value = load_checkpoint(path)
        if not value or not valid_completed(value, job):
            missing_jobs.append(job["job_id"])
        else:
            completed.append(value)
    if missing_jobs:
        print(
            f"ERROR: {len(missing_jobs)} jobs are missing or incomplete. "
            "Rerun the all-batches command."
        )
        for job_id in missing_jobs[:20]:
            print("-", job_id)
        return 2

    question_payload = json.loads(DATA.read_text(encoding="utf-8"))
    gold_by_id = {
        str(row["id"]): row for row in question_payload["questions"]
    }
    relevance = {
        row["question_id"]: set(row["relevant_snippet_ids"])
        for row in load_jsonl(GOLD_RELEVANCE)
    }
    rankings = {
        row["question_id"]: row for row in load_jsonl(RANKINGS)
    }

    completed.sort(key=lambda row: row["job_order"])
    result_rows = []
    retrieval_rows = []
    for result in completed:
        gold = gold_by_id[result["question_id"]]
        raw_answer = result["answer"]
        deployed_answer = (
            raw_answer
            if result["final_disposition"] == "release"
            else ""
        )
        deployed = score_answer(
            result["question_type"],
            deployed_answer,
            gold.get("exact_answer"),
            gold.get("ideal_answer"),
        )
        raw = score_answer(
            result["question_type"],
            raw_answer,
            gold.get("exact_answer"),
            gold.get("ideal_answer"),
        )
        ranking = rankings[result["question_id"]]
        retrieval = retrieval_metrics(
            ranking["bge_ranked_ids"],
            relevance[result["question_id"]],
        )
        retrieval_rows.append(
            {
                "stage7f_id": result["stage7f_id"],
                "question_id": result["question_id"],
                "question_type": result["question_type"],
                **retrieval,
            }
        )
        result_rows.append(
            {
                **result,
                "gold_exact_answer": gold.get("exact_answer"),
                "gold_ideal_answer": gold.get("ideal_answer"),
                "deployed_answer": deployed_answer,
                "deployed_primary_score": deployed["primary_score"],
                "deployed_strict_score": deployed["strict_score"],
                "raw_primary_score": raw["primary_score"],
                "raw_strict_score": raw["strict_score"],
                "retrieval": retrieval,
            }
        )

    with RESULTS_JSONL.open("w", encoding="utf-8") as handle:
        for row in result_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_fields = [
        "stage7f_id", "question_id", "question_type",
        "final_disposition", "answer", "deployed_primary_score",
        "deployed_strict_score", "raw_primary_score",
        "raw_strict_score", "hit_at_1", "hit_at_5",
        "hit_at_10", "hit_at_20", "reciprocal_rank_at_20",
    ]
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in result_rows:
            writer.writerow(
                {
                    "stage7f_id": row["stage7f_id"],
                    "question_id": row["question_id"],
                    "question_type": row["question_type"],
                    "final_disposition": row["final_disposition"],
                    "answer": row["answer"],
                    "deployed_primary_score": row["deployed_primary_score"],
                    "deployed_strict_score": row["deployed_strict_score"],
                    "raw_primary_score": row["raw_primary_score"],
                    "raw_strict_score": row["raw_strict_score"],
                    **row["retrieval"],
                }
            )

    by_type = {}
    for qtype in TYPE_ORDER:
        subset = [row for row in result_rows if row["question_type"] == qtype]
        by_type[qtype] = {
            "question_count": len(subset),
            "release_count": sum(
                row["final_disposition"] == "release" for row in subset
            ),
            "coverage": mean([
                1.0 if row["final_disposition"] == "release" else 0.0
                for row in subset
            ]),
            "deployed_primary_mean": mean([
                row["deployed_primary_score"] for row in subset
            ]),
            "deployed_strict_mean": mean([
                row["deployed_strict_score"] for row in subset
            ]),
            "raw_primary_mean": mean([
                row["raw_primary_score"] for row in subset
            ]),
            "raw_strict_mean": mean([
                row["raw_strict_score"] for row in subset
            ]),
        }

    deployed_composite = mean([
        by_type[qtype]["deployed_primary_mean"] for qtype in TYPE_ORDER
    ])
    strict_composite = mean([
        by_type[qtype]["deployed_strict_mean"] for qtype in TYPE_ORDER
    ])
    raw_composite = mean([
        by_type[qtype]["raw_primary_mean"] for qtype in TYPE_ORDER
    ])
    ci_lower, ci_upper = stratified_bootstrap(result_rows)

    labels = [
        item["status"]
        for row in result_rows
        for item in row["verifications"]
    ]
    answer_call_records = {
        row["answer_call"]["call_name"]
        for row in result_rows
        if row.get("answer_call")
    }
    verifier_call_records = {
        row["verification_call"]["call_name"]
        for row in result_rows
        if row.get("verification_call")
    }

    retrieval_by_type = {}
    for qtype in TYPE_ORDER:
        subset = [
            row for row in retrieval_rows if row["question_type"] == qtype
        ]
        retrieval_by_type[qtype] = {
            "question_count": len(subset),
            "hit_at_1": mean([row["hit_at_1"] for row in subset]),
            "hit_at_5": mean([row["hit_at_5"] for row in subset]),
            "hit_at_10": mean([row["hit_at_10"] for row in subset]),
            "hit_at_20": mean([row["hit_at_20"] for row in subset]),
            "mrr_at_20": mean([
                row["reciprocal_rank_at_20"] for row in subset
            ]),
        }
    retrieval_overall = {
        "question_count": len(retrieval_rows),
        "hit_at_1": mean([row["hit_at_1"] for row in retrieval_rows]),
        "hit_at_5": mean([row["hit_at_5"] for row in retrieval_rows]),
        "hit_at_10": mean([row["hit_at_10"] for row in retrieval_rows]),
        "hit_at_20": mean([row["hit_at_20"] for row in retrieval_rows]),
        "mrr_at_20": mean([
            row["reciprocal_rank_at_20"] for row in retrieval_rows
        ]),
    }
    retrieval_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sealed_test_accessed": True,
        "test_gold_relevance_accessed": True,
        "route": "bge_text_only",
        "overall": retrieval_overall,
        "by_question_type": retrieval_by_type,
    }
    RETRIEVAL_SUMMARY.write_text(
        json.dumps(retrieval_summary, indent=2),
        encoding="utf-8",
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "one_time_sealed_test_evaluation": True,
        "sealed_test_accessed": True,
        "test_gold_answers_accessed": True,
        "test_gold_relevance_accessed": True,
        "selected_route": "bge_text_only",
        "question_count": len(result_rows),
        "question_type_counts": dict(
            sorted(Counter(row["question_type"] for row in result_rows).items())
        ),
        "answer_call_record_count": len(answer_call_records),
        "verifier_call_record_count": len(verifier_call_records),
        "claim_count": len(labels),
        "verifier_label_counts": dict(sorted(Counter(labels).items())),
        "release_count": sum(
            row["final_disposition"] == "release" for row in result_rows
        ),
        "abstention_count": sum(
            row["final_disposition"] == "abstain" for row in result_rows
        ),
        "unsupported_claims_released": sum(
            row["final_disposition"] == "release"
            and any(
                item["status"] != "supported"
                for item in row["verifications"]
            )
            for row in result_rows
        ),
        "by_question_type": by_type,
        "deployed_primary_macro_composite": deployed_composite,
        "deployed_strict_macro_composite": strict_composite,
        "raw_primary_macro_composite": raw_composite,
        "deployed_primary_macro_bootstrap_95_ci": [
            ci_lower, ci_upper
        ],
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "retrieval_summary_path": str(RETRIEVAL_SUMMARY),
        "scientific_note": (
            "These are the single frozen held-out test results. No tuning, "
            "route changes or prompt changes are permitted after this run."
        ),
    }
    SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    final_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "one_time_sealed_test_evaluation": True,
        "selected_route": "bge_text_only",
        "test_ids_sha256": sha256(TEST_IDS),
        "jobs_sha256": sha256(JOBS_JSON),
        "rankings_sha256": sha256(RANKINGS),
        "results_sha256": sha256(RESULTS_JSONL),
        "summary_sha256": sha256(SUMMARY),
        "retrieval_summary_sha256": sha256(RETRIEVAL_SUMMARY),
        "test_question_count": len(result_rows),
        "no_post_test_tuning": True,
    }
    FINAL_MANIFEST.write_text(
        json.dumps(final_manifest, indent=2), encoding="utf-8"
    )

    print("Stage 7F sealed-test finalization completed.")
    print(f"- Questions: {len(result_rows)}")
    print(f"- Releases: {summary['release_count']}")
    print(f"- Abstentions: {summary['abstention_count']}")
    print(
        "- Deployed primary macro composite: "
        f"{deployed_composite:.6f}"
    )
    print(f"- Summary: {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
