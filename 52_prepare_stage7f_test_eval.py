from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "bioasq11" / "training11b.json"
TEST_IDS = ROOT / "data" / "processed" / "stage3" / "test_ids_v2.txt"
CORPUS = ROOT / "data" / "processed" / "stage2" / "candidate_snippets.jsonl"
EMBEDDINGS = (
    ROOT / "models" / "stage3_embeddings"
    / "bge_small_corpus_embeddings.npy"
)
EMBED_IDS = (
    ROOT / "models" / "stage3_embeddings"
    / "bge_small_corpus_ids.json"
)
STAGE7E_DECISION = (
    ROOT / "outputs" / "stage7e_gold_scoring"
    / "stage7e_text_route_decision.json"
)

OUTPUT_DIR = ROOT / "outputs" / "stage7f_test_eval"
PRIVATE_DIR = ROOT / "data" / "processed" / "stage7f"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRIVATE_DIR.mkdir(parents=True, exist_ok=True)

JOBS_JSON = OUTPUT_DIR / "stage7f_test_jobs.json"
JOBS_CSV = OUTPUT_DIR / "stage7f_test_jobs.csv"
BATCH_PLAN = OUTPUT_DIR / "stage7f_test_batch_plan.json"
MANIFEST = OUTPUT_DIR / "stage7f_test_manifest.json"
RANKINGS = PRIVATE_DIR / "test_bge_rankings_private.jsonl"

MODEL_ID = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
GENERATOR_MODEL = "openai/gpt-oss-20b"
PROMPT_VERSION = "stage7d_evidence_closed_v1"
TOP_K = 100
BATCH_SIZE = 6
EXPECTED_TEST_IDS_SHA256 = "925e8029179c57a6f7c3bb6a1c120d90dd3c497a968c1b2a66fe65df5b17c0f3"
EXPECTED_STAGE7E_DECISION_SHA256 = "6a6d420c3c250c5861bbdfb8b9634b59a36114ac3cf8cde3f98d95a104ff7fe8"
EXPECTED_COUNTS = {"total": 706, "factoid": 212, "list": 135, "summary": 169, "yesno": 190}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def top_dense(
    corpus_embeddings: np.ndarray,
    query_embedding: np.ndarray,
) -> tuple[list[int], list[float]]:
    scores = corpus_embeddings @ query_embedding
    k = min(TOP_K, len(scores))
    candidates = np.argpartition(scores, -k)[-k:]
    ordered = candidates[np.argsort(scores[candidates])[::-1]]
    return [int(i) for i in ordered], [float(scores[i]) for i in ordered]


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
        "model": GENERATOR_MODEL,
        "prompt_version": PROMPT_VERSION,
        "route": "bge_text_only",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def main() -> int:
    required = [
        DATA, TEST_IDS, CORPUS, EMBEDDINGS, EMBED_IDS,
        STAGE7E_DECISION,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("ERROR: Missing Stage 7F inputs:")
        for path in missing:
            print("-", path)
        return 1

    if sha256(TEST_IDS) != EXPECTED_TEST_IDS_SHA256:
        raise RuntimeError("The sealed test ID file hash has changed.")
    if sha256(STAGE7E_DECISION) != EXPECTED_STAGE7E_DECISION_SHA256:
        raise RuntimeError("The accepted Stage 7E route decision has changed.")

    route_decision = json.loads(
        STAGE7E_DECISION.read_text(encoding="utf-8")
    )
    if route_decision.get("selected_final_text_route") != "bge_text_only":
        raise RuntimeError("Stage 7F is frozen to the accepted BGE route.")
    if route_decision.get("sealed_test_accessed") is not False:
        raise RuntimeError("Stage 7E unexpectedly reports test access.")

    test_ids = [
        line.strip()
        for line in TEST_IDS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(test_ids) != EXPECTED_COUNTS["total"]:
        raise RuntimeError(
            f"Expected {EXPECTED_COUNTS['total']} test IDs; found {len(test_ids)}."
        )

    questions_payload = json.loads(DATA.read_text(encoding="utf-8"))
    question_map = {
        str(row["id"]): row for row in questions_payload["questions"]
    }
    questions = []
    for index, question_id in enumerate(test_ids, start=1):
        question = question_map.get(question_id)
        if not question:
            raise RuntimeError(f"Missing BioASQ question {question_id}.")
        questions.append(
            {
                "stage7f_id": f"S7T-{index:04d}",
                "question_id": question_id,
                "question_type": str(question["type"]),
                "question": str(question["body"]),
            }
        )

    type_counts = {}
    for row in questions:
        type_counts[row["question_type"]] = (
            type_counts.get(row["question_type"], 0) + 1
        )
    for qtype in ("factoid", "list", "summary", "yesno"):
        if type_counts.get(qtype) != EXPECTED_COUNTS[qtype]:
            raise RuntimeError(
                f"Unexpected {qtype} count: {type_counts.get(qtype)}."
            )

    corpus_rows = load_jsonl(CORPUS)
    corpus_ids = [row["snippet_id"] for row in corpus_rows]
    cached_ids = json.loads(EMBED_IDS.read_text(encoding="utf-8"))
    if cached_ids != corpus_ids:
        raise RuntimeError("Cached BGE corpus IDs do not match the corpus.")

    print(
        f"SEALED TEST ACCESS BEGINS: encoding {len(questions)} "
        "frozen test questions with BGE."
    )
    model = SentenceTransformer(MODEL_ID, device="cpu")
    model.max_seq_length = 256
    query_embeddings = model.encode(
        [QUERY_PREFIX + row["question"] for row in questions],
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    corpus_embeddings = np.load(EMBEDDINGS, mmap_mode="r")

    rankings = []
    jobs = []
    for order, (question, embedding) in enumerate(
        zip(questions, query_embeddings), start=1
    ):
        indices, scores = top_dense(corpus_embeddings, embedding)
        ranked_ids = [corpus_ids[index] for index in indices]
        top5 = ranked_ids[:5]
        if len(top5) != 5 or len(set(top5)) != 5:
            raise RuntimeError(
                f"Invalid top-five evidence for {question['stage7f_id']}."
            )
        signature = job_signature(
            question["question_id"],
            question["question_type"],
            question["question"],
            top5,
        )
        job_id = f"S7FJ-{signature[:16].upper()}"
        batch_index = math.ceil(order / BATCH_SIZE)
        job = {
            "job_id": job_id,
            "job_signature_sha256": signature,
            **question,
            "route_selected": "bge_text_only",
            "evidence_snippet_ids": top5,
            "conditions": ["bge_text_only"],
            "logical_arm_count": 1,
            "job_order": order,
            "batch_index": batch_index,
            "model": GENERATOR_MODEL,
            "prompt_version": PROMPT_VERSION,
        }
        jobs.append(job)
        rankings.append(
            {
                **question,
                "bge_ranked_ids": ranked_ids,
                "bge_scores": scores,
                "evidence_snippet_ids": top5,
                "job_id": job_id,
            }
        )

    with RANKINGS.open("w", encoding="utf-8") as handle:
        for row in rankings:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    jobs_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sealed_test_accessed": True,
        "test_gold_answers_accessed": False,
        "test_gold_relevance_accessed": False,
        "selected_route": "bge_text_only",
        "unique_job_count": len(jobs),
        "jobs": jobs,
    }
    JOBS_JSON.write_text(
        json.dumps(jobs_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fields = [
        "job_id", "stage7f_id", "question_id", "question_type",
        "route_selected", "job_order", "batch_index",
        "evidence_snippet_ids", "job_signature_sha256",
    ]
    with JOBS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in jobs:
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "evidence_snippet_ids": json.dumps(
                        row["evidence_snippet_ids"]
                    ),
                }
            )

    batch_count = math.ceil(len(jobs) / BATCH_SIZE)
    batches = []
    for batch_index in range(1, batch_count + 1):
        batch_jobs = [
            row for row in jobs if row["batch_index"] == batch_index
        ]
        batches.append(
            {
                "batch_index": batch_index,
                "job_count": len(batch_jobs),
                "job_ids": [row["job_id"] for row in batch_jobs],
                "stage7f_ids": [
                    row["stage7f_id"] for row in batch_jobs
                ],
                "maximum_generation_calls": len(batch_jobs),
                "maximum_verifier_calls": len(batch_jobs),
            }
        )
    BATCH_PLAN.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "batch_size": BATCH_SIZE,
                "batch_count": batch_count,
                "batches": batches,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "one_time_sealed_test_evaluation": True,
        "sealed_test_accessed": True,
        "test_gold_answers_accessed": False,
        "test_gold_relevance_accessed": False,
        "test_question_count": len(questions),
        "test_question_type_counts": type_counts,
        "selected_route": "bge_text_only",
        "model": GENERATOR_MODEL,
        "prompt_version": PROMPT_VERSION,
        "evidence_count": 5,
        "job_count": len(jobs),
        "batch_size": BATCH_SIZE,
        "batch_count": batch_count,
        "test_ids_sha256": sha256(TEST_IDS),
        "stage7e_decision_sha256": sha256(STAGE7E_DECISION),
        "corpus_sha256": sha256(CORPUS),
        "embedding_ids_sha256": sha256(EMBED_IDS),
        "rankings_sha256": sha256(RANKINGS),
        "jobs_sha256": sha256(JOBS_JSON),
        "batch_plan_sha256": sha256(BATCH_PLAN),
        "post_access_rule": (
            "No model, route, prompt, verifier, evidence count, threshold or "
            "metric may be changed after this manifest is created."
        ),
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("Stage 7F sealed-test preparation completed.")
    print(f"- Test questions: {len(questions)}")
    print(f"- Jobs: {len(jobs)}")
    print(f"- Batches: {batch_count}")
    print(f"- Manifest: {MANIFEST}")
    print("STOP: upload the manifest and batch plan before API execution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
