from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from huggingface_hub import model_info
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
PILOT_PATH = ROOT / "data" / "processed" / "pilot" / "bioasq11_pilot_80.json"
CORPUS_PATH = ROOT / "data" / "processed" / "stage2" / "candidate_snippets.jsonl"
GOLD_PATH = ROOT / "data" / "processed" / "stage2" / "gold_relevance.jsonl"
RISK_PATH = ROOT / "data" / "processed" / "stage2" / "query_evidence_risk_prior.jsonl"
CACHE_DIR = ROOT / "models" / "stage3_embeddings"
OUTPUT_DIR = ROOT / "outputs" / "stage3"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "dense_pilot_results.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "dense_models_summary.json"
BY_TYPE_PATH = OUTPUT_DIR / "dense_pilot_by_model_type.csv"

TOP_K = 100
EVAL_K = 20
MAX_SEQ_LENGTH = 256

MODELS = [
    {
        "key": "bge_small",
        "model_id": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "batch_size": 64,
    },
    {
        "key": "pubmedbert",
        "model_id": "NeuML/pubmedbert-base-embeddings",
        "query_prefix": "",
        "batch_size": 16,
    },
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "question_count": len(rows),
        "hit_rate_at_1": mean([r["hit_at_1"] for r in rows]),
        "hit_rate_at_5": mean([r["hit_at_5"] for r in rows]),
        "hit_rate_at_10": mean([r["hit_at_10"] for r in rows]),
        "hit_rate_at_20": mean([r["hit_at_20"] for r in rows]),
        "mrr_at_20": mean([r["reciprocal_rank_at_20"] for r in rows]),
        "mean_query_encode_ms": mean([r["query_encode_ms"] for r in rows]),
        "mean_search_ms": mean([r["search_ms"] for r in rows]),
    }


def top_k_scores(corpus_embeddings: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    scores = corpus_embeddings @ query
    k = min(k, scores.shape[0])
    candidate_indices = np.argpartition(scores, -k)[-k:]
    ordered = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
    return ordered, scores[ordered]


def main() -> int:
    required = [PILOT_PATH, CORPUS_PATH, GOLD_PATH, RISK_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("ERROR: Missing Stage 2/Pilot files:")
        for path in missing:
            print(f"- {path}")
        return 1

    pilot_questions = json.loads(PILOT_PATH.read_text(encoding="utf-8")).get("questions", [])
    corpus = load_jsonl(CORPUS_PATH)
    gold_rows = load_jsonl(GOLD_PATH)
    risk_rows = load_jsonl(RISK_PATH)

    corpus_ids = [row["snippet_id"] for row in corpus]
    corpus_texts = [row["text"] for row in corpus]
    gold = {
        row["question_id"]: set(row["relevant_snippet_ids"])
        for row in gold_rows
    }
    risk = {row["question_id"]: row for row in risk_rows}

    all_results: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}

    # If a previous complete results file exists, rebuild it model-by-model to avoid duplicates.
    if RESULTS_PATH.exists():
        RESULTS_PATH.unlink()

    for spec in MODELS:
        model_key = spec["key"]
        model_id = spec["model_id"]
        embedding_path = CACHE_DIR / f"{model_key}_corpus_embeddings.npy"
        id_path = CACHE_DIR / f"{model_key}_corpus_ids.json"
        model_manifest_path = CACHE_DIR / f"{model_key}_manifest.json"

        print()
        print("=" * 72)
        print(f"Loading model: {model_id}")
        load_started = time.perf_counter()
        model = SentenceTransformer(model_id, device="cpu")
        model.max_seq_length = MAX_SEQ_LENGTH
        model_load_ms = round((time.perf_counter() - load_started) * 1000, 2)

        try:
            info = model_info(model_id)
            revision = getattr(info, "sha", None)
        except Exception as exc:
            revision = None
            print(f"Warning: model revision lookup failed: {exc}")

        if embedding_path.exists() and id_path.exists():
            cached_ids = json.loads(id_path.read_text(encoding="utf-8"))
            if cached_ids != corpus_ids:
                print("Cached corpus IDs do not match current corpus. Re-encoding.")
                embedding_path.unlink()
                id_path.unlink()

        encode_ms = None
        if not embedding_path.exists():
            print(f"Encoding {len(corpus_texts)} snippets with batch size {spec['batch_size']}...")
            started = time.perf_counter()
            embeddings = model.encode(
                corpus_texts,
                batch_size=spec["batch_size"],
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            encode_ms = round((time.perf_counter() - started) * 1000, 2)
            embeddings = embeddings.astype(np.float32, copy=False)
            np.save(embedding_path, embeddings)
            id_path.write_text(json.dumps(corpus_ids), encoding="utf-8")
            del embeddings
            gc.collect()
        else:
            print(f"Using cached embeddings: {embedding_path}")

        corpus_embeddings = np.load(embedding_path, mmap_mode="r")
        embedding_dim = int(corpus_embeddings.shape[1])

        model_results: list[dict[str, Any]] = []

        for q in pilot_questions:
            qid = str(q.get("id", ""))
            query_text = spec["query_prefix"] + str(q.get("body", ""))

            started = time.perf_counter()
            query_embedding = model.encode(
                [query_text],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0].astype(np.float32, copy=False)
            query_encode_ms = round((time.perf_counter() - started) * 1000, 2)

            started = time.perf_counter()
            indices, scores = top_k_scores(corpus_embeddings, query_embedding, TOP_K)
            search_ms = round((time.perf_counter() - started) * 1000, 2)

            ranked_ids = [corpus_ids[int(index)] for index in indices]
            relevant = gold.get(qid, set())
            first_rank = None
            for rank, snippet_id in enumerate(ranked_ids[:EVAL_K], start=1):
                if snippet_id in relevant:
                    first_rank = rank
                    break

            record = {
                "model_key": model_key,
                "model_id": model_id,
                "model_revision": revision,
                "question_id": qid,
                "question_type": q.get("type"),
                "question": q.get("body"),
                "evidence_risk_prior_label": risk.get(qid, {}).get("evidence_risk_prior_label"),
                "gold_relevant_count": len(relevant),
                "ranked": [
                    {
                        "rank": rank,
                        "snippet_id": ranked_ids[rank - 1],
                        "score": round(float(scores[rank - 1]), 8),
                    }
                    for rank in range(1, min(TOP_K, len(ranked_ids)) + 1)
                ],
                "hit_at_1": int(any(x in relevant for x in ranked_ids[:1])),
                "hit_at_5": int(any(x in relevant for x in ranked_ids[:5])),
                "hit_at_10": int(any(x in relevant for x in ranked_ids[:10])),
                "hit_at_20": int(any(x in relevant for x in ranked_ids[:20])),
                "first_relevant_rank_at_20": first_rank,
                "reciprocal_rank_at_20": 0.0 if first_rank is None else round(1 / first_rank, 8),
                "query_encode_ms": query_encode_ms,
                "search_ms": search_ms,
            }
            model_results.append(record)
            all_results.append(record)

        with RESULTS_PATH.open("a", encoding="utf-8") as handle:
            for record in model_results:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        by_type = {}
        for qtype in sorted({str(r["question_type"]) for r in model_results}):
            by_type[qtype] = aggregate(
                [r for r in model_results if str(r["question_type"]) == qtype]
            )
        by_risk = {}
        for label in sorted({str(r["evidence_risk_prior_label"]) for r in model_results}):
            by_risk[label] = aggregate(
                [r for r in model_results if str(r["evidence_risk_prior_label"]) == label]
            )

        manifest = {
            "model_key": model_key,
            "model_id": model_id,
            "model_revision": revision,
            "license_note": "Refer to the model card and repository licence at release time.",
            "max_seq_length": MAX_SEQ_LENGTH,
            "batch_size": spec["batch_size"],
            "query_prefix": spec["query_prefix"],
            "embedding_dimension": embedding_dim,
            "corpus_embedding_file": str(embedding_path.relative_to(ROOT)),
            "corpus_embedding_sha256": sha256_file(embedding_path),
            "corpus_embedding_size_bytes": embedding_path.stat().st_size,
            "corpus_count": len(corpus_ids),
            "model_load_ms": model_load_ms,
            "corpus_encode_ms": encode_ms,
            "peak_system_memory_percent_after_run": psutil.virtual_memory().percent,
        }
        model_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        model_summaries[model_key] = {
            "manifest": manifest,
            "overall": aggregate(model_results),
            "by_question_type": by_type,
            "by_evidence_risk_prior": by_risk,
        }

        print(f"{model_key} overall: {model_summaries[model_key]['overall']}")

        del corpus_embeddings
        del model
        gc.collect()

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "development pilot retrieval only",
        "pilot_question_count": len(pilot_questions),
        "candidate_snippet_count": len(corpus_ids),
        "top_k_saved": TOP_K,
        "evaluation_cutoff": EVAL_K,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "total_ram_gib": round(psutil.virtual_memory().total / 1024**3, 2),
        },
        "models": model_summaries,
        "scientific_note": (
            "Results are development-only and use the repaired pilot-pinned development split. "
            "No test-set performance is calculated."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with BY_TYPE_PATH.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "model_key",
            "model_id",
            "question_type",
            "question_count",
            "hit_rate_at_1",
            "hit_rate_at_5",
            "hit_rate_at_10",
            "hit_rate_at_20",
            "mrr_at_20",
            "mean_query_encode_ms",
            "mean_search_ms",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_key, model_summary in model_summaries.items():
            for qtype, metrics in model_summary["by_question_type"].items():
                writer.writerow(
                    {
                        "model_key": model_key,
                        "model_id": model_summary["manifest"]["model_id"],
                        "question_type": qtype,
                        **metrics,
                    }
                )

    print()
    print(f"Dense summary: {SUMMARY_PATH}")
    print(f"Dense results: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
