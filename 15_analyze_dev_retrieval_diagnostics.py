from __future__ import annotations

import csv
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "bioasq11" / "training11b.json"
DEV_IDS_PATH = ROOT / "data" / "processed" / "stage3" / "dev_ids_v2.txt"
CORPUS_PATH = ROOT / "data" / "processed" / "stage2" / "candidate_snippets.jsonl"
GOLD_PATH = ROOT / "data" / "processed" / "stage2" / "gold_relevance.jsonl"
RISK_PATH = ROOT / "data" / "processed" / "stage2" / "query_evidence_risk_prior.jsonl"
EMBED_PATH = ROOT / "models" / "stage3_embeddings" / "bge_small_corpus_embeddings.npy"
EMBED_IDS_PATH = ROOT / "models" / "stage3_embeddings" / "bge_small_corpus_ids.json"

OUTPUT_DIR = ROOT / "outputs" / "stage4"
PRIVATE_DIR = ROOT / "data" / "processed" / "stage4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRIVATE_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUTPUT_DIR / "dev_retrieval_diagnostics_summary.json"
BY_TYPE_PATH = OUTPUT_DIR / "dev_retrieval_by_type.csv"
DIAGNOSTICS_PATH = PRIVATE_DIR / "dev_retrieval_diagnostics_private.jsonl"
RANKINGS_PATH = PRIVATE_DIR / "dev_retrieval_rankings_private.jsonl"

MODEL_ID = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K = 100
EVAL_K = 20
RRF_K = 60
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def percentile_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    if len(values) <= 1:
        return ranks
    for rank, index in enumerate(order):
        ranks[index] = rank / (len(values) - 1)
    return ranks


def evaluate(ranked: list[str], relevant: set[str]) -> dict[str, Any]:
    first_rank = None
    for rank, snippet_id in enumerate(ranked[:EVAL_K], start=1):
        if snippet_id in relevant:
            first_rank = rank
            break
    return {
        "hit_at_1": int(any(x in relevant for x in ranked[:1])),
        "hit_at_5": int(any(x in relevant for x in ranked[:5])),
        "hit_at_10": int(any(x in relevant for x in ranked[:10])),
        "hit_at_20": int(any(x in relevant for x in ranked[:20])),
        "first_relevant_rank_at_20": first_rank,
        "reciprocal_rank_at_20": 0.0 if first_rank is None else round(1 / first_rank, 8),
    }


def rrf(a: list[str], b: list[str]) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in (a, b):
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += 1.0 / (RRF_K + rank)
    return [
        item_id
        for item_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:TOP_K]
    ]


def top_dense(
    corpus_embeddings: np.ndarray,
    query_embedding: np.ndarray,
) -> tuple[list[int], list[float]]:
    scores = corpus_embeddings @ query_embedding
    k = min(TOP_K, len(scores))
    candidates = np.argpartition(scores, -k)[-k:]
    ordered = candidates[np.argsort(scores[candidates])[::-1]]
    return [int(i) for i in ordered], [float(scores[i]) for i in ordered]


def aggregate(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    return {
        "question_count": len(rows),
        "hit_rate_at_1": mean([r[f"{prefix}_hit_at_1"] for r in rows]),
        "hit_rate_at_5": mean([r[f"{prefix}_hit_at_5"] for r in rows]),
        "hit_rate_at_10": mean([r[f"{prefix}_hit_at_10"] for r in rows]),
        "hit_rate_at_20": mean([r[f"{prefix}_hit_at_20"] for r in rows]),
        "mrr_at_20": mean([r[f"{prefix}_reciprocal_rank_at_20"] for r in rows]),
    }


def main() -> int:
    required = [
        DATA_PATH,
        DEV_IDS_PATH,
        CORPUS_PATH,
        GOLD_PATH,
        RISK_PATH,
        EMBED_PATH,
        EMBED_IDS_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("ERROR: Missing Stage 3 inputs:")
        for path in missing:
            print(f"- {path}")
        return 1

    all_questions = json.loads(DATA_PATH.read_text(encoding="utf-8")).get("questions", [])
    question_map = {str(q.get("id", "")): q for q in all_questions}
    dev_ids = [
        line.strip()
        for line in DEV_IDS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    questions = [question_map[qid] for qid in dev_ids]

    corpus = load_jsonl(CORPUS_PATH)
    corpus_ids = [row["snippet_id"] for row in corpus]
    corpus_texts = [row["text"] for row in corpus]
    cached_ids = json.loads(EMBED_IDS_PATH.read_text(encoding="utf-8"))
    if cached_ids != corpus_ids:
        print("ERROR: Cached BGE corpus IDs do not match the Stage 2 corpus.")
        return 2

    gold = {
        row["question_id"]: set(row["relevant_snippet_ids"])
        for row in load_jsonl(GOLD_PATH)
    }
    risk = {row["question_id"]: row for row in load_jsonl(RISK_PATH)}

    # Build BM25
    print(f"Building BM25 index for {len(corpus)} snippets...")
    doc_lengths: list[int] = []
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    dfs: Counter[str] = Counter()
    for doc_index, text in enumerate(corpus_texts):
        frequencies = Counter(tokenize(text))
        doc_lengths.append(sum(frequencies.values()))
        for term, tf in frequencies.items():
            postings[term].append((doc_index, tf))
            dfs[term] += 1

    avgdl = sum(doc_lengths) / max(len(doc_lengths), 1)
    n_docs = len(corpus_ids)
    k1 = 1.5
    b = 0.75

    def bm25_search(query: str) -> tuple[list[str], list[float]]:
        scores: dict[int, float] = defaultdict(float)
        for term, qtf in Counter(tokenize(query)).items():
            df = dfs.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for doc_index, tf in postings[term]:
                dl = doc_lengths[doc_index]
                denom = tf + k1 * (1 - b + b * dl / avgdl)
                scores[doc_index] += qtf * idf * (tf * (k1 + 1) / denom)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], corpus_ids[item[0]]))
        ranked = ranked[:TOP_K]
        return (
            [corpus_ids[index] for index, _ in ranked],
            [float(score) for _, score in ranked],
        )

    print(f"Encoding {len(questions)} development queries with BGE-small...")
    model = SentenceTransformer(MODEL_ID, device="cpu")
    model.max_seq_length = 256
    query_texts = [QUERY_PREFIX + str(q.get("body", "")) for q in questions]
    query_embeddings = model.encode(
        query_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    corpus_embeddings = np.load(EMBED_PATH, mmap_mode="r")

    raw_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []

    print("Running retrieval diagnostics...")
    for q, query_embedding in zip(questions, query_embeddings):
        qid = str(q.get("id", ""))
        relevant = gold[qid]

        bm25_started = time.perf_counter()
        bm25_ids, bm25_scores = bm25_search(str(q.get("body", "")))
        bm25_ms = round((time.perf_counter() - bm25_started) * 1000, 3)

        dense_started = time.perf_counter()
        dense_indices, dense_scores = top_dense(corpus_embeddings, query_embedding)
        bge_ms = round((time.perf_counter() - dense_started) * 1000, 3)
        bge_ids = [corpus_ids[index] for index in dense_indices]

        hybrid_started = time.perf_counter()
        hybrid_ids = rrf(bm25_ids, bge_ids)
        hybrid_ms = round(
            bm25_ms + bge_ms + (time.perf_counter() - hybrid_started) * 1000,
            3,
        )

        bm = evaluate(bm25_ids, relevant)
        bg = evaluate(bge_ids, relevant)
        hy = evaluate(hybrid_ids, relevant)

        overlap5 = len(set(bm25_ids[:5]) & set(bge_ids[:5])) / 5
        overlap10 = len(set(bm25_ids[:10]) & set(bge_ids[:10])) / 10
        overlap20 = len(set(bm25_ids[:20]) & set(bge_ids[:20])) / 20

        bm25_top1 = bm25_scores[0] if bm25_scores else 0.0
        bm25_top2 = bm25_scores[1] if len(bm25_scores) > 1 else 0.0
        bge_top1 = dense_scores[0] if dense_scores else 0.0
        bge_top2 = dense_scores[1] if len(dense_scores) > 1 else 0.0

        row = {
            "question_id": qid,
            "question_type": str(q.get("type", "")),
            "question": str(q.get("body", "")),
            "question_word_count": len(str(q.get("body", "")).split()),
            "query_only_risk_label": risk[qid]["evidence_risk_prior_label"],
            "query_only_risk_score": risk[qid]["evidence_risk_prior_score"],
            "bm25_top1_score": bm25_top1,
            "bm25_margin": bm25_top1 - bm25_top2,
            "bge_top1_score": bge_top1,
            "bge_margin": bge_top1 - bge_top2,
            "same_top1": int(bool(bm25_ids and bge_ids and bm25_ids[0] == bge_ids[0])),
            "overlap_at_5": overlap5,
            "overlap_at_10": overlap10,
            "overlap_at_20": overlap20,
            "bm25_latency_ms": bm25_ms,
            "bge_search_latency_ms": bge_ms,
            "hybrid_latency_ms": hybrid_ms,
        }
        for prefix, metrics in (("bm25", bm), ("bge", bg), ("hybrid", hy)):
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        raw_rows.append(row)

        ranking_rows.append(
            {
                "question_id": qid,
                "question_type": str(q.get("type", "")),
                "question": str(q.get("body", "")),
                "bm25_ranked_ids": bm25_ids,
                "bge_ranked_ids": bge_ids,
                "hybrid_bge_ranked_ids": hybrid_ids,
            }
        )

    bm25_margin_ranks = percentile_ranks([row["bm25_margin"] for row in raw_rows])
    bge_margin_ranks = percentile_ranks([row["bge_margin"] for row in raw_rows])

    for index, row in enumerate(raw_rows):
        uncertainty = (
            0.30 * (1 - bge_margin_ranks[index])
            + 0.20 * (1 - bm25_margin_ranks[index])
            + 0.25 * (1 - row["overlap_at_10"])
            + 0.25 * (1 - row["same_top1"])
        )
        row["retrieval_uncertainty_score"] = round(uncertainty, 6)
        if uncertainty < 0.40:
            label = "low"
        elif uncertainty < 0.65:
            label = "medium"
        else:
            label = "high"
        row["retrieval_uncertainty_label"] = label

        bge_rank = row["bge_first_relevant_rank_at_20"] or 999
        hybrid_rank = row["hybrid_first_relevant_rank_at_20"] or 999
        row["hybrid_improves_rank_by_3_or_more"] = int(hybrid_rank + 3 <= bge_rank)
        row["hybrid_hurts_rank_by_3_or_more"] = int(bge_rank + 3 <= hybrid_rank)
        row["bge_failure_at_5"] = int(row["bge_hit_at_5"] == 0)

    with DIAGNOSTICS_PATH.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with RANKINGS_PATH.open("w", encoding="utf-8") as handle:
        for row in ranking_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    methods = {}
    for prefix in ("bm25", "bge", "hybrid"):
        methods[prefix] = {
            "overall": aggregate(raw_rows, prefix),
            "by_uncertainty": {
                label: aggregate(
                    [r for r in raw_rows if r["retrieval_uncertainty_label"] == label],
                    prefix,
                )
                for label in ("low", "medium", "high")
            },
            "by_question_type": {
                qtype: aggregate(
                    [r for r in raw_rows if r["question_type"] == qtype],
                    prefix,
                )
                for qtype in sorted({r["question_type"] for r in raw_rows})
            },
        }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_question_count": len(raw_rows),
        "test_questions_accessed": False,
        "methods": methods,
        "retrieval_uncertainty": {
            "definition": (
                "Deployment-available composite of BGE/BM25 score margins, top-10 "
                "overlap and top-1 agreement. It uses no gold answers or relevance labels."
            ),
            "label_counts": dict(
                sorted(Counter(r["retrieval_uncertainty_label"] for r in raw_rows).items())
            ),
            "bge_failure_at_5_by_label": {
                label: {
                    "question_count": len(
                        [r for r in raw_rows if r["retrieval_uncertainty_label"] == label]
                    ),
                    "failure_count": sum(
                        r["bge_failure_at_5"]
                        for r in raw_rows
                        if r["retrieval_uncertainty_label"] == label
                    ),
                }
                for label in ("low", "medium", "high")
            },
        },
        "hybrid_effect": {
            "improves_rank_by_3_or_more_count": sum(
                r["hybrid_improves_rank_by_3_or_more"] for r in raw_rows
            ),
            "hurts_rank_by_3_or_more_count": sum(
                r["hybrid_hurts_rank_by_3_or_more"] for r in raw_rows
            ),
            "bge_failure_at_5_count": sum(r["bge_failure_at_5"] for r in raw_rows),
        },
        "private_outputs": {
            "diagnostics": str(DIAGNOSTICS_PATH.relative_to(ROOT)),
            "rankings": str(RANKINGS_PATH.relative_to(ROOT)),
        },
        "scientific_note": (
            "All performance analysis is development-only. The uncertainty score is "
            "provisional and will be retained only if it separates retrieval failures "
            "or graph/verification utility."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with BY_TYPE_PATH.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "method",
            "question_type",
            "question_count",
            "hit_rate_at_1",
            "hit_rate_at_5",
            "hit_rate_at_10",
            "hit_rate_at_20",
            "mrr_at_20",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, method_summary in methods.items():
            for qtype, metrics in method_summary["by_question_type"].items():
                writer.writerow({"method": method, "question_type": qtype, **metrics})

    print(f"Development questions analysed: {len(raw_rows)}")
    print(f"Uncertainty labels: {summary['retrieval_uncertainty']['label_counts']}")
    print(f"Summary saved: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
