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

ROOT = Path(__file__).resolve().parent
PILOT_PATH = ROOT / "data" / "processed" / "pilot" / "bioasq11_pilot_80.json"
CORPUS_PATH = ROOT / "data" / "processed" / "stage2" / "candidate_snippets.jsonl"
GOLD_PATH = ROOT / "data" / "processed" / "stage2" / "gold_relevance.jsonl"
RISK_PATH = ROOT / "data" / "processed" / "stage2" / "query_evidence_risk_prior.jsonl"
DENSE_PATH = ROOT / "outputs" / "stage3" / "dense_pilot_results.jsonl"
OUTPUT_DIR = ROOT / "outputs" / "stage3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUTPUT_DIR / "hybrid_rrf_summary.json"
BY_TYPE_PATH = OUTPUT_DIR / "hybrid_rrf_by_type.csv"
ROUTE_PATH = OUTPUT_DIR / "route_utility_analysis.json"
HYBRID_RESULTS_PATH = OUTPUT_DIR / "hybrid_rrf_results_private.jsonl"

TOP_K_RETRIEVE = 100
EVAL_K = 20
RRF_K = 60
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def evaluate(ranked_ids: list[str], relevant: set[str]) -> dict[str, Any]:
    first_rank = None
    for rank, snippet_id in enumerate(ranked_ids[:EVAL_K], start=1):
        if snippet_id in relevant:
            first_rank = rank
            break
    return {
        "hit_at_1": int(any(x in relevant for x in ranked_ids[:1])),
        "hit_at_5": int(any(x in relevant for x in ranked_ids[:5])),
        "hit_at_10": int(any(x in relevant for x in ranked_ids[:10])),
        "hit_at_20": int(any(x in relevant for x in ranked_ids[:20])),
        "first_relevant_rank_at_20": first_rank,
        "reciprocal_rank_at_20": 0.0 if first_rank is None else round(1 / first_rank, 8),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "question_count": len(rows),
        "hit_rate_at_1": mean([r["hit_at_1"] for r in rows]),
        "hit_rate_at_5": mean([r["hit_at_5"] for r in rows]),
        "hit_rate_at_10": mean([r["hit_at_10"] for r in rows]),
        "hit_rate_at_20": mean([r["hit_at_20"] for r in rows]),
        "mrr_at_20": mean([r["reciprocal_rank_at_20"] for r in rows]),
        "mean_latency_ms": mean([r.get("latency_ms", 0.0) for r in rows]),
    }


def rrf(rankings: list[list[str]], k: int = RRF_K, top_k: int = TOP_K_RETRIEVE) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += 1.0 / (k + rank)
    return [
        item_id
        for item_id, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    ]


def main() -> int:
    required = [PILOT_PATH, CORPUS_PATH, GOLD_PATH, RISK_PATH, DENSE_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("ERROR: Missing required files:")
        for path in missing:
            print(f"- {path}")
        return 1

    pilot = json.loads(PILOT_PATH.read_text(encoding="utf-8")).get("questions", [])
    corpus = load_jsonl(CORPUS_PATH)
    gold_rows = load_jsonl(GOLD_PATH)
    risk_rows = load_jsonl(RISK_PATH)
    dense_rows = load_jsonl(DENSE_PATH)

    gold = {row["question_id"]: set(row["relevant_snippet_ids"]) for row in gold_rows}
    risk = {row["question_id"]: row for row in risk_rows}
    dense_by_model_question: dict[str, dict[str, list[str]]] = defaultdict(dict)
    dense_metrics: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for row in dense_rows:
        model_key = row["model_key"]
        qid = row["question_id"]
        dense_by_model_question[model_key][qid] = [
            item["snippet_id"] for item in row["ranked"]
        ]
        dense_metrics[model_key][qid] = row

    # BM25 index
    doc_ids: list[str] = []
    doc_lengths: list[int] = []
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    dfs: Counter[str] = Counter()

    print(f"Building BM25 index for {len(corpus)} snippets...")
    for doc_index, row in enumerate(corpus):
        frequencies = Counter(tokenize(row["text"]))
        doc_ids.append(row["snippet_id"])
        doc_lengths.append(sum(frequencies.values()))
        for term, tf in frequencies.items():
            postings[term].append((doc_index, tf))
            dfs[term] += 1

    avgdl = sum(doc_lengths) / max(len(doc_lengths), 1)
    n_docs = len(doc_ids)
    k1 = 1.5
    b = 0.75

    def bm25_search(query: str) -> list[str]:
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
        ranked = sorted(scores.items(), key=lambda x: (-x[1], doc_ids[x[0]]))
        return [doc_ids[index] for index, _ in ranked[:TOP_K_RETRIEVE]]

    method_rows: list[dict[str, Any]] = []
    route_details: list[dict[str, Any]] = []
    method_cost_order = [
        "bm25",
        "bge_small",
        "pubmedbert",
        "hybrid_bge_small",
        "hybrid_pubmedbert",
    ]

    for q in pilot:
        qid = str(q.get("id", ""))
        relevant = gold.get(qid, set())
        started = time.perf_counter()
        bm25_ranked = bm25_search(str(q.get("body", "")))
        bm25_ms = round((time.perf_counter() - started) * 1000, 2)

        rankings = {"bm25": bm25_ranked}
        latencies = {"bm25": bm25_ms}

        for model_key in sorted(dense_by_model_question):
            rankings[model_key] = dense_by_model_question[model_key][qid]
            latencies[model_key] = (
                dense_metrics[model_key][qid]["query_encode_ms"]
                + dense_metrics[model_key][qid]["search_ms"]
            )
            hybrid_key = f"hybrid_{model_key}"
            started = time.perf_counter()
            rankings[hybrid_key] = rrf([bm25_ranked, rankings[model_key]])
            fusion_ms = round((time.perf_counter() - started) * 1000, 2)
            latencies[hybrid_key] = bm25_ms + latencies[model_key] + fusion_ms

        per_method = {}
        for method, ranking in rankings.items():
            metrics = evaluate(ranking, relevant)
            record = {
                "method": method,
                "question_id": qid,
                "question_type": q.get("type"),
                "evidence_risk_prior_label": risk.get(qid, {}).get("evidence_risk_prior_label"),
                "latency_ms": round(latencies[method], 3),
                **metrics,
            }
            method_rows.append(record)
            per_method[method] = record

        # Exploratory route utility: prefer success, then better rank, then lower-cost method.
        successful = [
            method for method in method_cost_order
            if method in per_method and per_method[method]["hit_at_20"] == 1
        ]
        if successful:
            recommended = min(
                successful,
                key=lambda method: (
                    per_method[method]["first_relevant_rank_at_20"],
                    method_cost_order.index(method),
                ),
            )
        else:
            recommended = "none_successful_at_20"

        route_details.append(
            {
                "question_id": qid,
                "question_type": q.get("type"),
                "question": q.get("body"),
                "evidence_risk_prior_label": risk.get(qid, {}).get("evidence_risk_prior_label"),
                "recommended_retrieval_route_exploratory": recommended,
                "method_metrics": per_method,
            }
        )

    with HYBRID_RESULTS_PATH.open("w", encoding="utf-8") as handle:
        for row in route_details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    method_summaries = {}
    for method in sorted({row["method"] for row in method_rows}):
        rows = [row for row in method_rows if row["method"] == method]
        by_type = {}
        for qtype in sorted({str(row["question_type"]) for row in rows}):
            by_type[qtype] = aggregate(
                [row for row in rows if str(row["question_type"]) == qtype]
            )
        by_risk = {}
        for label in sorted({str(row["evidence_risk_prior_label"]) for row in rows}):
            by_risk[label] = aggregate(
                [row for row in rows if str(row["evidence_risk_prior_label"]) == label]
            )
        method_summaries[method] = {
            "overall": aggregate(rows),
            "by_question_type": by_type,
            "by_evidence_risk_prior": by_risk,
        }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fusion": {
            "method": "Reciprocal Rank Fusion",
            "rrf_k": RRF_K,
            "input_depth": TOP_K_RETRIEVE,
            "evaluation_cutoff": EVAL_K,
        },
        "methods": method_summaries,
        "scientific_note": (
            "Development-only retrieval comparison. Hybrid selection and route rules are "
            "not frozen and no test-set metrics are calculated."
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
            "mean_latency_ms",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, method_summary in method_summaries.items():
            for qtype, metrics in method_summary["by_question_type"].items():
                writer.writerow({"method": method, "question_type": qtype, **metrics})

    # Route and risk-prior exploratory analysis
    route_counts = Counter(
        row["recommended_retrieval_route_exploratory"] for row in route_details
    )

    bm25_failures = {
        row["question_id"]
        for row in method_rows
        if row["method"] == "bm25" and row["hit_at_20"] == 0
    }
    rescue_counts = {}
    for method in method_summaries:
        if method == "bm25":
            continue
        rescued = {
            row["question_id"]
            for row in method_rows
            if row["method"] == method
            and row["question_id"] in bm25_failures
            and row["hit_at_20"] == 1
        }
        rescue_counts[method] = {
            "bm25_failure_count": len(bm25_failures),
            "rescued_count": len(rescued),
            "rescued_question_ids": sorted(rescued),
        }

    risk_prior_analysis = {}
    for label in sorted(
        {row["evidence_risk_prior_label"] for row in route_details}
    ):
        subset = [row for row in route_details if row["evidence_risk_prior_label"] == label]
        risk_prior_analysis[label] = {
            "question_count": len(subset),
            "bm25_failure_at_20_count": sum(
                1 for row in subset if row["method_metrics"]["bm25"]["hit_at_20"] == 0
            ),
            "bm25_failure_at_20_rate": (
                round(
                    sum(
                        1
                        for row in subset
                        if row["method_metrics"]["bm25"]["hit_at_20"] == 0
                    )
                    / len(subset),
                    6,
                )
                if subset else None
            ),
            "hybrid_pubmedbert_failure_at_20_count": sum(
                1
                for row in subset
                if row["method_metrics"].get("hybrid_pubmedbert", {}).get("hit_at_20") == 0
            ),
        }

    route_analysis = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "route_counts_exploratory": dict(sorted(route_counts.items())),
        "bm25_rescue_analysis": rescue_counts,
        "evidence_risk_prior_analysis": risk_prior_analysis,
        "risk_prior_status": (
            "Provisional. It is strongly associated with question type and will be retained "
            "only if it predicts retrieval failure or route utility on development data "
            "beyond question type."
        ),
        "next_design_rule": (
            "Final adaptive routing must use query-only and retrieval-diagnostic features, "
            "such as score margin, score dispersion, agreement between retrievers and graph "
            "path availability. Gold relevance is used only to train/evaluate the route policy "
            "on train/development data."
        ),
    }
    ROUTE_PATH.write_text(json.dumps(route_analysis, indent=2), encoding="utf-8")

    print("Overall retrieval comparison:")
    for method, values in method_summaries.items():
        print(f"- {method}: {values['overall']}")
    print(f"Route analysis saved: {ROUTE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
