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
OUTPUT_DIR = ROOT / "outputs" / "stage2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "bm25_pilot_results.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "bm25_pilot_summary.json"
BY_TYPE_PATH = OUTPUT_DIR / "bm25_pilot_by_type.csv"

K1 = 1.5
B = 0.75
TOP_K = 20
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


def main() -> int:
    required = [PILOT_PATH, CORPUS_PATH, GOLD_PATH, RISK_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("ERROR: Missing required Stage 2 inputs:")
        for path in missing:
            print(f"- {path}")
        return 1

    pilot_questions = json.loads(PILOT_PATH.read_text(encoding="utf-8")).get("questions", [])
    corpus = load_jsonl(CORPUS_PATH)
    gold_rows = load_jsonl(GOLD_PATH)
    risk_rows = load_jsonl(RISK_PATH)

    gold = {
        row["question_id"]: set(row["relevant_snippet_ids"])
        for row in gold_rows
    }
    risk = {row["question_id"]: row for row in risk_rows}

    doc_ids: list[str] = []
    doc_lengths: list[int] = []
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    document_frequency: Counter[str] = Counter()

    print(f"Building BM25 index for {len(corpus)} snippets...")
    index_started = time.perf_counter()

    for doc_index, row in enumerate(corpus):
        tokens = tokenize(row["text"])
        frequencies = Counter(tokens)
        doc_ids.append(row["snippet_id"])
        doc_lengths.append(len(tokens))
        for term, tf in frequencies.items():
            postings[term].append((doc_index, tf))
            document_frequency[term] += 1

    avgdl = sum(doc_lengths) / max(len(doc_lengths), 1)
    n_docs = len(doc_ids)
    index_ms = round((time.perf_counter() - index_started) * 1000, 2)

    def search(query: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        query_terms = Counter(tokenize(query))

        for term, qtf in query_terms.items():
            df = document_frequency.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for doc_index, tf in postings[term]:
                dl = doc_lengths[doc_index]
                denom = tf + K1 * (1 - B + B * dl / avgdl)
                scores[doc_index] += qtf * idf * (tf * (K1 + 1) / denom)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], doc_ids[item[0]]))
        return [(doc_ids[index], round(score, 8)) for index, score in ranked[:top_k]]

    records: list[dict[str, Any]] = []
    print(f"Searching {len(pilot_questions)} pilot questions...")

    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        for q in pilot_questions:
            qid = str(q.get("id", ""))
            query = str(q.get("body", ""))
            started = time.perf_counter()
            ranked = search(query)
            latency_ms = round((time.perf_counter() - started) * 1000, 2)

            relevant = gold.get(qid, set())
            ranked_ids = [snippet_id for snippet_id, _ in ranked]

            first_rank = None
            for index, snippet_id in enumerate(ranked_ids, start=1):
                if snippet_id in relevant:
                    first_rank = index
                    break

            record = {
                "question_id": qid,
                "question_type": q.get("type"),
                "question": query,
                "evidence_risk_prior_label": risk.get(qid, {}).get("evidence_risk_prior_label"),
                "gold_relevant_count": len(relevant),
                "top_k": TOP_K,
                "ranked": [
                    {"snippet_id": snippet_id, "score": score}
                    for snippet_id, score in ranked
                ],
                "recall_at_1": int(any(x in relevant for x in ranked_ids[:1])),
                "recall_at_5": int(any(x in relevant for x in ranked_ids[:5])),
                "recall_at_10": int(any(x in relevant for x in ranked_ids[:10])),
                "recall_at_20": int(any(x in relevant for x in ranked_ids[:20])),
                "first_relevant_rank": first_rank,
                "reciprocal_rank": 0.0 if first_rank is None else round(1 / first_rank, 8),
                "latency_ms": latency_ms,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "question_count": len(rows),
            "hit_rate_at_1": mean([row["recall_at_1"] for row in rows]),
            "hit_rate_at_5": mean([row["recall_at_5"] for row in rows]),
            "hit_rate_at_10": mean([row["recall_at_10"] for row in rows]),
            "hit_rate_at_20": mean([row["recall_at_20"] for row in rows]),
            "mrr_at_20": mean([row["reciprocal_rank"] for row in rows]),
            "mean_query_latency_ms": mean([row["latency_ms"] for row in rows]),
        }

    overall = aggregate(records)
    by_type: dict[str, dict[str, Any]] = {}
    for qtype in sorted({str(row["question_type"]) for row in records}):
        by_type[qtype] = aggregate(
            [row for row in records if str(row["question_type"]) == qtype]
        )

    by_risk: dict[str, dict[str, Any]] = {}
    for label in sorted({str(row["evidence_risk_prior_label"]) for row in records}):
        by_risk[label] = aggregate(
            [row for row in records if str(row["evidence_risk_prior_label"]) == label]
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Pure-Python BM25 over global deduplicated snippet bank",
        "parameters": {"k1": K1, "b": B, "top_k": TOP_K},
        "candidate_snippet_count": n_docs,
        "pilot_question_count": len(records),
        "index_build_latency_ms": index_ms,
        "average_document_length_tokens": round(avgdl, 3),
        "overall": overall,
        "by_question_type": by_type,
        "by_evidence_risk_prior": by_risk,
        "interpretation": (
            "Development retrieval baseline only. This does not evaluate answer generation, "
            "claim verification, graph reasoning or final test performance."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with BY_TYPE_PATH.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "question_type",
            "question_count",
            "hit_rate_at_1",
            "hit_rate_at_5",
            "hit_rate_at_10",
            "hit_rate_at_20",
            "mrr_at_20",
            "mean_query_latency_ms",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for qtype, metrics in by_type.items():
            writer.writerow({"question_type": qtype, **metrics})

    print(f"Overall BM25 metrics: {overall}")
    print(f"Summary saved: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
