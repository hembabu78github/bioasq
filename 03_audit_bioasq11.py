from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "bioasq11" / "training11b.json"
OUTPUT_DIR = ROOT / "outputs" / "stage1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def percentile(values: list[int], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def describe(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "p90": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.mean(values), 3),
        "p90": round(percentile(values, 0.90) or 0, 3),
        "max": max(values),
    }


def main() -> int:
    if not DATA_PATH.exists():
        print(f"ERROR: Missing dataset: {DATA_PATH}")
        return 1

    try:
        root = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ERROR: Invalid dataset JSON: {exc}")
        return 1

    questions = root.get("questions")
    if not isinstance(questions, list):
        print("ERROR: Top-level 'questions' is not a list.")
        return 1

    type_counts: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    bodies: dict[str, list[str]] = defaultdict(list)
    unique_documents: set[str] = set()
    unique_snippets: set[str] = set()
    duplicate_snippet_occurrences = 0
    malformed_snippets = 0
    exact_shapes: Counter[str] = Counter()

    document_counts: list[int] = []
    snippet_counts: list[int] = []
    body_words: list[int] = []
    snippet_words: list[int] = []

    required = ["id", "body", "type", "documents", "snippets", "ideal_answer"]

    for index, q in enumerate(questions):
        if not isinstance(q, dict):
            missing["non_object_question"] += 1
            continue

        for field in required:
            if field not in q or q[field] in (None, "", []):
                missing[field] += 1

        qid = str(q.get("id", f"__index_{index}"))
        if qid in seen_ids:
            duplicate_ids.append(qid)
        seen_ids.add(qid)

        qtype = str(q.get("type", "missing")).strip().lower()
        type_counts[qtype] += 1

        body = str(q.get("body", ""))
        norm_body = normalise(body)
        if norm_body:
            bodies[norm_body].append(qid)
        body_words.append(len(body.split()))

        documents = q.get("documents") if isinstance(q.get("documents"), list) else []
        snippets = q.get("snippets") if isinstance(q.get("snippets"), list) else []
        document_counts.append(len(documents))
        snippet_counts.append(len(snippets))
        unique_documents.update(str(item) for item in documents)

        for snippet in snippets:
            if not isinstance(snippet, dict) or not isinstance(snippet.get("text"), str):
                malformed_snippets += 1
                continue
            norm_text = normalise(snippet["text"])
            snippet_words.append(len(snippet["text"].split()))
            if norm_text in unique_snippets:
                duplicate_snippet_occurrences += 1
            else:
                unique_snippets.add(norm_text)

        if "exact_answer" not in q:
            exact_shapes["missing"] += 1
        else:
            exact = q["exact_answer"]
            if isinstance(exact, str):
                exact_shapes["string"] += 1
            elif isinstance(exact, list):
                if exact and all(isinstance(x, list) for x in exact):
                    exact_shapes["list_of_lists"] += 1
                else:
                    exact_shapes["flat_list"] += 1
            else:
                exact_shapes[type(exact).__name__] += 1

    duplicate_bodies = {body: ids for body, ids in bodies.items() if len(ids) > 1}

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_relative_path": str(DATA_PATH.relative_to(ROOT)),
        "question_count": len(questions),
        "question_type_counts": dict(sorted(type_counts.items())),
        "missing_or_empty_required_fields": dict(sorted(missing.items())),
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_ids": duplicate_ids[:100],
        "duplicate_normalised_body_group_count": len(duplicate_bodies),
        "duplicate_normalised_body_question_count": sum(len(ids) for ids in duplicate_bodies.values()),
        "duplicate_body_examples": [
            {"normalised_body": body, "ids": ids}
            for body, ids in list(duplicate_bodies.items())[:25]
        ],
        "unique_document_count": len(unique_documents),
        "unique_normalised_snippet_count": len(unique_snippets),
        "duplicate_snippet_occurrence_count": duplicate_snippet_occurrences,
        "malformed_snippet_count": malformed_snippets,
        "exact_answer_shape_counts": dict(sorted(exact_shapes.items())),
        "distributions": {
            "documents_per_question": describe(document_counts),
            "snippets_per_question": describe(snippet_counts),
            "question_words": describe(body_words),
            "snippet_words": describe(snippet_words),
        },
        "retrieval_design_warning": (
            "The file contains question-linked gold documents and snippets, not a complete "
            "PubMed corpus. The later retrieval design must define a candidate corpus and "
            "must not provide test-question gold evidence directly to the retriever."
        ),
        "split_status": "No final train/development/test split has been created.",
    }

    audit_path = OUTPUT_DIR / "bioasq11_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    counts_path = OUTPUT_DIR / "bioasq11_question_type_counts.csv"
    with counts_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question_type", "count"])
        for qtype, count in sorted(type_counts.items()):
            writer.writerow([qtype, count])

    print(f"Questions: {len(questions)}")
    print(f"Types: {dict(sorted(type_counts.items()))}")
    print(f"Unique documents: {len(unique_documents)}")
    print(f"Unique snippets: {len(unique_snippets)}")
    print(f"Duplicate body groups: {len(duplicate_bodies)}")
    print(f"Audit saved: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
