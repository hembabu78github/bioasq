from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage7d_common import ROOT

SOURCE_SAMPLE = (
    ROOT / "outputs" / "stage7_sampling" / "stage7_question_sample.json"
)
OUTPUT_DIR = ROOT / "outputs" / "stage7d_route_freeze"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_JSON = OUTPUT_DIR / "stage7d_route_sample.json"
SAMPLE_CSV = OUTPUT_DIR / "stage7d_route_sample.csv"
MANIFEST = OUTPUT_DIR / "stage7d_route_manifest.json"

QUESTION_TYPES = {"factoid", "list", "summary", "yesno"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "stage7_id",
        "question_id",
        "question_type",
        "retrieval_uncertainty_label",
        "graph_role",
        "question",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> int:
    if not SOURCE_SAMPLE.exists():
        print(f"ERROR: Missing Stage 7 sample: {SOURCE_SAMPLE}")
        return 1

    source = json.loads(SOURCE_SAMPLE.read_text(encoding="utf-8"))
    questions = source.get("questions", [])
    if len(questions) != 24:
        raise RuntimeError(
            f"Expected 24 development questions, found {len(questions)}."
        )

    stage7_ids = [row["stage7_id"] for row in questions]
    question_ids = [row["question_id"] for row in questions]
    if len(set(stage7_ids)) != 24 or len(set(question_ids)) != 24:
        raise RuntimeError("Duplicate Stage 7 or BioASQ question IDs found.")

    type_counts = Counter(row["question_type"] for row in questions)
    if set(type_counts) != QUESTION_TYPES or any(
        type_counts[qtype] != 6 for qtype in QUESTION_TYPES
    ):
        raise RuntimeError(
            f"Expected six questions per type; found {dict(type_counts)}."
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "sealed_test_accessed": False,
        "purpose": (
            "Freeze risk-adaptive routing across the balanced 24-question "
            "development sample before answer generation."
        ),
        "frozen_route_policy": {
            "non_list_questions": "hybrid_text_only",
            "list_questions": (
                "Use graph-selected text only when deterministic "
                "counterfactual eligibility passes; otherwise hybrid text."
            ),
            "downstream_evidence_rule": (
                "The graph may select five snippets but is never supplied to "
                "the generator or verifier."
            ),
        },
        "questions": questions,
    }
    SAMPLE_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(SAMPLE_CSV, questions)

    manifest = {
        "generated_at_utc": payload["generated_at_utc"],
        "question_count": 24,
        "question_type_counts": dict(sorted(type_counts.items())),
        "list_question_count": type_counts["list"],
        "candidate_pool_size": 20,
        "selected_evidence_count": 5,
        "reusable_stage7b_v3_routes": [
            "S7Q-007",
            "S7Q-009",
            "S7Q-011",
        ],
        "maximum_new_graph_calls": 3,
        "answer_generation_calls": 0,
        "verification_calls": 0,
        "route_scope": "deterministically eligible list questions only",
        "sample_sha256": hashlib.sha256(
            SAMPLE_JSON.read_bytes()
        ).hexdigest(),
        "scientific_note": (
            "This step freezes routing prevalence and evidence sets. It does "
            "not estimate answer quality or access the sealed test."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Stage 7D route-freeze sample prepared.")
    print("- Questions: 24")
    print("- List questions requiring route evaluation: 6")
    print("- Maximum new graph calls: 3")
    print(f"- Sample: {SAMPLE_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
