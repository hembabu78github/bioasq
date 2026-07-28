from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DIAGNOSTICS_PATH = (
    ROOT / "data" / "processed" / "stage4" / "dev_retrieval_diagnostics_private.jsonl"
)
OUTPUT_DIR = ROOT / "outputs" / "stage4"
PRIVATE_DIR = ROOT / "data" / "processed" / "stage4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRIVATE_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = OUTPUT_DIR / "graph_claim_pilot_manifest.json"
SELECTED_PATH = PRIVATE_DIR / "graph_claim_pilot_ids.json"

PER_TYPE = 3


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    if not DIAGNOSTICS_PATH.exists():
        print(f"ERROR: Missing diagnostics: {DIAGNOSTICS_PATH}")
        return 1

    rows = load_jsonl(DIAGNOSTICS_PATH)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["question_type"]].append(row)

    selected: list[dict[str, Any]] = []
    selection_reasons: dict[str, str] = {}

    for qtype in sorted(grouped):
        candidates = grouped[qtype]
        chosen: list[dict[str, Any]] = []

        priorities = [
            (
                "retrieval_hard_case",
                sorted(
                    [
                        r
                        for r in candidates
                        if r["bge_failure_at_5"] == 1
                        or r["hybrid_improves_rank_by_3_or_more"] == 1
                    ],
                    key=lambda r: (
                        -r["retrieval_uncertainty_score"],
                        r["question_id"],
                    ),
                ),
            ),
            (
                "high_uncertainty",
                sorted(
                    [
                        r
                        for r in candidates
                        if r["retrieval_uncertainty_label"] == "high"
                    ],
                    key=lambda r: (-r["retrieval_uncertainty_score"], r["question_id"]),
                ),
            ),
            (
                "medium_uncertainty",
                sorted(
                    [
                        r
                        for r in candidates
                        if r["retrieval_uncertainty_label"] == "medium"
                    ],
                    key=lambda r: (-r["retrieval_uncertainty_score"], r["question_id"]),
                ),
            ),
            (
                "low_uncertainty_control",
                sorted(
                    [
                        r
                        for r in candidates
                        if r["retrieval_uncertainty_label"] == "low"
                    ],
                    key=lambda r: (r["retrieval_uncertainty_score"], r["question_id"]),
                ),
            ),
        ]

        for reason, pool in priorities:
            for row in pool:
                if len(chosen) >= PER_TYPE:
                    break
                if any(x["question_id"] == row["question_id"] for x in chosen):
                    continue
                chosen.append(row)
                selection_reasons[row["question_id"]] = reason
            if len(chosen) >= PER_TYPE:
                break

        selected.extend(chosen)

    selected.sort(key=lambda row: (row["question_type"], row["question_id"]))

    selected_records = []
    for row in selected:
        route = (
            "hybrid_bge"
            if row["retrieval_uncertainty_label"] == "high"
            else "bge"
        )
        selected_records.append(
            {
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "question": row["question"],
                "selection_reason": selection_reasons[row["question_id"]],
                "retrieval_uncertainty_score": row["retrieval_uncertainty_score"],
                "retrieval_uncertainty_label": row["retrieval_uncertainty_label"],
                "provisional_route": route,
            }
        )

    SELECTED_PATH.write_text(
        json.dumps({"questions": selected_records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    digest = hashlib.sha256(SELECTED_PATH.read_bytes()).hexdigest()

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_question_count": len(selected_records),
        "questions_per_type": PER_TYPE,
        "selection": (
            "Development-only purposive feasibility sample: prioritize retrieval-hard "
            "and high-uncertainty questions, then include medium/low controls, with equal "
            "question-type representation."
        ),
        "not_for_final_performance_estimation": True,
        "selected_questions": selected_records,
        "selected_file_relative_path": str(SELECTED_PATH.relative_to(ROOT)),
        "selected_file_sha256": digest,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Selected questions: {len(selected_records)}")
    for row in selected_records:
        print(
            f"- {row['question_type']:<8} {row['retrieval_uncertainty_label']:<6} "
            f"{row['provisional_route']:<10} {row['question']}"
        )
    print(f"Manifest saved: {MANIFEST_PATH}")
    return 0 if len(selected_records) == 12 else 2


if __name__ == "__main__":
    raise SystemExit(main())
