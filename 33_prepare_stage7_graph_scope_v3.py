from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone

from stage7_common import ROOT

FULL_SAMPLE = (
    ROOT / "outputs" / "stage7_sampling" / "stage7_question_sample.json"
)
OUTPUT_DIR = ROOT / "outputs" / "stage7_graph_scope_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_JSON = OUTPUT_DIR / "stage7_graph_scope_v3_sample.json"
SAMPLE_CSV = OUTPUT_DIR / "stage7_graph_scope_v3_sample.csv"
MANIFEST = OUTPUT_DIR / "stage7_graph_scope_v3_manifest.json"

ORDER = [
    "S7Q-007",
    "S7Q-009",
    "S7Q-011",
    "S7Q-013",
    "S7Q-015",
    "S7Q-017",
]

REASONS = {
    "S7Q-007": (
        "Multi-entity disease list requiring coverage of distinct DRD4 associations."
    ),
    "S7Q-009": (
        "Multi-gene list requiring isolated-NCCM qualifier coverage."
    ),
    "S7Q-011": (
        "Multi-entity cancer-type list requiring coverage across TWIST1 evidence."
    ),
    "S7Q-013": (
        "Protein-role synthesis potentially requiring several function and mechanism aspects."
    ),
    "S7Q-015": (
        "Interaction-role synthesis requiring directed Hof1-Cyk3 mechanism evidence."
    ),
    "S7Q-017": (
        "Multi-mechanism adaptive-mutagenesis synthesis."
    ),
}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if not FULL_SAMPLE.exists():
        print(f"ERROR: Missing full Stage 7 sample: {FULL_SAMPLE}")
        return 1

    source = json.loads(FULL_SAMPLE.read_text(encoding="utf-8"))
    by_id = {row["stage7_id"]: row for row in source["questions"]}
    missing = [stage7_id for stage7_id in ORDER if stage7_id not in by_id]
    if missing:
        print("ERROR: Missing sample IDs:", ", ".join(missing))
        return 1

    selected = []
    for order_index, stage7_id in enumerate(ORDER, start=1):
        row = dict(by_id[stage7_id])
        row["scope_v3_order"] = order_index
        row["scope_v3_reason"] = REASONS[stage7_id]
        row["scope_v3_question_class"] = (
            "multi_entity_list"
            if row["question_type"] == "list"
            else "multi_aspect_summary"
        )
        selected.append(row)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "sealed_test_accessed": False,
        "purpose": (
            "Final technical routing-scope gate; not an efficacy estimate."
        ),
        "questions": selected,
    }
    SAMPLE_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(SAMPLE_CSV, selected)

    manifest = {
        "generated_at_utc": payload["generated_at_utc"],
        "question_count": 6,
        "question_type_counts": {"list": 3, "summary": 3},
        "candidate_pool_size": 20,
        "selected_evidence_count": 5,
        "expected_new_graph_calls": 4,
        "reusable_v2_graphs": ["S7Q-009", "S7Q-017"],
        "answer_generation_calls": 0,
        "verification_calls": 0,
        "success_gate": (
            "At least two of six questions must have useful graph-exclusive "
            "evidence under the deterministic scope rule."
        ),
        "sample_sha256": hashlib.sha256(
            SAMPLE_JSON.read_bytes()
        ).hexdigest(),
        "scientific_note": (
            "Question selection uses pre-existing question structure only. "
            "V3 is architecture debugging and will not be reported as final efficacy."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Stage 7B V3 routing-scope sample prepared.")
    print(f"- Sample: {SAMPLE_JSON}")
    print(f"- Manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
