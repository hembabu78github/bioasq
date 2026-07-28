from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from stage7_common import ROOT

FULL_SAMPLE = (
    ROOT / "outputs" / "stage7_sampling" / "stage7_question_sample.json"
)
OUTPUT_DIR = ROOT / "outputs" / "stage7_graph_routing_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_JSON = OUTPUT_DIR / "stage7_graph_routing_v2_sample.json"
SAMPLE_CSV = OUTPUT_DIR / "stage7_graph_routing_v2_sample.csv"
MANIFEST = OUTPUT_DIR / "stage7_graph_routing_v2_manifest.json"

SELECTION = {
    "S7Q-005": {
        "structural_role": "graph_stress",
        "structural_reason": (
            "Causal gene-mutation-to-disorder relation with a material biallelic qualifier."
        ),
    },
    "S7Q-002": {
        "structural_role": "direct_control",
        "structural_reason": "Single numeric fact lookup; graph route should not be required.",
    },
    "S7Q-009": {
        "structural_role": "graph_stress",
        "structural_reason": (
            "Multi-entity list question requiring coverage and the material isolated-disease qualifier."
        ),
    },
    "S7Q-008": {
        "structural_role": "direct_control",
        "structural_reason": "Direct taxonomy enumeration without a multi-hop relation.",
    },
    "S7Q-017": {
        "structural_role": "graph_stress",
        "structural_reason": (
            "Multi-mechanism synthesis requiring coverage of several repair and polymerase factors."
        ),
    },
    "S7Q-018": {
        "structural_role": "direct_control",
        "structural_reason": (
            "Reimbursement summary is a policy synthesis task, not a biomedical entity-relation graph task."
        ),
    },
    "S7Q-021": {
        "structural_role": "graph_stress",
        "structural_reason": (
            "Mechanistic protein-regulation question with a directed relation."
        ),
    },
    "S7Q-020": {
        "structural_role": "direct_control",
        "structural_reason": "Direct feasibility yes/no lookup; graph route should not be required.",
    },
}

ORDER = [
    "S7Q-005", "S7Q-002",
    "S7Q-009", "S7Q-008",
    "S7Q-017", "S7Q-018",
    "S7Q-021", "S7Q-020",
]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if not FULL_SAMPLE.exists():
        print(f"ERROR: Missing Stage 7 full sample: {FULL_SAMPLE}")
        return 1

    source = json.loads(FULL_SAMPLE.read_text(encoding="utf-8"))
    by_id = {row["stage7_id"]: row for row in source["questions"]}

    missing = [stage7_id for stage7_id in ORDER if stage7_id not in by_id]
    if missing:
        print("ERROR: Missing required Stage 7 sample IDs:", ", ".join(missing))
        return 1

    selected = []
    for order_index, stage7_id in enumerate(ORDER, start=1):
        row = dict(by_id[stage7_id])
        row.update(SELECTION[stage7_id])
        row["routing_v2_order"] = order_index
        selected.append(row)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "sealed_test_accessed": False,
        "purpose": (
            "Technical graph-routing stress test only; not an efficacy estimate."
        ),
        "selection_basis": (
            "One structurally graph-demanding and one direct-control question "
            "per BioASQ question type, with a 3/2/3 low-medium-high uncertainty distribution."
        ),
        "questions": selected,
    }
    SAMPLE_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(SAMPLE_CSV, selected)

    manifest = {
        "generated_at_utc": payload["generated_at_utc"],
        "question_count": 8,
        "question_type_counts": {
            "factoid": 2,
            "list": 2,
            "summary": 2,
            "yesno": 2,
        },
        "uncertainty_counts": {
            "low": 3,
            "medium": 2,
            "high": 3,
        },
        "structural_role_counts": {
            "graph_stress": 4,
            "direct_control": 4,
        },
        "candidate_pool_size": 20,
        "selected_evidence_count": 5,
        "answer_generation_calls": 0,
        "verification_calls": 0,
        "expected_graph_calls": 8,
        "sample_sha256": hashlib.sha256(SAMPLE_JSON.read_bytes()).hexdigest(),
        "scientific_note": (
            "This protocol amendment follows a failed technical smoke gate. "
            "Its outputs will not be reported as final performance estimates."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Stage 7 graph-routing V2 sample prepared.")
    print(f"- Sample: {SAMPLE_JSON}")
    print(f"- Manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
