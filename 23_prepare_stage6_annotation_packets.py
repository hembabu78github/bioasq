from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from stage6_common import ROOT, SEED, load_jsonl, write_csv

INPUT_PATH = (
    ROOT / "outputs" / "stage6_prep"
    / "adversarial_claim_candidates_flat_private.jsonl"
)
OUTPUT_DIR = ROOT / "outputs" / "stage6_annotation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "audit_id",
    "question_type",
    "question",
    "claim",
    "retrieved_evidence",
    "local_graph",
    "annotator_label",
    "annotator_rationale",
]


def packet_rows(rows: list[dict], seed: int) -> list[dict]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return [
        {
            "audit_id": row["audit_id"],
            "question_type": row["question_type"],
            "question": row["question"],
            "claim": row["adversarial_claim"],
            "retrieved_evidence": row["retrieved_evidence"],
            "local_graph": row["local_graph"],
            "annotator_label": "",
            "annotator_rationale": "",
        }
        for row in shuffled
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"ERROR: Missing adversarial candidates: {INPUT_PATH}")
        return 1

    rows = load_jsonl(INPUT_PATH)
    if len(rows) != 48:
        print(f"ERROR: Expected 48 candidates, found {len(rows)}.")
        return 2

    packet_a = OUTPUT_DIR / "stage6_annotator_A.csv"
    packet_b = OUTPUT_DIR / "stage6_annotator_B.csv"
    write_csv(packet_a, packet_rows(rows, SEED + 101), FIELDS)
    write_csv(packet_b, packet_rows(rows, SEED + 202), FIELDS)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_count": len(rows),
        "packet_A_sha256": sha256(packet_a),
        "packet_B_sha256": sha256(packet_b),
        "packet_orders_are_independently_randomized": True,
        "model_labels_in_packets": False,
        "permitted_labels": [
            "supported",
            "contradicted",
            "insufficient_evidence",
        ],
        "annotation_rule": (
            "Judge only against displayed evidence and graph. No outside knowledge."
        ),
    }
    path = OUTPUT_DIR / "stage6_annotation_packet_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Created: {packet_a}")
    print(f"Created: {packet_b}")
    print(f"Manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
