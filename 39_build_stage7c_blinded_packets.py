from __future__ import annotations

import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage7_common import ROOT, load_jsonl

OUTPUT_DIR = ROOT / "outputs" / "stage7c_selective_answer_smoke"
RESULTS = OUTPUT_DIR / "stage7c_answer_smoke_results.jsonl"

PACKET_A = OUTPUT_DIR / "stage7c_blinded_annotator_A.csv"
PACKET_B = OUTPUT_DIR / "stage7c_blinded_annotator_B.csv"
PRIVATE_KEY = OUTPUT_DIR / "stage7c_blinded_private_key.json"
MANIFEST = OUTPUT_DIR / "stage7c_blinded_packet_manifest.json"


def blind_id(arm_id: str) -> str:
    digest = hashlib.sha256(
        f"stage7c|{arm_id}|20260726".encode("utf-8")
    ).hexdigest()[:10].upper()
    return f"BL-{digest}"


def evidence_text(row: dict[str, Any]) -> str:
    return "\n\n".join(
        f"[{item['evidence_id']}] {item['text']}"
        for item in row["evidence"]
    )


def packet_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for result in results:
        claims = result["claims"] or [
            {
                "claim_id": "NO-CLAIM",
                "text": "(Generator produced no atomic claim.)",
            }
        ]
        for claim in claims:
            output.append(
                {
                    "blind_item_id": blind_id(result["arm_id"]),
                    "question_type": result["question_type"],
                    "question": result["question"],
                    "answer_text": result["answer"],
                    "generator_abstained": result[
                        "generator_abstained"
                    ],
                    "claim_id": claim["claim_id"],
                    "claim_text": claim["text"],
                    "evidence_text": evidence_text(result),
                    "human_claim_label": "",
                    "answer_complete_yes_no": "",
                    "answer_utility_1_to_5": "",
                    "notes": "",
                }
            )
    return output


def write_packet(path: Path, rows: list[dict[str, Any]], seed: int) -> None:
    rows = [dict(row) for row in rows]
    random.Random(seed).shuffle(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if not RESULTS.exists():
        print(f"ERROR: Missing results: {RESULTS}")
        return 1

    results = load_jsonl(RESULTS)
    rows = packet_rows(results)
    write_packet(PACKET_A, rows, 2026072601)
    write_packet(PACKET_B, rows, 2026072602)

    private_key = {
        blind_id(result["arm_id"]): {
            "arm_id": result["arm_id"],
            "stage7_id": result["stage7_id"],
            "route": result["route"],
            "evidence_snippet_ids": result[
                "evidence_snippet_ids"
            ],
            "automated_verifications": result[
                "verifications"
            ],
            "automated_final_disposition": result[
                "final_disposition"
            ],
        }
        for result in results
    }
    PRIVATE_KEY.write_text(
        json.dumps(private_key, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "answer_count": len(results),
        "claim_row_count_per_packet": len(rows),
        "packet_A": str(PACKET_A),
        "packet_B": str(PACKET_B),
        "labels_allowed": [
            "supported",
            "contradicted",
            "insufficient_evidence",
        ],
        "blinding": (
            "Route names, automated verifier labels and final dispositions are "
            "excluded from annotator packets."
        ),
        "instruction": (
            "Each annotator independently labels every claim against only the "
            "displayed evidence, then rates answer completeness and utility."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Annotator A packet: {PACKET_A}")
    print(f"Annotator B packet: {PACKET_B}")
    print(f"Packet manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
