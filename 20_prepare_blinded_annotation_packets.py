from __future__ import annotations

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs" / "stage4_correction" / "claim_manual_audit_blinded.csv"
OUT = ROOT / "outputs" / "stage5_annotation"
OUT.mkdir(parents=True, exist_ok=True)

ALLOWED_BASE_COLUMNS = [
    "audit_id", "question_type", "question", "claim", "claim_importance",
    "retrieved_evidence", "local_graph"
]


def main() -> int:
    if not SOURCE.exists():
        print(f"ERROR: Missing blinded source file: {SOURCE}")
        return 1

    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 28:
        print(f"ERROR: Expected 28 audit rows, found {len(rows)}.")
        return 2

    for annotator, seed in (("A", 20260725), ("B", 20260726)):
        packet = []
        for row in rows:
            item = {column: row.get(column, "") for column in ALLOWED_BASE_COLUMNS}
            item["annotator_label"] = ""
            item["annotator_rationale"] = ""
            packet.append(item)
        random.Random(seed).shuffle(packet)

        path = OUT / f"claim_audit_annotator_{annotator}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(packet[0].keys()))
            writer.writeheader()
            writer.writerows(packet)
        print(f"Created: {path}")

    print("Give Annotator A and Annotator B only their own CSV and the protocol DOCX.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
