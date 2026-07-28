from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from stage6_common import (
    ALLOWED_LABELS,
    ROOT,
    cohen_kappa,
    read_csv,
    write_csv,
)

INPUT_DIR = ROOT / "outputs" / "stage6_annotation"
OUTPUT_DIR = ROOT / "outputs" / "stage6_annotation_review"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

A_PATH = INPUT_DIR / "stage6_annotator_A.csv"
B_PATH = INPUT_DIR / "stage6_annotator_B.csv"

MERGED_PATH = OUTPUT_DIR / "stage6_merged_for_adjudication.csv"
DISAGREEMENTS_PATH = OUTPUT_DIR / "stage6_disagreements.csv"
SUMMARY_PATH = OUTPUT_DIR / "stage6_agreement_summary.json"

FIELDS = [
    "audit_id",
    "question_type",
    "question",
    "claim",
    "retrieved_evidence",
    "local_graph",
    "annotator_A_label",
    "annotator_A_rationale",
    "annotator_B_label",
    "annotator_B_rationale",
    "agreement",
    "adjudicated_label",
    "adjudication_note",
]


def main() -> int:
    for path in (A_PATH, B_PATH):
        if not path.exists():
            print(f"ERROR: Missing annotation packet: {path}")
            return 1

    a_rows = {row["audit_id"]: row for row in read_csv(A_PATH)}
    b_rows = {row["audit_id"]: row for row in read_csv(B_PATH)}
    if set(a_rows) != set(b_rows) or len(a_rows) != 48:
        print("ERROR: Annotation packets do not contain the same 48 audit IDs.")
        return 2

    merged = []
    invalid = []
    for audit_id in sorted(a_rows):
        a = a_rows[audit_id]
        b = b_rows[audit_id]
        a_label = a["annotator_label"].strip().lower()
        b_label = b["annotator_label"].strip().lower()
        if a_label not in ALLOWED_LABELS:
            invalid.append(f"{audit_id}: invalid Annotator A label {a_label!r}")
        if b_label not in ALLOWED_LABELS:
            invalid.append(f"{audit_id}: invalid Annotator B label {b_label!r}")
        agreement = a_label == b_label
        merged.append(
            {
                "audit_id": audit_id,
                "question_type": a["question_type"],
                "question": a["question"],
                "claim": a["claim"],
                "retrieved_evidence": a["retrieved_evidence"],
                "local_graph": a["local_graph"],
                "annotator_A_label": a_label,
                "annotator_A_rationale": a["annotator_rationale"].strip(),
                "annotator_B_label": b_label,
                "annotator_B_rationale": b["annotator_rationale"].strip(),
                "agreement": "yes" if agreement else "no",
                "adjudicated_label": a_label if agreement else "",
                "adjudication_note": (
                    "Automatic: annotators agreed." if agreement else ""
                ),
            }
        )

    if invalid:
        print("ERROR: Complete both packets using the permitted labels.")
        for item in invalid:
            print("-", item)
        return 3

    write_csv(MERGED_PATH, merged, FIELDS)
    disagreements = [row for row in merged if row["agreement"] == "no"]
    write_csv(DISAGREEMENTS_PATH, disagreements, FIELDS)

    labels_a = [row["annotator_A_label"] for row in merged]
    labels_b = [row["annotator_B_label"] for row in merged]
    agreement_count = sum(a == b for a, b in zip(labels_a, labels_b))
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_count": len(merged),
        "agreement_count": agreement_count,
        "disagreement_count": len(disagreements),
        "percent_agreement": agreement_count / len(merged),
        "cohen_kappa": cohen_kappa(labels_a, labels_b),
        "annotator_A_label_counts": dict(sorted(Counter(labels_a).items())),
        "annotator_B_label_counts": dict(sorted(Counter(labels_b).items())),
        "model_predictions_accessed": False,
        "status": (
            "complete_agreement"
            if not disagreements
            else "adjudication_required"
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Agreement: {agreement_count}/{len(merged)}")
    print(f"Disagreements: {len(disagreements)}")
    print(f"Summary: {SUMMARY_PATH}")
    if disagreements:
        print("Complete adjudicated_label and adjudication_note before evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
