from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IN_DIR = ROOT / "outputs" / "stage5_annotation"
A_PATH = IN_DIR / "claim_audit_annotator_A.csv"
B_PATH = IN_DIR / "claim_audit_annotator_B.csv"
OUT_DIR = ROOT / "outputs" / "stage5_annotation_review"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED = ["supported", "contradicted", "insufficient_evidence"]


def read_packet(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        audit_id = row.get("audit_id", "").strip()
        label = row.get("annotator_label", "").strip().lower()
        rationale = row.get("annotator_rationale", "").strip()
        if not audit_id:
            raise ValueError(f"Missing audit_id in {path.name}")
        if label not in ALLOWED:
            raise ValueError(f"{path.name} {audit_id}: invalid or blank label '{label}'")
        if not rationale:
            raise ValueError(f"{path.name} {audit_id}: rationale is blank")
        result[audit_id] = {**row, "annotator_label": label, "annotator_rationale": rationale}
    return result


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    n = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    ca, cb = Counter(labels_a), Counter(labels_b)
    expected = sum((ca[label] / n) * (cb[label] / n) for label in ALLOWED)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def main() -> int:
    if not A_PATH.exists() or not B_PATH.exists():
        print("ERROR: Both completed annotator packets are required.")
        return 1

    a = read_packet(A_PATH)
    b = read_packet(B_PATH)
    if set(a) != set(b) or len(a) != 28:
        print("ERROR: Annotator packets do not contain the same 28 audit IDs.")
        return 2

    merged = []
    labels_a, labels_b = [], []
    confusion = {la: {lb: 0 for lb in ALLOWED} for la in ALLOWED}
    for audit_id in sorted(a):
        la, lb = a[audit_id]["annotator_label"], b[audit_id]["annotator_label"]
        labels_a.append(la); labels_b.append(lb)
        confusion[la][lb] += 1
        merged.append({
            "audit_id": audit_id,
            "question_type": a[audit_id].get("question_type", ""),
            "question": a[audit_id].get("question", ""),
            "claim": a[audit_id].get("claim", ""),
            "claim_importance": a[audit_id].get("claim_importance", ""),
            "retrieved_evidence": a[audit_id].get("retrieved_evidence", ""),
            "local_graph": a[audit_id].get("local_graph", ""),
            "annotator_A_label": la,
            "annotator_A_rationale": a[audit_id]["annotator_rationale"],
            "annotator_B_label": lb,
            "annotator_B_rationale": b[audit_id]["annotator_rationale"],
            "agreement": "yes" if la == lb else "no",
            "adjudicated_label": la if la == lb else "",
            "adjudication_note": "",
        })

    agreement_count = sum(row["agreement"] == "yes" for row in merged)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_count": 28,
        "agreement_count": agreement_count,
        "disagreement_count": 28 - agreement_count,
        "percent_agreement": round(agreement_count / 28, 6),
        "cohen_kappa": round(cohen_kappa(labels_a, labels_b), 6),
        "annotator_A_label_counts": dict(Counter(labels_a)),
        "annotator_B_label_counts": dict(Counter(labels_b)),
        "confusion_matrix_A_rows_B_columns": confusion,
        "model_predictions_accessed": False,
        "status": "ready_for_adjudication" if agreement_count < 28 else "complete_agreement",
    }
    (OUT_DIR / "claim_audit_agreement_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fields = list(merged[0].keys())
    with (OUT_DIR / "claim_audit_merged_for_adjudication.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(merged)

    disagreements = [row for row in merged if row["agreement"] == "no"]
    with (OUT_DIR / "claim_audit_disagreements.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(disagreements)

    print(json.dumps(summary, indent=2))
    print(f"Adjudication file: {OUT_DIR / 'claim_audit_merged_for_adjudication.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
