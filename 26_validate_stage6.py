from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from stage6_common import ALLOWED_LABELS, ROOT, load_jsonl

PREP_SUMMARY = ROOT / "outputs" / "stage6_prep" / "adversarial_generation_summary.json"
AGREEMENT = ROOT / "outputs" / "stage6_annotation_review" / "stage6_agreement_summary.json"
MERGED = ROOT / "outputs" / "stage6_annotation_review" / "stage6_merged_for_adjudication.csv"
COMPARISON = ROOT / "outputs" / "stage6_evaluation" / "stage6_verifier_comparison_summary.json"
PREDICTIONS = ROOT / "outputs" / "stage6_evaluation" / "stage6_verifier_predictions.jsonl"
DISCOVERY = ROOT / "outputs" / "stage6_evaluation" / "stage6_model_discovery.json"
COMPLETENESS = ROOT / "outputs" / "stage6_evaluation" / "stage6_output_completeness_report.json"
REPORT = ROOT / "outputs" / "stage6_evaluation" / "stage6_validation_report.json"


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolved_candidate_count(prep: dict) -> int | None:
    for key in ("corrected_candidate_count", "candidate_count"):
        value = prep.get(key)
        if isinstance(value, int):
            return value
    return None


def main() -> int:
    required = [
        PREP_SUMMARY, AGREEMENT, MERGED, COMPARISON,
        PREDICTIONS, DISCOVERY, COMPLETENESS,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("ERROR: Missing Stage 6 outputs:")
        for path in missing:
            print("-", path)
        return 1

    prep = json.loads(PREP_SUMMARY.read_text(encoding="utf-8"))
    agreement = json.loads(AGREEMENT.read_text(encoding="utf-8"))
    merged = read_csv(MERGED)
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    predictions = load_jsonl(PREDICTIONS)
    discovery = json.loads(DISCOVERY.read_text(encoding="utf-8"))
    completeness = json.loads(COMPLETENESS.read_text(encoding="utf-8"))

    final_labels = [
        row.get("adjudicated_label", "").strip()
        or row.get("annotator_A_label", "").strip()
        for row in merged
    ]
    configuration_counts = Counter(row["configuration"] for row in predictions)
    expected_configs = {
        "original_prompt_same_model",
        "hardened_prompt_same_model",
        "hardened_prompt_alternate_model",
    }
    candidate_count = resolved_candidate_count(prep)

    nonempty_rationales = sum(
        bool(str(row.get("model_rationale", "")).strip())
        for row in predictions
    )
    hardened_predictions = [
        row for row in predictions
        if row["configuration"].startswith("hardened_prompt")
    ]
    hardened_with_qualifiers = sum(
        isinstance(row.get("material_qualifiers_checked"), list)
        and bool(row.get("material_qualifiers_checked"))
        for row in hardened_predictions
    )

    checks = {
        "generated_48_adversarial_candidates": candidate_count == 48,
        "all_48_adversarial_claims_human_labelled": len(merged) == 48
        and all(label in ALLOWED_LABELS for label in final_labels),
        "human_agreement_summary_complete": agreement.get("claim_count") == 48,
        "combined_evaluation_has_76_claims": comparison.get("combined_claim_count") == 76,
        "three_verifier_configurations_present": set(comparison.get("configurations", {})) == expected_configs,
        "each_configuration_has_76_predictions": all(configuration_counts[name] == 76 for name in expected_configs),
        "alternate_model_differs_from_baseline": discovery.get("selected_alternate_model") != discovery.get("baseline_model"),
        "all_prediction_labels_valid": all(row["predicted_label"] in ALLOWED_LABELS for row in predictions),
        "all_228_predictions_have_rationales": nonempty_rationales == 228,
        "all_152_hardened_predictions_have_qualifier_checks": hardened_with_qualifiers == 152,
        "output_completeness_report_passed": completeness.get("status") == "pass",
        "sealed_test_not_accessed": comparison.get("sealed_test_accessed") is False,
        "selected_configuration_declared": comparison.get("selected_configuration") in expected_configs,
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "schema_resolution": {
            "candidate_count_source_field": (
                "corrected_candidate_count"
                if isinstance(prep.get("corrected_candidate_count"), int)
                else "candidate_count"
            ),
            "resolved_candidate_count": candidate_count,
        },
        "counts": {
            "adversarial_claims": len(merged),
            "combined_claims": comparison.get("combined_claim_count"),
            "predictions": len(predictions),
            "nonempty_rationales": nonempty_rationales,
            "hardened_predictions_with_qualifier_checks": hardened_with_qualifiers,
            "predictions_by_configuration": dict(sorted(configuration_counts.items())),
            "human_adversarial_label_distribution": dict(sorted(Counter(final_labels).items())),
        },
        "selected_configuration": comparison.get("selected_configuration"),
        "scientific_note": (
            "Passing validation confirms structural and semantic completeness "
            "of the development verifier comparison. It does not establish "
            "final test performance or clinical deployment validity."
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Stage 6 validation status: {report['status']}")
    for name, passed in checks.items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Report: {REPORT}")
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
