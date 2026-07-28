from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "stage3"
REPORT_PATH = OUTPUT_DIR / "stage3_validation_report.json"

SPLIT_PATH = OUTPUT_DIR / "split_manifest_v2.json"
DENSE_SUMMARY = OUTPUT_DIR / "dense_models_summary.json"
DENSE_RESULTS = OUTPUT_DIR / "dense_pilot_results.jsonl"
HYBRID_SUMMARY = OUTPUT_DIR / "hybrid_rrf_summary.json"
ROUTE_ANALYSIS = OUTPUT_DIR / "route_utility_analysis.json"
BY_MODEL_TYPE = OUTPUT_DIR / "dense_pilot_by_model_type.csv"
BY_HYBRID_TYPE = OUTPUT_DIR / "hybrid_rrf_by_type.csv"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    required = [
        SPLIT_PATH,
        DENSE_SUMMARY,
        DENSE_RESULTS,
        HYBRID_SUMMARY,
        ROUTE_ANALYSIS,
        BY_MODEL_TYPE,
        BY_HYBRID_TYPE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("ERROR: Missing Stage 3 outputs:")
        for path in missing:
            print(f"- {path}")
        return 1

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    dense_summary = json.loads(DENSE_SUMMARY.read_text(encoding="utf-8"))
    dense_rows = load_jsonl(DENSE_RESULTS)
    hybrid_summary = json.loads(HYBRID_SUMMARY.read_text(encoding="utf-8"))
    route = json.loads(ROUTE_ANALYSIS.read_text(encoding="utf-8"))

    model_keys = sorted(dense_summary.get("models", {}).keys())
    counts_by_model = Counter(row.get("model_key") for row in dense_rows)
    pilot_membership = split.get("pinned_pilot", {}).get("membership_after_repair", {})

    hybrid_methods = set(hybrid_summary.get("methods", {}).keys())
    required_methods = {
        "bm25",
        "bge_small",
        "pubmedbert",
        "hybrid_bge_small",
        "hybrid_pubmedbert",
    }

    checks = {
        "pilot_pinned_only_to_development": pilot_membership
        == {"train": 0, "dev": 80, "test": 0},
        "repaired_split_has_no_id_overlap": sum(
            split.get("id_overlap_counts", {}).values()
        )
        == 0,
        "repaired_split_has_no_duplicate_body_leakage": split.get(
            "duplicate_body_group_leakage_count"
        )
        == 0,
        "both_dense_models_completed": set(model_keys) == {"bge_small", "pubmedbert"},
        "each_dense_model_has_80_pilot_records": all(
            counts_by_model.get(model_key, 0) == 80 for model_key in model_keys
        ),
        "all_required_retrieval_methods_present": required_methods.issubset(
            hybrid_methods
        ),
        "route_analysis_is_explicitly_exploratory": "exploratory"
        in json.dumps(route).lower(),
        "no_test_metrics_in_dense_summary": "test" not in dense_summary.get(
            "stage", ""
        ).lower(),
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "counts": {
            "dense_rows": len(dense_rows),
            "dense_rows_by_model": dict(counts_by_model),
            "hybrid_method_count": len(hybrid_methods),
            "repaired_split_counts": split.get("counts"),
        },
        "scientific_note": (
            "Passing Stage 3 validates the repaired development/test boundary and the "
            "development-only lexical, dense and hybrid retrieval comparison. It does not "
            "validate graph reasoning, agentic routing, claim verification, abstention or "
            "final test performance."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Stage 3 validation status: {report['status']}")
    for name, passed in checks.items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Report saved: {REPORT_PATH}")
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
