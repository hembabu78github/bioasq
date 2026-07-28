from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage7_common import ROOT, load_jsonl

V3_DIR = ROOT / "outputs" / "stage7_graph_scope_v3"
V3_RESULTS = V3_DIR / "stage7_graph_scope_v3_results.jsonl"
V3_REPORT = V3_DIR / "stage7_graph_scope_v3_validation_report.json"

OUTPUT_DIR = ROOT / "outputs" / "stage7c_selective_answer_smoke"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_JSON = OUTPUT_DIR / "stage7c_answer_smoke_sample.json"
SAMPLE_CSV = OUTPUT_DIR / "stage7c_answer_smoke_sample.csv"
MANIFEST = OUTPUT_DIR / "stage7c_answer_smoke_manifest.json"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "arm_id",
        "paired_group_id",
        "stage7_id",
        "question_id",
        "question_type",
        "question",
        "route",
        "is_graph_route",
        "evidence_snippet_ids",
        "graph_route_eligible",
        "eligibility_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(row[key], ensure_ascii=False)
                        if isinstance(row.get(key), list)
                        else row.get(key)
                    )
                    for key in fields
                }
            )


def main() -> int:
    if not V3_RESULTS.exists() or not V3_REPORT.exists():
        print("ERROR: Missing Stage 7B V3 outputs.")
        print(f"- {V3_RESULTS}: {'OK' if V3_RESULTS.exists() else 'MISSING'}")
        print(f"- {V3_REPORT}: {'OK' if V3_REPORT.exists() else 'MISSING'}")
        return 1

    report = json.loads(V3_REPORT.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        print("ERROR: Stage 7B V3 did not pass.")
        return 1

    source_rows = load_jsonl(V3_RESULTS)
    source_rows.sort(key=lambda row: row["scope_v3_order"])

    eligible = [
        row for row in source_rows if row.get("graph_route_eligible")
    ]
    if len(eligible) != 2:
        raise RuntimeError(
            f"Expected exactly two eligible V3 routes, found {len(eligible)}."
        )
    if any(row["question_type"] != "list" for row in eligible):
        raise RuntimeError(
            "The frozen Stage 7C graph route is restricted to eligible list questions."
        )

    arms: list[dict[str, Any]] = []
    arm_index = 1
    for row in source_rows:
        common = {
            "paired_group_id": row["stage7_id"],
            "stage7_id": row["stage7_id"],
            "question_id": row["question_id"],
            "question_type": row["question_type"],
            "question": row["question"],
            "graph_route_eligible": bool(row["graph_route_eligible"]),
            "eligibility_source": row["eligibility_source"],
        }

        arms.append(
            {
                "arm_id": f"S7C-A{arm_index:02d}",
                **common,
                "route": "hybrid_baseline",
                "is_graph_route": False,
                "evidence_snippet_ids": row["hybrid_top5"],
                "graph": None,
            }
        )
        arm_index += 1

        if row["graph_route_eligible"]:
            arms.append(
                {
                    "arm_id": f"S7C-A{arm_index:02d}",
                    **common,
                    "route": "selective_graph",
                    "is_graph_route": True,
                    "evidence_snippet_ids": row["graph_selected_top5"],
                    "graph": row["graph"],
                }
            )
            arm_index += 1

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "sealed_test_accessed": False,
        "purpose": (
            "Verifier-driven paired answer smoke for the frozen selective "
            "multi-entity list GraphRAG route."
        ),
        "frozen_route_policy": (
            "Use GraphRAG only when question_type=list and Stage 7B V3 "
            "deterministic graph_route_eligible=true; otherwise use hybrid retrieval."
        ),
        "final_disposition_policy": (
            "Release an answer only when the generator did not abstain and every "
            "atomic claim is verifier-supported. Otherwise abstain."
        ),
        "arms": arms,
    }
    SAMPLE_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(SAMPLE_CSV, arms)

    manifest = {
        "generated_at_utc": payload["generated_at_utc"],
        "question_count": len(source_rows),
        "hybrid_baseline_answer_count": sum(
            arm["route"] == "hybrid_baseline" for arm in arms
        ),
        "selective_graph_answer_count": sum(
            arm["route"] == "selective_graph" for arm in arms
        ),
        "total_answer_count": len(arms),
        "paired_graph_question_ids": [
            row["stage7_id"] for row in eligible
        ],
        "graph_scope": "eligible multi-entity list questions only",
        "expected_generation_calls": len(arms),
        "expected_verifier_calls": len(arms),
        "new_graph_calls": 0,
        "sample_sha256": hashlib.sha256(
            SAMPLE_JSON.read_bytes()
        ).hexdigest(),
        "scientific_note": (
            "This is a development technical answer smoke, not a final performance "
            "estimate. The same-model verifier is not human gold."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Stage 7C answer-smoke sample prepared.")
    print(f"- Hybrid baselines: {manifest['hybrid_baseline_answer_count']}")
    print(f"- Graph-route answers: {manifest['selective_graph_answer_count']}")
    print(f"- Total generated answers: {manifest['total_answer_count']}")
    print(f"- Sample: {SAMPLE_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
