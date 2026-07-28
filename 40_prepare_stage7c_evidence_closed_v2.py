from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage7_common import ROOT, load_jsonl

V3_RESULTS = (
    ROOT / "outputs" / "stage7_graph_scope_v3"
    / "stage7_graph_scope_v3_results.jsonl"
)
V1_RESULTS = (
    ROOT / "outputs" / "stage7c_selective_answer_smoke"
    / "stage7c_answer_smoke_results.jsonl"
)

OUTPUT_DIR = ROOT / "outputs" / "stage7c_evidence_closed_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_JSON = OUTPUT_DIR / "stage7c_evidence_closed_v2_sample.json"
SAMPLE_CSV = OUTPUT_DIR / "stage7c_evidence_closed_v2_sample.csv"
MANIFEST = OUTPUT_DIR / "stage7c_evidence_closed_v2_manifest.json"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "arm_id",
        "stage7_id",
        "question_id",
        "question_type",
        "question",
        "route",
        "reuse_v1_result",
        "source_v1_arm_id",
        "evidence_snippet_ids",
        "graph_route_eligible",
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
    for path in (V3_RESULTS, V1_RESULTS):
        if not path.exists():
            print(f"ERROR: Missing input: {path}")
            return 1

    v3_rows = load_jsonl(V3_RESULTS)
    v3_rows.sort(key=lambda row: row["scope_v3_order"])
    v1_rows = load_jsonl(V1_RESULTS)

    v1_hybrid_by_stage = {
        row["stage7_id"]: row
        for row in v1_rows
        if row["route"] == "hybrid_baseline"
    }

    arms = []
    index = 1
    for row in v3_rows:
        hybrid = v1_hybrid_by_stage.get(row["stage7_id"])
        if not hybrid:
            raise RuntimeError(
                f"Missing reusable V1 hybrid result for {row['stage7_id']}."
            )

        arms.append(
            {
                "arm_id": f"S7C2-A{index:02d}",
                "stage7_id": row["stage7_id"],
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "question": row["question"],
                "route": "hybrid_baseline",
                "reuse_v1_result": True,
                "source_v1_arm_id": hybrid["arm_id"],
                "evidence_snippet_ids": row["hybrid_top5"],
                "graph_route_eligible": bool(
                    row["graph_route_eligible"]
                ),
            }
        )
        index += 1

        if row["graph_route_eligible"]:
            arms.append(
                {
                    "arm_id": f"S7C2-A{index:02d}",
                    "stage7_id": row["stage7_id"],
                    "question_id": row["question_id"],
                    "question_type": row["question_type"],
                    "question": row["question"],
                    "route": "graph_selected_text_only",
                    "reuse_v1_result": False,
                    "source_v1_arm_id": "",
                    "evidence_snippet_ids": row[
                        "graph_selected_top5"
                    ],
                    "graph_route_eligible": True,
                }
            )
            index += 1

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "sealed_test_accessed": False,
        "purpose": (
            "Evidence-closed Stage 7C rerun. The graph is used only to select "
            "five snippets; graph assertions are not supplied to the generator "
            "or verifier."
        ),
        "frozen_route_scope": (
            "Eligible multi-entity list questions only."
        ),
        "evidence_closure_rule": (
            "Generator and verifier receive exactly the same five displayed "
            "text snippets. No graph edges, answer aspects or hidden candidate "
            "snippets are supplied."
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
        "question_count": len(v3_rows),
        "answer_arm_count": len(arms),
        "reused_hybrid_answer_count": sum(
            arm["reuse_v1_result"] for arm in arms
        ),
        "new_graph_selected_answer_count": sum(
            arm["route"] == "graph_selected_text_only"
            for arm in arms
        ),
        "expected_new_generation_calls": 2,
        "expected_new_verifier_calls": 2,
        "graph_payload_calls": 0,
        "sample_sha256": hashlib.sha256(
            SAMPLE_JSON.read_bytes()
        ).hexdigest(),
        "scientific_note": (
            "This is a protocol-integrity correction, not answer-level tuning."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Stage 7C evidence-closed V2 sample prepared.")
    print(f"- Reused hybrid arms: {manifest['reused_hybrid_answer_count']}")
    print(f"- New graph-selected text-only arms: {manifest['new_graph_selected_answer_count']}")
    print(f"- Sample: {SAMPLE_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
