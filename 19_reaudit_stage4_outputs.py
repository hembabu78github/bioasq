
from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "outputs" / "stage4" / "graph_claim_pilot_results.jsonl"
OUTPUT_DIR = ROOT / "outputs" / "stage4_correction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORRECTED_SUMMARY = OUTPUT_DIR / "graph_claim_pilot_summary_corrected.json"
CORRECTED_BY_TYPE = OUTPUT_DIR / "graph_claim_pilot_by_type_corrected.csv"
REAUDIT_PATH = OUTPUT_DIR / "graph_claim_pilot_reaudit.json"
AUDIT_CSV = OUTPUT_DIR / "claim_manual_audit_blinded.csv"
AUDIT_KEY = OUTPUT_DIR / "claim_manual_audit_model_key_private.jsonl"
VALIDATION_PATH = OUTPUT_DIR / "stage4_correction_validation_report.json"

ALLOWED_STATUS = {"supported", "contradicted", "insufficient_evidence"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def raw_verification_items(record: dict[str, Any], variant: str) -> list[dict[str, Any]]:
    """
    Recover the verifier's original output from the stored raw response.

    Two early responses used `claims` instead of the requested `verifications`
    key. The Stage 4 sanitizer treated those valid items as missing. This
    function accepts either key while preserving the model's original labels.
    """
    verification_calls = [
        call for call in record.get("calls", [])
        if call.get("call_name") == "claim_verification"
    ]
    if not verification_calls:
        return []

    raw = verification_calls[-1].get("raw_response", "")
    try:
        parsed = json.loads(raw)
    except Exception:
        return []

    variant_obj = parsed.get(variant, {})
    if not isinstance(variant_obj, dict):
        return []

    items = variant_obj.get("verifications")
    if not isinstance(items, list):
        items = variant_obj.get("claims")
    return items if isinstance(items, list) else []


def corrected_verifications(
    record: dict[str, Any],
    variant: str,
) -> list[dict[str, Any]]:
    claims = record["answers"][variant]["claims"]
    claim_ids = {str(claim["claim_id"]) for claim in claims}
    valid_evidence_ids = {item["evidence_id"] for item in record["evidence"]}
    valid_edge_ids = {item["edge_id"] for item in record["graph"]["relations"]}

    by_claim: dict[str, dict[str, Any]] = {}
    for item in raw_verification_items(record, variant):
        if not isinstance(item, dict):
            continue
        claim_id = str(item.get("claim_id", ""))
        if claim_id not in claim_ids:
            continue
        status = str(item.get("status", "")).lower()
        if status not in ALLOWED_STATUS:
            status = "insufficient_evidence"
        by_claim[claim_id] = {
            "claim_id": claim_id,
            "status": status,
            "evidence_ids": sorted(
                {
                    str(x)
                    for x in item.get("evidence_ids", [])
                    if str(x) in valid_evidence_ids
                }
            ),
            "graph_edge_ids": (
                []
                if variant == "text_only"
                else sorted(
                    {
                        str(x)
                        for x in item.get("graph_edge_ids", [])
                        if str(x) in valid_edge_ids
                    }
                )
            ),
            "brief_rationale": str(item.get("brief_rationale", "")).strip(),
        }

    return [
        by_claim.get(
            str(claim["claim_id"]),
            {
                "claim_id": str(claim["claim_id"]),
                "status": "insufficient_evidence",
                "evidence_ids": [],
                "graph_edge_ids": [],
                "brief_rationale": (
                    "The verifier did not return a valid record for this claim."
                ),
            },
        )
        for claim in claims
    ]


def adjudicate(
    answer: dict[str, Any],
    verification_items: list[dict[str, Any]],
) -> dict[str, Any]:
    claim_by_id = {str(c["claim_id"]): c for c in answer["claims"]}
    supported = [x for x in verification_items if x["status"] == "supported"]
    contradicted = [x for x in verification_items if x["status"] == "contradicted"]
    critical_ids = {
        str(c["claim_id"])
        for c in answer["claims"]
        if c.get("importance") == "critical"
    }
    supported_ids = {x["claim_id"] for x in supported}
    contradicted_ids = {x["claim_id"] for x in contradicted}
    ratio = len(supported) / max(len(verification_items), 1)

    abstain = (
        not bool(answer.get("answerable"))
        or bool(critical_ids & contradicted_ids)
        or (bool(critical_ids) and not bool(critical_ids & supported_ids))
        or ratio < 0.5
    )

    return {
        "abstain": abstain,
        "supported_claim_ratio": round(ratio, 6),
        "supported_claim_count": len(supported),
        "contradicted_claim_count": len(contradicted),
        "insufficient_claim_count": sum(
            x["status"] == "insufficient_evidence" for x in verification_items
        ),
        "final_supported_claims": [
            claim_by_id[x["claim_id"]]["text"]
            for x in supported
            if x["claim_id"] in claim_by_id
        ],
    }


def variant_summary(
    corrected_records: list[dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    statuses = Counter()
    ratios = []
    abstentions = 0
    claim_count = 0

    for record in corrected_records:
        items = record["corrected_verifications"][variant]
        statuses.update(x["status"] for x in items)
        adjudication = record["corrected_adjudication"][variant]
        ratios.append(adjudication["supported_claim_ratio"])
        abstentions += int(adjudication["abstain"])
        claim_count += len(record["answers"][variant]["claims"])

    return {
        "question_count": len(corrected_records),
        "claim_count": claim_count,
        "verification_status_counts": dict(sorted(statuses.items())),
        "abstention_count": abstentions,
        "abstention_rate": round(abstentions / max(len(corrected_records), 1), 6),
        "mean_supported_claim_ratio": (
            round(statistics.mean(ratios), 6) if ratios else None
        ),
    }


def evidence_text(record: dict[str, Any]) -> str:
    return "\n\n".join(
        f"[{e['evidence_id']}] {e['text']}" for e in record["evidence"]
    )


def graph_text(record: dict[str, Any]) -> str:
    if not record["graph"]["relations"]:
        return "NO_SUPPORTED_GRAPH_EDGE"
    return "\n".join(
        (
            f"[{edge['edge_id']}] {edge['source_entity_id']} "
            f"--{edge['relation_type']}--> {edge['target_entity_id']} | "
            f"Evidence: {','.join(edge['evidence_ids'])} | "
            f"Quote: {edge.get('evidence_quote', '')}"
        )
        for edge in record["graph"]["relations"]
    )


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"ERROR: Missing Stage 4 result file: {INPUT_PATH}")
        return 1

    records = load_jsonl(INPUT_PATH)
    corrected = []

    schema_alias_questions = []
    incomplete_verification_questions = []
    identical_answer_count = 0
    questions_with_graph_claim_reference = 0

    for record in records:
        corrected_ver = {
            variant: corrected_verifications(record, variant)
            for variant in ("text_only", "graph_assisted")
        }
        corrected_adj = {
            variant: adjudicate(record["answers"][variant], corrected_ver[variant])
            for variant in ("text_only", "graph_assisted")
        }

        # Audit raw schema.
        for variant in ("text_only", "graph_assisted"):
            raw_items = raw_verification_items(record, variant)
            raw_call = [
                call for call in record["calls"]
                if call.get("call_name") == "claim_verification"
            ][-1]
            try:
                raw_parsed = json.loads(raw_call["raw_response"])
                variant_obj = raw_parsed.get(variant, {})
            except Exception:
                variant_obj = {}
            if isinstance(variant_obj, dict):
                if "claims" in variant_obj and "verifications" not in variant_obj:
                    schema_alias_questions.append(
                        {"question_id": record["question_id"], "variant": variant}
                    )
            if len(raw_items) < len(record["answers"][variant]["claims"]):
                incomplete_verification_questions.append(
                    {
                        "question_id": record["question_id"],
                        "variant": variant,
                        "claim_count": len(record["answers"][variant]["claims"]),
                        "returned_verification_count": len(raw_items),
                    }
                )

        identical = (
            record["answers"]["text_only"]["answer"].strip()
            == record["answers"]["graph_assisted"]["answer"].strip()
        )
        identical_answer_count += int(identical)
        graph_refs = sum(
            len(claim.get("graph_edge_ids", []))
            for claim in record["answers"]["graph_assisted"]["claims"]
        )
        questions_with_graph_claim_reference += int(graph_refs > 0)

        corrected.append(
            {
                **record,
                "corrected_verifications": corrected_ver,
                "corrected_adjudication": corrected_adj,
                "text_and_graph_answers_identical": identical,
            }
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "question_count": len(corrected),
        "text_only": variant_summary(corrected, "text_only"),
        "graph_assisted": variant_summary(corrected, "graph_assisted"),
        "graph_utility_audit": {
            "identical_text_and_graph_answer_count": identical_answer_count,
            "identical_text_and_graph_answer_rate": round(
                identical_answer_count / max(len(corrected), 1), 6
            ),
            "questions_with_at_least_one_graph_edge": sum(
                len(r["graph"]["relations"]) > 0 for r in corrected
            ),
            "questions_with_graph_edge_referenced_by_generated_claim": (
                questions_with_graph_claim_reference
            ),
            "questions_with_graph_answer_support_ratio_above_text": sum(
                r["corrected_adjudication"]["graph_assisted"][
                    "supported_claim_ratio"
                ]
                > r["corrected_adjudication"]["text_only"]["supported_claim_ratio"]
                for r in corrected
            ),
            "questions_with_graph_answer_support_ratio_below_text": sum(
                r["corrected_adjudication"]["graph_assisted"][
                    "supported_claim_ratio"
                ]
                < r["corrected_adjudication"]["text_only"]["supported_claim_ratio"]
                for r in corrected
            ),
        },
        "interpretation": (
            "Corrected development feasibility summary. Model-generated verification "
            "labels are not gold labels and may not be reported as verification accuracy."
        ),
    }
    CORRECTED_SUMMARY.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    with CORRECTED_BY_TYPE.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "question_type",
            "question_count",
            "text_only_abstention_rate",
            "text_only_mean_supported_claim_ratio",
            "graph_assisted_abstention_rate",
            "graph_assisted_mean_supported_claim_ratio",
            "identical_answer_rate",
            "mean_graph_entities",
            "mean_graph_relations",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for qtype in sorted({r["question_type"] for r in corrected}):
            subset = [r for r in corrected if r["question_type"] == qtype]
            writer.writerow(
                {
                    "question_type": qtype,
                    "question_count": len(subset),
                    "text_only_abstention_rate": sum(
                        r["corrected_adjudication"]["text_only"]["abstain"]
                        for r in subset
                    )
                    / len(subset),
                    "text_only_mean_supported_claim_ratio": statistics.mean(
                        r["corrected_adjudication"]["text_only"][
                            "supported_claim_ratio"
                        ]
                        for r in subset
                    ),
                    "graph_assisted_abstention_rate": sum(
                        r["corrected_adjudication"]["graph_assisted"]["abstain"]
                        for r in subset
                    )
                    / len(subset),
                    "graph_assisted_mean_supported_claim_ratio": statistics.mean(
                        r["corrected_adjudication"]["graph_assisted"][
                            "supported_claim_ratio"
                        ]
                        for r in subset
                    ),
                    "identical_answer_rate": sum(
                        r["text_and_graph_answers_identical"] for r in subset
                    )
                    / len(subset),
                    "mean_graph_entities": statistics.mean(
                        len(r["graph"]["entities"]) for r in subset
                    ),
                    "mean_graph_relations": statistics.mean(
                        len(r["graph"]["relations"]) for r in subset
                    ),
                }
            )

    reaudit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_record_count": len(records),
        "schema_alias_question_variants": schema_alias_questions,
        "incomplete_verification_question_variants": (
            incomplete_verification_questions
        ),
        "selection_issue": (
            "All 12 selected questions were high-uncertainty, retrieval-hard cases and "
            "all used hybrid_bge. The feasibility pilot therefore does not compare risk "
            "strata or agent routes."
        ),
        "graph_generation_issue": (
            "Text-only and graph-assisted final answer strings were identical for all "
            f"{identical_answer_count} of {len(corrected)} questions. The current pilot "
            "does not demonstrate that graph reasoning changed answer generation."
        ),
        "verifier_independence_issue": (
            "The same Groq model family generated and verified the claims. Independent "
            "manual labels and preferably a distinct verifier model are required."
        ),
        "corrected_summary_relative_path": str(CORRECTED_SUMMARY.relative_to(ROOT)),
    }
    REAUDIT_PATH.write_text(json.dumps(reaudit, indent=2), encoding="utf-8")

    # Create a blinded, deduplicated claim-audit sheet.
    audit_rows = []
    private_key_rows = []
    audit_id = 0
    seen = set()

    for record in corrected:
        for variant in ("text_only", "graph_assisted"):
            model_ver = {
                x["claim_id"]: x
                for x in record["corrected_verifications"][variant]
            }
            for claim in record["answers"][variant]["claims"]:
                key = (
                    record["question_id"],
                    claim["text"].strip().lower(),
                )
                if key in seen:
                    continue
                seen.add(key)
                audit_id += 1
                audit_code = f"AUD-{audit_id:03d}"

                audit_rows.append(
                    {
                        "audit_id": audit_code,
                        "question_type": record["question_type"],
                        "question": record["question"],
                        "claim": claim["text"],
                        "claim_importance": claim["importance"],
                        "retrieved_evidence": evidence_text(record),
                        "local_graph": graph_text(record),
                        "annotator_1_label": "",
                        "annotator_1_rationale": "",
                        "annotator_2_label": "",
                        "annotator_2_rationale": "",
                        "adjudicated_label": "",
                        "adjudication_note": "",
                    }
                )

                model_item = model_ver.get(claim["claim_id"], {})
                private_key_rows.append(
                    {
                        "audit_id": audit_code,
                        "question_id": record["question_id"],
                        "source_variant": variant,
                        "model_predicted_label": model_item.get("status"),
                        "model_rationale": model_item.get("brief_rationale"),
                    }
                )

    with AUDIT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)

    with AUDIT_KEY.open("w", encoding="utf-8") as handle:
        for row in private_key_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    checks = {
        "input_contains_12_questions": len(records) == 12,
        "corrected_summary_created": CORRECTED_SUMMARY.exists(),
        "schema_alias_recovered": len(schema_alias_questions) == 4,
        "all_claims_present_in_manual_audit": len(audit_rows)
        == len(
            {
                (r["question_id"], c["text"].strip().lower())
                for r in records
                for variant in ("text_only", "graph_assisted")
                for c in r["answers"][variant]["claims"]
            }
        ),
        "manual_audit_is_blinded": all(
            "model_predicted" not in key for key in audit_rows[0].keys()
        ),
        "private_model_key_created": AUDIT_KEY.exists(),
    }
    validation = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "corrected_output_files": [
            str(CORRECTED_SUMMARY.relative_to(ROOT)),
            str(CORRECTED_BY_TYPE.relative_to(ROOT)),
            str(REAUDIT_PATH.relative_to(ROOT)),
            str(AUDIT_CSV.relative_to(ROOT)),
            str(AUDIT_KEY.relative_to(ROOT)),
        ],
        "scientific_note": (
            "This correction repairs output parsing and prepares a blinded manual audit. "
            "It does not alter the stored model generations or create new performance data."
        ),
    }
    VALIDATION_PATH.write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )

    print("Stage 4 correction status:", validation["status"])
    print("Original summary text-only:", {
        "supported": 21, "insufficient": 7, "abstentions": 3
    })
    print("Corrected text-only:", summary["text_only"])
    print("Original summary graph-assisted:", {
        "supported": 18, "insufficient": 10, "abstentions": 4
    })
    print("Corrected graph-assisted:", summary["graph_assisted"])
    print(f"Blinded manual-audit claims: {len(audit_rows)}")
    print(f"Outputs saved under: {OUTPUT_DIR}")
    return 0 if validation["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
