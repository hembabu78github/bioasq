from __future__ import annotations

import csv
import json
import os
import statistics
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from stage6_common import (
    ALLOWED_LABELS,
    BASELINE_MODEL,
    ROOT,
    call_json,
    final_human_label,
    groq_client,
    metrics_from_pairs,
    normalize_label,
    read_csv,
    write_csv,
    write_jsonl,
)

ORIGINAL_AUDIT = (
    ROOT / "outputs" / "stage4_correction" / "claim_manual_audit_blinded.csv"
)
ORIGINAL_REFERENCE = (
    ROOT / "outputs" / "stage5_annotation_review"
    / "claim_audit_merged_for_adjudication.csv"
)
ADVERSARIAL_PRIVATE = (
    ROOT / "outputs" / "stage6_prep"
    / "adversarial_claim_candidates_flat_private.jsonl"
)
ADVERSARIAL_REFERENCE = (
    ROOT / "outputs" / "stage6_annotation_review"
    / "stage6_merged_for_adjudication.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "stage6_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = OUTPUT_DIR / "stage6_verifier_predictions.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "stage6_verifier_comparison_summary.json"
BY_CLASS_PATH = OUTPUT_DIR / "stage6_verifier_by_class.csv"
ERRORS_PATH = OUTPUT_DIR / "stage6_verifier_error_cases.csv"
DISCOVERY_PATH = OUTPUT_DIR / "stage6_model_discovery.json"
COMPLETENESS_PATH = OUTPUT_DIR / "stage6_output_completeness_report.json"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

PREFERRED_ALTERNATES = [
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "moonshotai/kimi-k2-instruct",
]


def load_jsonl_local(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def discover_alternate(client) -> tuple[str, dict[str, Any]]:
    load_dotenv(ROOT / ".env")
    forced = os.getenv("STAGE6_ALT_VERIFIER_MODEL", "").strip()
    response = client.models.list()
    model_ids = sorted(
        {
            str(getattr(model, "id", "")).strip()
            for model in getattr(response, "data", [])
            if str(getattr(model, "id", "")).strip()
        }
    )

    excluded_terms = (
        "whisper",
        "distil-whisper",
        "tts",
        "guard",
        "safety",
        "prompt-guard",
    )
    chat_candidates = [
        model_id
        for model_id in model_ids
        if model_id != BASELINE_MODEL
        and not any(term in model_id.lower() for term in excluded_terms)
    ]

    if forced:
        if forced == BASELINE_MODEL:
            raise RuntimeError(
                "STAGE6_ALT_VERIFIER_MODEL must differ from the baseline model."
            )
        if forced not in model_ids:
            raise RuntimeError(
                f"Forced alternate model {forced!r} was not returned by Groq."
            )
        selected = forced
        source = "environment_override"
    else:
        selected = next(
            (model for model in PREFERRED_ALTERNATES if model in chat_candidates),
            None,
        )
        if selected is None and chat_candidates:
            selected = chat_candidates[0]
        if selected is None:
            raise RuntimeError("No alternate Groq chat model was available.")
        source = "runtime_discovery"

    discovery = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_model": BASELINE_MODEL,
        "selected_alternate_model": selected,
        "selection_source": source,
        "preferred_alternates": PREFERRED_ALTERNATES,
        "available_model_ids": model_ids,
        "eligible_alternate_candidates": chat_candidates,
    }
    return selected, discovery


def original_prompt(
    question: str,
    evidence: str,
    graph: str,
    claims: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Verify every biomedical claim using only the supplied evidence and "
                "evidence-derived graph. Return JSON with a `verifications` list. "
                "Each item must contain claim_id, status, evidence_ids, graph_edge_ids "
                "and brief_rationale. status must be exactly supported, contradicted, "
                "or insufficient_evidence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\nEvidence:\n{evidence}\n\n"
                f"Graph:\n{graph}\n\nClaims:\n"
                f"{json.dumps(claims, ensure_ascii=False)}"
            ),
        },
    ]


def hardened_prompt(
    question: str,
    evidence: str,
    graph: str,
    claims: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Act as a strict claim-level evidence verifier. Use only the displayed "
                "retrieved evidence and evidence-derived graph. Return JSON with one "
                "`verifications` list. Every input claim_id must appear exactly once.\n\n"
                "Allowed status values:\n"
                "- supported\n"
                "- contradicted\n"
                "- insufficient_evidence\n\n"
                "A claim is supported only when every material component is directly "
                "supported. Separately inspect:\n"
                "- named entities and entity identity;\n"
                "- relation direction;\n"
                "- negation;\n"
                "- date, timing and sequence;\n"
                "- numbers, units and quantities;\n"
                "- population, disease and intervention scope;\n"
                "- comparative or superlative language;\n"
                "- regulatory status;\n"
                "- commercial or market availability;\n"
                "- association versus causation;\n"
                "- investigated versus effective;\n"
                "- treatment versus cure.\n\n"
                "Do not use plausible background knowledge. Do not make inferential "
                "bridges. In particular:\n"
                "- FDA approval does not prove market availability;\n"
                "- being studied does not prove effectiveness;\n"
                "- association does not prove causation;\n"
                "- evidence from one year does not prove a different stated year;\n"
                "- support for a broad relation does not support a stronger superlative.\n\n"
                "Use contradicted only when the displayed evidence directly supports the "
                "opposite or makes the claim incompatible. Use insufficient_evidence when "
                "a material qualifier is absent, ambiguous or only inferable.\n\n"
                "Each verification item must contain:\n"
                "- claim_id\n"
                "- status\n"
                "- evidence_ids\n"
                "- graph_edge_ids\n"
                "- unsupported_or_contradicted_span\n"
                "- material_qualifiers_checked\n"
                "- brief_rationale\n\nKeep each rationale to 20 words or fewer and keep "
"material_qualifiers_checked as a compact list."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\nEvidence:\n{evidence}\n\n"
                f"Graph:\n{graph}\n\nClaims:\n"
                f"{json.dumps(claims, ensure_ascii=False)}"
            ),
        },
    ]


def build_evaluation_claims() -> list[dict[str, Any]]:
    original_audit = {row["audit_id"]: row for row in read_csv(ORIGINAL_AUDIT)}
    original_reference_rows = read_csv(ORIGINAL_REFERENCE)
    adversarial_private = {
        row["audit_id"]: row for row in load_jsonl_local(ADVERSARIAL_PRIVATE)
    }
    adversarial_reference_rows = read_csv(ADVERSARIAL_REFERENCE)

    rows = []
    for ref in original_reference_rows:
        label = final_human_label(ref)
        if label not in ALLOWED_LABELS:
            raise RuntimeError(
                f"Original audit {ref['audit_id']} has no locked human label."
            )
        source = original_audit[ref["audit_id"]]
        rows.append(
            {
                "audit_id": ref["audit_id"],
                "set": "original",
                "source_question_id": hashlib.sha256(
                    " ".join(source["question"].split()).casefold().encode("utf-8")
                ).hexdigest()[:16],
                "question_type": source["question_type"],
                "question": source["question"],
                "claim": source["claim"],
                "evidence": source["retrieved_evidence"],
                "graph": source["local_graph"],
                "human_label": label,
                "perturbation_type": "none",
            }
        )

    for ref in adversarial_reference_rows:
        label = final_human_label(ref)
        if label not in ALLOWED_LABELS:
            raise RuntimeError(
                f"Adversarial audit {ref['audit_id']} has no locked human label."
            )
        private = adversarial_private[ref["audit_id"]]
        rows.append(
            {
                "audit_id": ref["audit_id"],
                "set": "adversarial",
                "source_question_id": private["source_question_id"],
                "question_type": ref["question_type"],
                "question": ref["question"],
                "claim": ref["claim"],
                "evidence": ref["retrieved_evidence"],
                "graph": ref["local_graph"],
                "human_label": label,
                "perturbation_type": private["perturbation_type"],
            }
        )

    rows.sort(key=lambda row: (row["source_question_id"], row["audit_id"]))
    if len(rows) != 76:
        raise RuntimeError(f"Expected 76 combined claims, found {len(rows)}.")
    return rows


def verification_items(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    items = parsed.get("verifications")
    if not isinstance(items, list):
        items = parsed.get("claims")
    return items if isinstance(items, list) else []


def checkpoint_path(configuration: str) -> Path:
    return CHECKPOINT_DIR / f"{configuration}.jsonl"


def group_fingerprint(
    question_id: str,
    evidence: str,
    graph: str,
) -> str:
    payload = "\0".join((question_id, evidence, graph))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def checkpoint_record_issues(
    record: dict[str, Any],
    prompt_kind: str,
) -> list[str]:
    issues: list[str] = []
    audit_ids = [str(x) for x in record.get("audit_ids", [])]
    predictions = record.get("predictions", [])

    if not isinstance(predictions, list):
        return ["predictions_not_a_list"]

    prediction_ids = [str(row.get("audit_id", "")) for row in predictions]
    if len(predictions) != len(audit_ids):
        issues.append("prediction_count_mismatch")
    if len(set(prediction_ids)) != len(prediction_ids):
        issues.append("duplicate_prediction_ids")
    if set(prediction_ids) != set(audit_ids):
        issues.append("prediction_id_set_mismatch")

    for row in predictions:
        audit_id = str(row.get("audit_id", "unknown"))
        if normalize_label(row.get("predicted_label")) not in ALLOWED_LABELS:
            issues.append(f"{audit_id}:invalid_label")
        if not str(row.get("model_rationale", "")).strip():
            issues.append(f"{audit_id}:empty_rationale")
        if prompt_kind == "hardened":
            qualifiers = row.get("material_qualifiers_checked", [])
            if not isinstance(qualifiers, list) or not qualifiers:
                issues.append(f"{audit_id}:missing_qualifier_checks")

    return issues


def load_configuration_checkpoints(
    configuration: str,
    model: str,
    prompt_kind: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    path = checkpoint_path(configuration)
    completed: dict[str, dict[str, Any]] = {}
    invalidated: list[dict[str, Any]] = []
    if not path.exists():
        return completed, invalidated

    # Last record for a group wins, which allows a repaired checkpoint to
    # supersede an earlier truncated checkpoint without deleting audit history.
    latest: dict[str, tuple[int, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("configuration") != configuration:
                raise RuntimeError(
                    f"Checkpoint configuration mismatch at {path}:{line_number}."
                )
            if record.get("model") != model:
                raise RuntimeError(
                    f"Checkpoint model {record.get('model')!r} differs from "
                    f"current model {model!r}. Delete {path} or set "
                    "STAGE6_ALT_VERIFIER_MODEL to the checkpoint model."
                )
            if record.get("prompt_kind") != prompt_kind:
                raise RuntimeError(
                    f"Checkpoint prompt mismatch at {path}:{line_number}."
                )
            latest[record["group_id"]] = (line_number, record)

    for group_id, (line_number, record) in latest.items():
        issues = checkpoint_record_issues(record, prompt_kind)
        if issues:
            invalidated.append(
                {
                    "configuration": configuration,
                    "group_id": group_id,
                    "group_index": record.get("group_index"),
                    "audit_ids": record.get("audit_ids", []),
                    "checkpoint_line": line_number,
                    "issues": issues,
                }
            )
        else:
            completed[group_id] = record

    return completed, invalidated


def append_configuration_checkpoint(
    configuration: str,
    record: dict[str, Any],
) -> None:
    path = checkpoint_path(configuration)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def parsed_items_by_id(
    parsed: dict[str, Any],
    expected_ids: set[str],
    prompt_kind: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    items = verification_items(parsed)
    by_id: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()

    for item in items:
        if not isinstance(item, dict):
            continue
        claim_id = str(item.get("claim_id", ""))
        if claim_id in by_id:
            duplicates.add(claim_id)
        by_id[claim_id] = item

    issues: list[str] = []
    if duplicates:
        issues.append(
            "duplicate_claim_ids:" + ",".join(sorted(duplicates))
        )
    missing = sorted(expected_ids - set(by_id))
    extra = sorted(set(by_id) - expected_ids)
    if missing:
        issues.append("missing_claim_ids:" + ",".join(missing))
    if extra:
        issues.append("unexpected_claim_ids:" + ",".join(extra))

    for claim_id in sorted(expected_ids & set(by_id)):
        item = by_id[claim_id]
        if normalize_label(item.get("status")) not in ALLOWED_LABELS:
            issues.append(f"{claim_id}:invalid_status")
        if not str(item.get("brief_rationale", "")).strip():
            issues.append(f"{claim_id}:empty_rationale")
        if prompt_kind == "hardened":
            qualifiers = item.get("material_qualifiers_checked", [])
            if not isinstance(qualifiers, list) or not qualifiers:
                issues.append(f"{claim_id}:missing_qualifier_checks")

    return by_id, issues


def completeness_retry_messages(
    messages: list[dict[str, str]],
    expected_ids: set[str],
    issues: list[str],
) -> list[dict[str, str]]:
    retry_messages = [dict(message) for message in messages]
    retry_messages[0] = {
        "role": "system",
        "content": (
            retry_messages[0]["content"]
            + "\n\nOUTPUT COMPLETENESS IS MANDATORY. Return exactly one verification "
              "object for every claim ID listed by the user. Do not omit claims. "
              "Use concise rationales and compact JSON."
        ),
    }
    retry_messages.append(
        {
            "role": "user",
            "content": (
                "The prior response was incomplete. Required claim IDs: "
                + ", ".join(sorted(expected_ids))
                + ". Detected issues: "
                + "; ".join(issues)
                + ". Return the complete JSON object now."
            ),
        }
    )
    return retry_messages


def run_configuration(
    client,
    configuration: str,
    model: str,
    prompt_kind: str,
    claims: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in claims:
        key = (row["source_question_id"], row["evidence"], row["graph"])
        grouped[key].append(row)

    completed, invalidated = load_configuration_checkpoints(
        configuration, model, prompt_kind
    )
    if completed:
        print(
            f"{configuration}: found {len(completed)} complete group "
            "checkpoint(s)."
        )
    if invalidated:
        print(
            f"{configuration}: invalidating {len(invalidated)} incomplete "
            "group checkpoint(s)."
        )
        for item in invalidated:
            print(
                f"  - group {item.get('group_index')}: "
                f"{len(item.get('audit_ids', []))} claims; "
                f"{len(item.get('issues', []))} completeness issue(s)"
            )

    predictions: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    repaired_groups: list[dict[str, Any]] = []
    ordered_groups = sorted(grouped.items(), key=lambda item: item[0][0])

    for group_index, ((question_id, evidence, graph), group_rows) in enumerate(
        ordered_groups,
        start=1,
    ):
        group_id = group_fingerprint(question_id, evidence, graph)
        cached = completed.get(group_id)
        if cached is not None:
            print(
                f"{configuration}: SKIP complete group "
                f"{group_index}/{len(ordered_groups)} "
                f"({len(group_rows)} claims)"
            )
            predictions.extend(cached["predictions"])
            calls.append(cached["call"])
            continue

        question = group_rows[0]["question"]
        claim_payload = [
            {"claim_id": row["audit_id"], "text": row["claim"]}
            for row in group_rows
        ]
        base_messages = (
            original_prompt(question, evidence, graph, claim_payload)
            if prompt_kind == "original"
            else hardened_prompt(question, evidence, graph, claim_payload)
        )
        expected_ids = {row["audit_id"] for row in group_rows}

        print(
            f"{configuration}: REPAIR group "
            f"{group_index}/{len(ordered_groups)} "
            f"({len(group_rows)} claims)"
        )

        # The earlier cap of 1,500 tokens truncated several multi-claim JSON
        # responses. The repair allowance scales more conservatively.
        max_tokens = min(
            4800,
            max(1200, 650 + 300 * len(group_rows)),
        )
        parsed = None
        call = None
        by_id: dict[str, dict[str, Any]] = {}
        semantic_issues: list[str] = []
        messages = base_messages

        for semantic_attempt in range(1, 4):
            parsed, call = call_json(
                client,
                model,
                messages,
                (
                    f"{configuration}_{question_id}"
                    f"_completeness_attempt_{semantic_attempt}"
                ),
                max_tokens=max_tokens,
            )
            by_id, semantic_issues = parsed_items_by_id(
                parsed, expected_ids, prompt_kind
            )
            if not semantic_issues:
                break

            print(
                f"{configuration}: group {group_index} response incomplete "
                f"on attempt {semantic_attempt}: "
                + "; ".join(semantic_issues)
            )
            if semantic_attempt == 3:
                raise RuntimeError(
                    f"{configuration} group {group_index} remained incomplete "
                    f"after three semantic retries: {semantic_issues}"
                )

            max_tokens = min(6000, int(max_tokens * 1.35) + 250)
            messages = completeness_retry_messages(
                base_messages, expected_ids, semantic_issues
            )

        group_predictions = []
        for row in group_rows:
            item = by_id[row["audit_id"]]
            prediction = normalize_label(item.get("status"))
            group_predictions.append(
                {
                    **row,
                    "configuration": configuration,
                    "model": model,
                    "predicted_label": prediction,
                    "evidence_ids": item.get("evidence_ids", []),
                    "graph_edge_ids": item.get("graph_edge_ids", []),
                    "unsupported_or_contradicted_span": item.get(
                        "unsupported_or_contradicted_span", ""
                    ),
                    "material_qualifiers_checked": item.get(
                        "material_qualifiers_checked", []
                    ),
                    "model_rationale": str(
                        item.get("brief_rationale", "")
                    ).strip(),
                }
            )

        checkpoint_record = {
            "configuration": configuration,
            "model": model,
            "prompt_kind": prompt_kind,
            "group_id": group_id,
            "group_index": group_index,
            "group_count": len(ordered_groups),
            "question_id": question_id,
            "audit_ids": sorted(expected_ids),
            "dynamic_max_tokens": max_tokens,
            "completeness_repair": True,
            "semantic_issues_before_success": semantic_issues,
            "predictions": group_predictions,
            "call": call,
        }
        append_configuration_checkpoint(
            configuration, checkpoint_record
        )

        repaired_groups.append(
            {
                "configuration": configuration,
                "group_id": group_id,
                "group_index": group_index,
                "claim_count": len(group_rows),
                "audit_ids": sorted(expected_ids),
                "max_tokens_used": max_tokens,
            }
        )
        predictions.extend(group_predictions)
        calls.append(call)

    if len(predictions) != len(claims):
        raise RuntimeError(
            f"{configuration}: expected {len(claims)} predictions, "
            f"collected {len(predictions)}."
        )
    if len({row["audit_id"] for row in predictions}) != len(claims):
        raise RuntimeError(
            f"{configuration}: duplicate or missing audit IDs in predictions."
        )

    final_issues = []
    for row in predictions:
        if not str(row.get("model_rationale", "")).strip():
            final_issues.append(f"{row['audit_id']}:empty_rationale")
        if prompt_kind == "hardened":
            qualifiers = row.get("material_qualifiers_checked", [])
            if not isinstance(qualifiers, list) or not qualifiers:
                final_issues.append(
                    f"{row['audit_id']}:missing_qualifier_checks"
                )
    if final_issues:
        raise RuntimeError(
            f"{configuration}: final output completeness failed: "
            + "; ".join(final_issues)
        )

    predictions.sort(key=lambda row: row["audit_id"])
    return predictions, calls, repaired_groups


def aggregate_numeric_usage(calls: list[dict[str, Any]]) -> dict[str, float]:
    totals: Counter[str] = Counter()
    for call in calls:
        usage = call.get("usage") or {}
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] += value
    return dict(totals)

def selection_tuple(summary: dict[str, Any]) -> tuple:
    metrics = summary["metrics"]
    insufficient = metrics["per_class"]["insufficient_evidence"]["recall"]
    contradicted = metrics["per_class"]["contradicted"]["recall"]
    false_support = metrics[
        "false_support_rate_among_supported_predictions"
    ]
    macro_f1 = metrics["macro_f1_observed_classes"]
    latency = summary["mean_call_latency_ms"]
    return (
        -(insufficient if insufficient is not None else -1),
        false_support if false_support is not None else 1,
        -(contradicted if contradicted is not None else -1),
        -(macro_f1 if macro_f1 is not None else -1),
        latency if latency is not None else float("inf"),
    )


def main() -> int:
    for path in (
        ORIGINAL_AUDIT,
        ORIGINAL_REFERENCE,
        ADVERSARIAL_PRIVATE,
        ADVERSARIAL_REFERENCE,
    ):
        if not path.exists():
            print(f"ERROR: Missing required file: {path}")
            return 1

    client = groq_client()
    alternate_model, discovery = discover_alternate(client)
    DISCOVERY_PATH.write_text(json.dumps(discovery, indent=2), encoding="utf-8")
    print(f"Alternate verifier model: {alternate_model}")

    claims = build_evaluation_claims()
    configurations = [
        ("original_prompt_same_model", BASELINE_MODEL, "original"),
        ("hardened_prompt_same_model", BASELINE_MODEL, "hardened"),
        ("hardened_prompt_alternate_model", alternate_model, "hardened"),
    ]

    all_predictions = []
    configuration_summaries = {}
    all_calls = {}
    repaired_groups_by_configuration = {}

    for name, model, prompt_kind in configurations:
        predictions, calls, repaired_groups = run_configuration(
            client, name, model, prompt_kind, claims
        )
        repaired_groups_by_configuration[name] = repaired_groups
        all_predictions.extend(predictions)
        all_calls[name] = calls

        actual = [row["human_label"] for row in predictions]
        predicted = [row["predicted_label"] for row in predictions]
        metrics = metrics_from_pairs(actual, predicted)
        configuration_summaries[name] = {
            "model": model,
            "prompt_kind": prompt_kind,
            "metrics": metrics,
            "mean_call_latency_ms": (
                statistics.mean(call["latency_ms"] for call in calls)
                if calls
                else None
            ),
            "call_count": len(calls),
            "output_completeness": {
                "prediction_count": len(predictions),
                "nonempty_rationale_count": sum(
                    bool(str(row.get("model_rationale", "")).strip())
                    for row in predictions
                ),
                "hardened_nonempty_qualifier_check_count": (
                    sum(
                        isinstance(
                            row.get("material_qualifiers_checked"), list
                        )
                        and bool(row.get("material_qualifiers_checked"))
                        for row in predictions
                    )
                    if prompt_kind == "hardened"
                    else None
                ),
                "repaired_group_count": len(repaired_groups),
            },
            "token_usage_totals": aggregate_numeric_usage(calls),
        }

    ranked = sorted(
        configuration_summaries,
        key=lambda name: selection_tuple(configuration_summaries[name]),
    )
    selected = ranked[0]

    write_jsonl(PREDICTIONS_PATH, all_predictions)

    by_class_rows = []
    for name in configuration_summaries:
        for label, values in configuration_summaries[name]["metrics"][
            "per_class"
        ].items():
            by_class_rows.append(
                {
                    "configuration": name,
                    "model": configuration_summaries[name]["model"],
                    "label": label,
                    **values,
                }
            )
    write_csv(
        BY_CLASS_PATH,
        by_class_rows,
        [
            "configuration",
            "model",
            "label",
            "support",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
        ],
    )

    errors = [
        {
            "configuration": row["configuration"],
            "model": row["model"],
            "audit_id": row["audit_id"],
            "set": row["set"],
            "question_type": row["question_type"],
            "question": row["question"],
            "claim": row["claim"],
            "human_label": row["human_label"],
            "predicted_label": row["predicted_label"],
            "perturbation_type": row["perturbation_type"],
            "unsupported_or_contradicted_span": row[
                "unsupported_or_contradicted_span"
            ],
            "model_rationale": row["model_rationale"],
        }
        for row in all_predictions
        if row["human_label"] != row["predicted_label"]
    ]
    write_csv(
        ERRORS_PATH,
        errors,
        [
            "configuration",
            "model",
            "audit_id",
            "set",
            "question_type",
            "question",
            "claim",
            "human_label",
            "predicted_label",
            "perturbation_type",
            "unsupported_or_contradicted_span",
            "model_rationale",
        ],
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "sealed_test_accessed": False,
        "combined_claim_count": len(claims),
        "claim_distribution": dict(
            sorted(Counter(row["human_label"] for row in claims).items())
        ),
        "configurations": configuration_summaries,
        "selection_rule": [
            "highest insufficient_evidence recall",
            "lowest false-support rate",
            "highest contradicted recall",
            "highest macro F1 across observed classes",
            "lowest mean latency",
        ],
        "ranking_best_to_worst": ranked,
        "selected_configuration": selected,
        "selected_model": configuration_summaries[selected]["model"],
        "scientific_note": (
            "Verifier selection is development-only. Freeze the selected prompt, model "
            "and thresholds before the one-time sealed test evaluation."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    completeness_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "criteria": {
            "all_configurations_have_76_predictions": all(
                configuration_summaries[name]["output_completeness"][
                    "prediction_count"
                ] == 76
                for name in configuration_summaries
            ),
            "all_predictions_have_nonempty_rationales": all(
                configuration_summaries[name]["output_completeness"][
                    "nonempty_rationale_count"
                ] == 76
                for name in configuration_summaries
            ),
            "all_hardened_predictions_have_qualifier_checks": all(
                configuration_summaries[name]["output_completeness"][
                    "hardened_nonempty_qualifier_check_count"
                ] == 76
                for name in (
                    "hardened_prompt_same_model",
                    "hardened_prompt_alternate_model",
                )
            ),
        },
        "repaired_groups_by_configuration": repaired_groups_by_configuration,
        "scientific_note": (
            "The initial Stage 6 run silently converted omitted verifier items "
            "to insufficient_evidence. This report confirms that the repaired "
            "outputs contain one complete, reasoned verification per claim."
        ),
    }
    completeness_report["status"] = (
        "pass"
        if all(completeness_report["criteria"].values())
        else "fail"
    )
    COMPLETENESS_PATH.write_text(
        json.dumps(completeness_report, indent=2),
        encoding="utf-8",
    )
    if completeness_report["status"] != "pass":
        raise RuntimeError(
            "Stage 6 output completeness report failed."
        )

    print("Verifier ranking:")
    for rank, name in enumerate(ranked, start=1):
        print(f"{rank}. {name}")
    print(f"Selected: {selected}")
    print(f"Summary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
