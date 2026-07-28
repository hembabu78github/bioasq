from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage7_common import (
    ALLOWED_LABELS,
    MODEL,
    ROOT,
    call_json,
    format_evidence,
    frozen_hardened_verifier_messages,
    groq_client,
    load_checkpoint,
    load_jsonl,
    normalize_label,
    save_checkpoint,
    verification_items,
    write_jsonl,
)

SAMPLE = (
    ROOT / "outputs" / "stage7c_evidence_closed_v2"
    / "stage7c_evidence_closed_v2_sample.json"
)
V1_RESULTS = (
    ROOT / "outputs" / "stage7c_selective_answer_smoke"
    / "stage7c_answer_smoke_results.jsonl"
)
CORPUS = (
    ROOT / "data" / "processed" / "stage2"
    / "candidate_snippets.jsonl"
)

OUTPUT_DIR = ROOT / "outputs" / "stage7c_evidence_closed_v2"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_JSONL = OUTPUT_DIR / "stage7c_evidence_closed_v2_results.jsonl"
RESULTS_CSV = OUTPUT_DIR / "stage7c_evidence_closed_v2_results.csv"
SUMMARY = OUTPUT_DIR / "stage7c_evidence_closed_v2_summary.json"
PAIRS = OUTPUT_DIR / "stage7c_evidence_closed_v2_paired_comparison.json"


def evidence_rows(
    snippet_ids: list[str],
    corpus: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for snippet_id in snippet_ids:
        if snippet_id in seen or snippet_id not in corpus:
            continue
        seen.add(snippet_id)
        source = corpus[snippet_id]
        rows.append(
            {
                "evidence_id": f"E{len(rows) + 1}",
                "snippet_id": snippet_id,
                "document_ids": source.get("document_ids", []),
                "text": source["text"],
            }
        )
        if len(rows) >= 5:
            break
    if len(rows) != 5:
        raise RuntimeError(
            f"Expected five evidence snippets, resolved {len(rows)}."
        )
    return rows


def generator_messages(
    question_type: str,
    question: str,
    evidence: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer the BioASQ biomedical question using only the five "
                "supplied text evidence snippets. Do not use external knowledge "
                "and do not infer facts from any graph or hidden candidate pool. "
                "Return one JSON object with answer, abstain, abstention_reason, "
                "and claims. claims must contain atomic factual claims with "
                "claim_id and text. For list questions, use one claim per proposed "
                "answer item. If a material qualifier is not directly supported, "
                "do not include the item. Set abstain=true when no reliable answer "
                "can be formed."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question type: {question_type}\n"
                f"Question: {question}\n\n"
                f"Evidence:\n{format_evidence(evidence)}"
            ),
        },
    ]


def normalize_answer(parsed: dict[str, Any]) -> dict[str, Any]:
    answer = str(parsed.get("answer", "")).strip()
    abstain = bool(parsed.get("abstain"))
    reason = str(parsed.get("abstention_reason", "")).strip()

    claims = []
    for item in parsed.get("claims", []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        claims.append(
            {
                "claim_id": f"C{len(claims) + 1}",
                "text": text,
            }
        )

    if not answer:
        raise ValueError("Generator returned an empty answer.")
    if not abstain and not claims:
        raise ValueError(
            "Non-abstaining answer returned no atomic claims."
        )
    return {
        "answer": answer,
        "abstain": abstain,
        "abstention_reason": reason,
        "claims": claims,
    }


def checkpoint_valid_answer(value: dict[str, Any] | None) -> bool:
    if not value:
        return False
    answer = value.get("answer_record", {})
    return bool(str(answer.get("answer", "")).strip()) and (
        bool(answer.get("abstain")) or bool(answer.get("claims"))
    )


def verification_complete(
    value: dict[str, Any] | None,
    claims: list[dict[str, str]],
) -> bool:
    if not value:
        return False
    items = value.get("verifications", [])
    if len(items) != len(claims):
        return False
    expected = {claim["claim_id"] for claim in claims}
    if {
        str(item.get("claim_id", "")) for item in items
    } != expected:
        return False
    allowed_evidence = {"E1", "E2", "E3", "E4", "E5"}
    return all(
        item.get("status") in ALLOWED_LABELS
        and str(item.get("brief_rationale", "")).strip()
        and isinstance(
            item.get("material_qualifiers_checked"), list
        )
        and item.get("material_qualifiers_checked")
        and set(item.get("evidence_ids", [])) <= allowed_evidence
        and not item.get("graph_edge_ids")
        for item in items
    )


def complete_verifications(
    client,
    question: str,
    evidence: list[dict[str, Any]],
    claims: list[dict[str, str]],
    call_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not claims:
        return [], None

    expected = {claim["claim_id"] for claim in claims}
    allowed_evidence = {"E1", "E2", "E3", "E4", "E5"}
    messages = frozen_hardened_verifier_messages(
        question,
        format_evidence(evidence),
        (
            "No graph is supplied. Verify only against the displayed text "
            "evidence E1-E5. graph_edge_ids must be an empty list."
        ),
        claims,
    )
    messages.append(
        {
            "role": "user",
            "content": (
                "Output constraint: evidence_ids may contain only E1, E2, E3, "
                "E4 or E5. graph_edge_ids must be []."
            ),
        }
    )

    last_issues = []
    for semantic_attempt in range(1, 4):
        parsed, call = call_json(
            client,
            messages,
            f"{call_name}_attempt_{semantic_attempt}",
            max_tokens=min(3200, 900 + 320 * len(claims)),
        )
        by_id = {}
        duplicates = set()
        for item in verification_items(parsed):
            if not isinstance(item, dict):
                continue
            claim_id = str(item.get("claim_id", "")).strip()
            if claim_id in by_id:
                duplicates.add(claim_id)
            by_id[claim_id] = item

        issues = []
        if duplicates:
            issues.append(
                "duplicate IDs: " + ", ".join(sorted(duplicates))
            )
        if set(by_id) != expected:
            issues.append(
                "expected IDs "
                + ", ".join(sorted(expected))
                + "; received "
                + ", ".join(sorted(by_id))
            )

        normalized = []
        for claim in claims:
            item = by_id.get(claim["claim_id"], {})
            status = normalize_label(item.get("status"))
            rationale = str(
                item.get("brief_rationale", "")
            ).strip()
            qualifiers = item.get(
                "material_qualifiers_checked", []
            )
            evidence_ids = item.get("evidence_ids", [])
            graph_edge_ids = item.get("graph_edge_ids", [])

            if not status:
                issues.append(
                    f"{claim['claim_id']}: invalid status"
                )
            if not rationale:
                issues.append(
                    f"{claim['claim_id']}: empty rationale"
                )
            if not isinstance(qualifiers, list) or not qualifiers:
                issues.append(
                    f"{claim['claim_id']}: missing qualifier checks"
                )
            if (
                not isinstance(evidence_ids, list)
                or not set(evidence_ids) <= allowed_evidence
            ):
                issues.append(
                    f"{claim['claim_id']}: invalid evidence IDs"
                )
            if graph_edge_ids:
                issues.append(
                    f"{claim['claim_id']}: graph edge IDs are not allowed"
                )

            normalized.append(
                {
                    "claim_id": claim["claim_id"],
                    "claim_text": claim["text"],
                    "status": status,
                    "evidence_ids": evidence_ids,
                    "graph_edge_ids": [],
                    "unsupported_or_contradicted_span": item.get(
                        "unsupported_or_contradicted_span", ""
                    ),
                    "material_qualifiers_checked": qualifiers,
                    "brief_rationale": rationale,
                }
            )

        if not issues:
            return normalized, call

        last_issues = issues
        messages = list(messages) + [
            {
                "role": "user",
                "content": (
                    "Correct the JSON. Return exactly one verification per "
                    "claim. Use only evidence IDs E1-E5 and no graph edge IDs. "
                    "Problems: " + "; ".join(issues)
                ),
            }
        ]

    raise RuntimeError(
        f"{call_name} remained incomplete: "
        + "; ".join(last_issues)
    )


def final_disposition(
    generator_abstained: bool,
    claims: list[dict[str, str]],
    verifications: list[dict[str, Any]],
    answer: str,
) -> dict[str, str]:
    labels = [item["status"] for item in verifications]
    if (
        not generator_abstained
        and claims
        and labels
        and all(label == "supported" for label in labels)
    ):
        return {
            "final_disposition": "release",
            "disposition_reason": "all_atomic_claims_supported",
            "final_answer": answer,
        }
    return {
        "final_disposition": "abstain",
        "disposition_reason": (
            "generator_abstained"
            if generator_abstained
            else "one_or_more_claims_not_supported"
        ),
        "final_answer": (
            "Insufficient supported evidence to provide a reliable answer."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "arm_id",
        "stage7_id",
        "question_type",
        "route",
        "reuse_v1_result",
        "claim_count",
        "supported_claim_count",
        "contradicted_claim_count",
        "insufficient_claim_count",
        "final_disposition",
        "answer",
        "final_answer",
        "evidence_snippet_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            counts = Counter(
                item["status"] for item in row["verifications"]
            )
            writer.writerow(
                {
                    "arm_id": row["arm_id"],
                    "stage7_id": row["stage7_id"],
                    "question_type": row["question_type"],
                    "route": row["route"],
                    "reuse_v1_result": row["reuse_v1_result"],
                    "claim_count": len(row["claims"]),
                    "supported_claim_count": counts["supported"],
                    "contradicted_claim_count": counts["contradicted"],
                    "insufficient_claim_count": counts[
                        "insufficient_evidence"
                    ],
                    "final_disposition": row[
                        "final_disposition"
                    ],
                    "answer": row["answer"],
                    "final_answer": row["final_answer"],
                    "evidence_snippet_ids": json.dumps(
                        row["evidence_snippet_ids"]
                    ),
                }
            )


def main() -> int:
    for path in (SAMPLE, V1_RESULTS, CORPUS):
        if not path.exists():
            print(f"ERROR: Missing input: {path}")
            return 1

    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    arms = sample["arms"]
    v1_rows = load_jsonl(V1_RESULTS)
    v1_by_arm = {row["arm_id"]: row for row in v1_rows}
    corpus = {row["snippet_id"]: row for row in load_jsonl(CORPUS)}
    client = None

    results = []
    for index, arm in enumerate(arms, start=1):
        print(
            f"{index}/{len(arms)} {arm['arm_id']} "
            f"{arm['stage7_id']} {arm['route']}"
        )
        evidence = evidence_rows(
            arm["evidence_snippet_ids"], corpus
        )

        if arm["reuse_v1_result"]:
            source = v1_by_arm[arm["source_v1_arm_id"]]
            print("  result: REUSE Stage 7C V1 hybrid baseline")
            result = {
                "run_utc": datetime.now(timezone.utc).isoformat(),
                **arm,
                "model": source["model"],
                "evidence": evidence,
                "graph_supplied_to_generator": False,
                "graph_supplied_to_verifier": False,
                "generator_abstained": source[
                    "generator_abstained"
                ],
                "generator_abstention_reason": source[
                    "generator_abstention_reason"
                ],
                "answer": source["answer"],
                "claims": source["claims"],
                "verifications": source["verifications"],
                "final_disposition": source[
                    "final_disposition"
                ],
                "disposition_reason": source[
                    "disposition_reason"
                ],
                "final_answer": source["final_answer"],
                "answer_call": source.get("answer_call"),
                "verification_call": source.get(
                    "verification_call"
                ),
            }
            results.append(result)
            continue

        if client is None:
            client = groq_client()

        answer_path = (
            CHECKPOINT_DIR / "answers" / f"{arm['arm_id']}.json"
        )
        answer_checkpoint = load_checkpoint(answer_path)
        if checkpoint_valid_answer(answer_checkpoint):
            answer_record = answer_checkpoint["answer_record"]
            answer_call = answer_checkpoint.get("call")
            print("  answer: SKIP checkpoint")
        else:
            parsed, answer_call = call_json(
                client,
                generator_messages(
                    arm["question_type"],
                    arm["question"],
                    evidence,
                ),
                f"stage7c2_answer_{arm['arm_id']}",
                max_tokens=1600,
            )
            answer_record = normalize_answer(parsed)
            save_checkpoint(
                answer_path,
                {
                    "answer_record": answer_record,
                    "call": answer_call,
                    "evidence_snippet_ids": arm[
                        "evidence_snippet_ids"
                    ],
                },
            )

        claims = answer_record["claims"]
        verify_path = (
            CHECKPOINT_DIR
            / "verifications"
            / f"{arm['arm_id']}.json"
        )
        verify_checkpoint = load_checkpoint(verify_path)
        if verification_complete(verify_checkpoint, claims):
            verifications = verify_checkpoint["verifications"]
            verification_call = verify_checkpoint.get("call")
            print("  verifier: SKIP checkpoint")
        else:
            verifications, verification_call = complete_verifications(
                client,
                arm["question"],
                evidence,
                claims,
                f"stage7c2_verify_{arm['arm_id']}",
            )
            save_checkpoint(
                verify_path,
                {
                    "verifications": verifications,
                    "call": verification_call,
                },
            )

        disposition = final_disposition(
            answer_record["abstain"],
            claims,
            verifications,
            answer_record["answer"],
        )

        results.append(
            {
                "run_utc": datetime.now(timezone.utc).isoformat(),
                **arm,
                "model": MODEL,
                "evidence": evidence,
                "graph_supplied_to_generator": False,
                "graph_supplied_to_verifier": False,
                "generator_abstained": answer_record["abstain"],
                "generator_abstention_reason": answer_record[
                    "abstention_reason"
                ],
                "answer": answer_record["answer"],
                "claims": claims,
                "verifications": verifications,
                **disposition,
                "answer_call": answer_call,
                "verification_call": verification_call,
            }
        )

    results.sort(key=lambda row: row["arm_id"])
    write_jsonl(RESULTS_JSONL, results)
    write_csv(RESULTS_CSV, results)

    by_route = {}
    for route in ("hybrid_baseline", "graph_selected_text_only"):
        route_rows = [row for row in results if row["route"] == route]
        labels = [
            item["status"]
            for row in route_rows
            for item in row["verifications"]
        ]
        by_route[route] = {
            "answer_count": len(route_rows),
            "claim_count": len(labels),
            "verifier_label_counts": dict(
                sorted(Counter(labels).items())
            ),
            "final_release_count": sum(
                row["final_disposition"] == "release"
                for row in route_rows
            ),
            "final_abstention_count": sum(
                row["final_disposition"] == "abstain"
                for row in route_rows
            ),
            "mean_claim_count": (
                statistics.mean(
                    len(row["claims"]) for row in route_rows
                )
                if route_rows
                else 0
            ),
        }

    grouped = defaultdict(dict)
    for row in results:
        grouped[row["stage7_id"]][row["route"]] = row

    pairs = []
    for stage7_id, routes in grouped.items():
        if "graph_selected_text_only" not in routes:
            continue
        hybrid = routes["hybrid_baseline"]
        graph = routes["graph_selected_text_only"]
        pairs.append(
            {
                "stage7_id": stage7_id,
                "question": hybrid["question"],
                "hybrid_answer": hybrid["answer"],
                "graph_selected_text_answer": graph["answer"],
                "answer_text_changed": (
                    hybrid["answer"] != graph["answer"]
                ),
                "hybrid_verifier_label_counts": dict(
                    sorted(
                        Counter(
                            item["status"]
                            for item in hybrid["verifications"]
                        ).items()
                    )
                ),
                "graph_selected_text_verifier_label_counts": dict(
                    sorted(
                        Counter(
                            item["status"]
                            for item in graph["verifications"]
                        ).items()
                    )
                ),
                "hybrid_final_disposition": hybrid[
                    "final_disposition"
                ],
                "graph_selected_text_final_disposition": graph[
                    "final_disposition"
                ],
                "evidence_set_changed": (
                    set(hybrid["evidence_snippet_ids"])
                    != set(graph["evidence_snippet_ids"])
                ),
            }
        )

    PAIRS.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "paired_question_count": len(pairs),
                "pairs": pairs,
                "scientific_note": (
                    "Graph is used only for evidence selection. Automated "
                    "same-model verifier results remain descriptive."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "sealed_test_accessed": False,
        "question_count": len(
            {row["stage7_id"] for row in results}
        ),
        "answer_count": len(results),
        "reused_hybrid_answer_count": sum(
            row["reuse_v1_result"] for row in results
        ),
        "new_graph_selected_answer_count": sum(
            row["route"] == "graph_selected_text_only"
            for row in results
        ),
        "by_route": by_route,
        "graph_payload_supplied_count": sum(
            row["graph_supplied_to_generator"]
            or row["graph_supplied_to_verifier"]
            for row in results
        ),
        "unsupported_claims_released": sum(
            row["final_disposition"] == "release"
            and any(
                item["status"] != "supported"
                for item in row["verifications"]
            )
            for row in results
        ),
        "scientific_note": (
            "Evidence-closed protocol correction: graph assertions are never "
            "presented as evidence."
        ),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Results: {RESULTS_JSONL}")
    print(f"Summary: {SUMMARY}")
    print(f"Pairs: {PAIRS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
