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
    ROOT / "outputs" / "stage7c_selective_answer_smoke"
    / "stage7c_answer_smoke_sample.json"
)
CORPUS = (
    ROOT / "data" / "processed" / "stage2"
    / "candidate_snippets.jsonl"
)
OUTPUT_DIR = ROOT / "outputs" / "stage7c_selective_answer_smoke"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

RESULTS_JSONL = OUTPUT_DIR / "stage7c_answer_smoke_results.jsonl"
RESULTS_CSV = OUTPUT_DIR / "stage7c_answer_smoke_results.csv"
SUMMARY = OUTPUT_DIR / "stage7c_answer_smoke_summary.json"
PAIRS = OUTPUT_DIR / "stage7c_paired_comparison.json"


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


def compact_graph_text(graph: dict[str, Any] | None) -> str:
    if not graph:
        return "No graph supplied. Use the retrieved evidence directly."
    compact = {
        "entities": graph.get("entities", []),
        "relations": graph.get("relations", []),
        "answer_aspects": graph.get("answer_aspects", []),
        "model_graph_sufficient": graph.get(
            "model_graph_sufficient",
            graph.get("graph_sufficient"),
        ),
    }
    return json.dumps(compact, ensure_ascii=False)


def generator_messages(
    question_type: str,
    question: str,
    evidence: list[dict[str, Any]],
    graph_text: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer the BioASQ biomedical question using only the supplied "
                "evidence and, when present, the evidence-derived graph. Do not use "
                "external knowledge. Return one JSON object with answer, abstain, "
                "abstention_reason, and claims. claims must be a list of atomic "
                "factual claims, each with claim_id and text. For a list question, "
                "use one atomic claim per proposed answer item. For a summary "
                "question, use one atomic claim per material mechanism, function or "
                "qualifier. If the evidence cannot support a reliable answer, set "
                "abstain=true. Keep the answer concise and evidence-bounded."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question type: {question_type}\n"
                f"Question: {question}\n\n"
                f"Evidence:\n{format_evidence(evidence)}\n\n"
                f"Evidence-derived graph:\n{graph_text}"
            ),
        },
    ]


def normalize_answer(parsed: dict[str, Any]) -> dict[str, Any]:
    answer = str(parsed.get("answer", "")).strip()
    abstain = bool(parsed.get("abstain"))
    reason = str(parsed.get("abstention_reason", "")).strip()

    claims = []
    raw_claims = parsed.get("claims", [])
    if isinstance(raw_claims, list):
        for index, item in enumerate(raw_claims, start=1):
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
    if not claims:
        return items == []
    if len(items) != len(claims):
        return False
    expected = {claim["claim_id"] for claim in claims}
    received = {
        str(item.get("claim_id", "")) for item in items
    }
    if expected != received:
        return False
    return all(
        item.get("status") in ALLOWED_LABELS
        and str(item.get("brief_rationale", "")).strip()
        and isinstance(
            item.get("material_qualifiers_checked"), list
        )
        and item.get("material_qualifiers_checked")
        for item in items
    )


def complete_verifications(
    client,
    question: str,
    evidence: list[dict[str, Any]],
    graph_text: str,
    claims: list[dict[str, str]],
    call_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not claims:
        return [], None

    expected = {claim["claim_id"] for claim in claims}
    messages = frozen_hardened_verifier_messages(
        question,
        format_evidence(evidence),
        graph_text,
        claims,
    )
    last_issues = []

    for semantic_attempt in range(1, 4):
        parsed, call = call_json(
            client,
            messages,
            f"{call_name}_attempt_{semantic_attempt}",
            max_tokens=min(3400, 900 + 320 * len(claims)),
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

            normalized.append(
                {
                    "claim_id": claim["claim_id"],
                    "claim_text": claim["text"],
                    "status": status,
                    "evidence_ids": item.get("evidence_ids", []),
                    "graph_edge_ids": item.get(
                        "graph_edge_ids", []
                    ),
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
                    "The previous JSON was incomplete. Return exactly one "
                    "complete verification for every required claim ID. "
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
) -> dict[str, Any]:
    labels = [item["status"] for item in verifications]
    if generator_abstained:
        return {
            "final_disposition": "abstain",
            "disposition_reason": "generator_abstained",
            "final_answer": (
                "Insufficient supported evidence to provide a reliable answer."
            ),
        }
    if not claims:
        return {
            "final_disposition": "abstain",
            "disposition_reason": "no_atomic_claims",
            "final_answer": (
                "Insufficient supported evidence to provide a reliable answer."
            ),
        }
    if labels and all(label == "supported" for label in labels):
        return {
            "final_disposition": "release",
            "disposition_reason": "all_atomic_claims_supported",
            "final_answer": answer,
        }
    return {
        "final_disposition": "abstain",
        "disposition_reason": "one_or_more_claims_not_supported",
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
        "is_graph_route",
        "generator_abstained",
        "claim_count",
        "supported_claim_count",
        "contradicted_claim_count",
        "insufficient_claim_count",
        "final_disposition",
        "disposition_reason",
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
                    "is_graph_route": row["is_graph_route"],
                    "generator_abstained": row["generator_abstained"],
                    "claim_count": len(row["claims"]),
                    "supported_claim_count": counts["supported"],
                    "contradicted_claim_count": counts["contradicted"],
                    "insufficient_claim_count": counts[
                        "insufficient_evidence"
                    ],
                    "final_disposition": row[
                        "final_disposition"
                    ],
                    "disposition_reason": row[
                        "disposition_reason"
                    ],
                    "answer": row["answer"],
                    "final_answer": row["final_answer"],
                    "evidence_snippet_ids": json.dumps(
                        row["evidence_snippet_ids"]
                    ),
                }
            )


def main() -> int:
    if not SAMPLE.exists() or not CORPUS.exists():
        print("ERROR: Missing Stage 7C inputs.")
        for path in (SAMPLE, CORPUS):
            print(f"- {path}: {'OK' if path.exists() else 'MISSING'}")
        return 1

    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    arms = sample["arms"]
    corpus = {row["snippet_id"]: row for row in load_jsonl(CORPUS)}
    client = groq_client()

    results = []
    for index, arm in enumerate(arms, start=1):
        print(
            f"{index}/{len(arms)} {arm['arm_id']} "
            f"{arm['stage7_id']} {arm['route']}"
        )
        evidence = evidence_rows(
            arm["evidence_snippet_ids"], corpus
        )
        graph_text = compact_graph_text(arm.get("graph"))

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
                    graph_text,
                ),
                f"stage7c_answer_{arm['arm_id']}",
                max_tokens=1700,
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
                    "route": arm["route"],
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
                graph_text,
                claims,
                f"stage7c_verify_{arm['arm_id']}",
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
    for route in ("hybrid_baseline", "selective_graph"):
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
            "generator_abstention_count": sum(
                row["generator_abstained"]
                for row in route_rows
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

    pair_rows = []
    for stage7_id, routes in grouped.items():
        if "selective_graph" not in routes:
            continue
        hybrid = routes["hybrid_baseline"]
        graph = routes["selective_graph"]
        hybrid_labels = Counter(
            item["status"] for item in hybrid["verifications"]
        )
        graph_labels = Counter(
            item["status"] for item in graph["verifications"]
        )
        pair_rows.append(
            {
                "stage7_id": stage7_id,
                "question": hybrid["question"],
                "hybrid_answer": hybrid["answer"],
                "graph_answer": graph["answer"],
                "answer_text_changed": (
                    hybrid["answer"] != graph["answer"]
                ),
                "hybrid_claim_count": len(hybrid["claims"]),
                "graph_claim_count": len(graph["claims"]),
                "hybrid_verifier_label_counts": dict(
                    sorted(hybrid_labels.items())
                ),
                "graph_verifier_label_counts": dict(
                    sorted(graph_labels.items())
                ),
                "hybrid_final_disposition": hybrid[
                    "final_disposition"
                ],
                "graph_final_disposition": graph[
                    "final_disposition"
                ],
                "graph_evidence_changed": (
                    set(hybrid["evidence_snippet_ids"])
                    != set(graph["evidence_snippet_ids"])
                ),
            }
        )

    pair_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paired_question_count": len(pair_rows),
        "pairs": pair_rows,
        "scientific_note": (
            "These automated same-model verifier comparisons are descriptive "
            "only. Blinded human review is required."
        ),
    }
    PAIRS.write_text(
        json.dumps(pair_payload, indent=2, ensure_ascii=False),
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
        "paired_graph_question_count": len(pair_rows),
        "by_route": by_route,
        "final_disposition_policy": (
            "Release only when every atomic claim is verifier-supported; "
            "otherwise abstain."
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
            "Stage 7C is a technical smoke. The frozen verifier uses the same "
            "model family as the generator and is not human gold."
        ),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Results: {RESULTS_JSONL}")
    print(f"Summary: {SUMMARY}")
    print(f"Paired comparison: {PAIRS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
