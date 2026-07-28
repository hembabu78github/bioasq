from __future__ import annotations

import argparse
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

RANKINGS = (
    ROOT / "data" / "processed" / "stage4"
    / "dev_retrieval_rankings_private.jsonl"
)
CORPUS = ROOT / "data" / "processed" / "stage2" / "candidate_snippets.jsonl"
SAMPLE_DIR = ROOT / "outputs" / "stage7_sampling"

CONDITIONS = [
    "bge_text_only",
    "hybrid_text_only",
    "graph_reranked",
    "risk_adaptive_agentic",
]
EVIDENCE_COUNT = 5
GRAPH_CANDIDATE_COUNT = 10


def evidence_rows(
    snippet_ids: list[str],
    corpus: dict[str, dict[str, Any]],
    count: int = EVIDENCE_COUNT,
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
        if len(rows) >= count:
            break
    return rows


def graph_messages(
    question: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    payload = [
        {
            "snippet_id": row["snippet_id"],
            "text": row["text"],
        }
        for row in candidates
    ]
    return [
        {
            "role": "system",
            "content": (
                "Use only the candidate biomedical snippets. Construct a small "
                "question-focused evidence graph and rerank the snippets. Return JSON "
                "with entities, relations, ranked_snippet_ids, graph_sufficient, and "
                "brief_route_reason. Each relation must contain edge_id, source, "
                "relation, target, and evidence_snippet_ids. ranked_snippet_ids must "
                "contain each useful candidate at most once. Do not add external facts."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\nCandidate snippets:\n"
                f"{json.dumps(payload, ensure_ascii=False)}"
            ),
        },
    ]


def normalized_graph(
    parsed: dict[str, Any],
    candidate_ids: list[str],
) -> dict[str, Any]:
    allowed = set(candidate_ids)
    ranked = []
    for value in parsed.get("ranked_snippet_ids", []):
        snippet_id = str(value)
        if snippet_id in allowed and snippet_id not in ranked:
            ranked.append(snippet_id)
    for snippet_id in candidate_ids:
        if snippet_id not in ranked:
            ranked.append(snippet_id)

    entities = parsed.get("entities", [])
    relations = parsed.get("relations", [])
    return {
        "entities": entities if isinstance(entities, list) else [],
        "relations": relations if isinstance(relations, list) else [],
        "ranked_snippet_ids": ranked,
        "graph_sufficient": bool(parsed.get("graph_sufficient")),
        "brief_route_reason": str(
            parsed.get("brief_route_reason", "")
        ).strip(),
    }


def graph_text(graph: dict[str, Any]) -> str:
    compact = {
        "entities": graph.get("entities", []),
        "relations": graph.get("relations", []),
        "graph_sufficient": graph.get("graph_sufficient", False),
    }
    return json.dumps(compact, ensure_ascii=False)


def generator_messages(
    question_type: str,
    question: str,
    evidence: list[dict[str, Any]],
    graph: str,
    condition: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer a BioASQ biomedical question using only the supplied evidence "
                "and, when present, the evidence-derived graph. Do not use external "
                "knowledge. Return JSON with answer, abstain, abstention_reason, and "
                "claims. claims must be a list of atomic factual claims, each with "
                "claim_id and text. For yes/no questions, begin the answer with Yes or "
                "No. If a material answer cannot be supported, set abstain=true and "
                "explain briefly. Keep the answer concise and evidence-bounded."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Condition: {condition}\n"
                f"Question type: {question_type}\n"
                f"Question: {question}\n\n"
                f"Evidence:\n{format_evidence(evidence)}\n\n"
                f"Graph:\n{graph}"
            ),
        },
    ]


def normalize_answer(parsed: dict[str, Any]) -> dict[str, Any]:
    answer = str(parsed.get("answer", "")).strip()
    abstain = bool(parsed.get("abstain"))
    reason = str(parsed.get("abstention_reason", "")).strip()
    raw_claims = parsed.get("claims", [])
    claims = []
    if isinstance(raw_claims, list):
        for index, item in enumerate(raw_claims, start=1):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            claim_id = str(item.get("claim_id", "")).strip() or f"C{index}"
            claims.append({"claim_id": claim_id, "text": text})

    if not answer:
        raise ValueError("Generator returned an empty answer.")
    if not abstain and not claims:
        raise ValueError("Non-abstaining answer returned no atomic claims.")
    return {
        "answer": answer,
        "abstain": abstain,
        "abstention_reason": reason,
        "claims": claims,
    }


def complete_verifications(
    client,
    question: str,
    evidence: list[dict[str, Any]],
    graph: str,
    claims: list[dict[str, str]],
    call_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not claims:
        return [], None

    expected = {row["claim_id"] for row in claims}
    messages = frozen_hardened_verifier_messages(
        question, format_evidence(evidence), graph, claims
    )
    last_issues = []

    for semantic_attempt in range(1, 4):
        parsed, call = call_json(
            client,
            messages,
            f"{call_name}_attempt_{semantic_attempt}",
            max_tokens=min(3200, 800 + 300 * len(claims)),
        )
        by_id = {}
        duplicates = set()
        for item in verification_items(parsed):
            if not isinstance(item, dict):
                continue
            claim_id = str(item.get("claim_id", ""))
            if claim_id in by_id:
                duplicates.add(claim_id)
            by_id[claim_id] = item

        issues = []
        if duplicates:
            issues.append("duplicate IDs: " + ", ".join(sorted(duplicates)))
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
            rationale = str(item.get("brief_rationale", "")).strip()
            qualifiers = item.get("material_qualifiers_checked", [])
            if not status:
                issues.append(f"{claim['claim_id']}: invalid status")
            if not rationale:
                issues.append(f"{claim['claim_id']}: empty rationale")
            if not isinstance(qualifiers, list) or not qualifiers:
                issues.append(f"{claim['claim_id']}: missing qualifier checks")
            normalized.append(
                {
                    "claim_id": claim["claim_id"],
                    "claim_text": claim["text"],
                    "status": status,
                    "evidence_ids": item.get("evidence_ids", []),
                    "graph_edge_ids": item.get("graph_edge_ids", []),
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
                    "The prior JSON was incomplete. Return exactly one complete "
                    "verification for every required claim ID. Problems: "
                    + "; ".join(issues)
                ),
            }
        ]

    raise RuntimeError(
        f"{call_name} remained incomplete: {'; '.join(last_issues)}"
    )


def checkpoint_valid_answer(value: dict[str, Any] | None) -> bool:
    if not value:
        return False
    answer = value.get("answer_record", {})
    return bool(str(answer.get("answer", "")).strip()) and (
        bool(answer.get("abstain")) or bool(answer.get("claims"))
    )


def checkpoint_valid_verification(
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
    expected = {row["claim_id"] for row in claims}
    if {str(row.get("claim_id", "")) for row in items} != expected:
        return False
    return all(
        row.get("status") in ALLOWED_LABELS
        and str(row.get("brief_rationale", "")).strip()
        and isinstance(row.get("material_qualifiers_checked"), list)
        and row.get("material_qualifiers_checked")
        for row in items
    )


def choose_condition_inputs(
    condition: str,
    sample_row: dict[str, Any],
    bge_evidence: list[dict[str, Any]],
    hybrid_evidence: list[dict[str, Any]],
    graph_evidence: list[dict[str, Any]],
    graph: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, str]:
    no_graph = "No graph supplied for this text-only condition."
    if condition == "bge_text_only":
        return bge_evidence, no_graph, "bge"
    if condition == "hybrid_text_only":
        return hybrid_evidence, no_graph, "hybrid"
    if condition == "graph_reranked":
        return graph_evidence, graph_text(graph), "graph_reranked"

    uncertainty = sample_row["retrieval_uncertainty_label"]
    graph_role = sample_row["graph_role"]
    if uncertainty == "low":
        return bge_evidence, no_graph, "bge"
    if uncertainty == "medium" and graph_role != "graph_suitable_candidate":
        return hybrid_evidence, no_graph, "hybrid"
    return graph_evidence, graph_text(graph), "graph_reranked"


def write_flat_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "stage7_id", "question_id", "question_type",
        "retrieval_uncertainty_label", "graph_role", "condition",
        "route_selected", "evidence_changed_vs_hybrid",
        "graph_sufficient", "abstain", "claim_count",
        "supported_claim_count", "contradicted_claim_count",
        "insufficient_claim_count", "answer",
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
                    "stage7_id": row["stage7_id"],
                    "question_id": row["question_id"],
                    "question_type": row["question_type"],
                    "retrieval_uncertainty_label": row[
                        "retrieval_uncertainty_label"
                    ],
                    "graph_role": row["graph_role"],
                    "condition": row["condition"],
                    "route_selected": row["route_selected"],
                    "evidence_changed_vs_hybrid": row[
                        "evidence_changed_vs_hybrid"
                    ],
                    "graph_sufficient": row["graph_sufficient"],
                    "abstain": row["abstain"],
                    "claim_count": len(row["claims"]),
                    "supported_claim_count": counts["supported"],
                    "contradicted_claim_count": counts["contradicted"],
                    "insufficient_claim_count": counts[
                        "insufficient_evidence"
                    ],
                    "answer": row["answer"],
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()

    sample_path = SAMPLE_DIR / (
        "stage7_smoke_sample.json"
        if args.mode == "smoke"
        else "stage7_question_sample.json"
    )
    if not sample_path.exists() or not RANKINGS.exists() or not CORPUS.exists():
        print("ERROR: Stage 7 input files are missing.")
        for path in (sample_path, RANKINGS, CORPUS):
            print(f"- {path}: {'OK' if path.exists() else 'MISSING'}")
        return 1

    sample = json.loads(sample_path.read_text(encoding="utf-8"))["questions"]
    rankings = {
        row["question_id"]: row for row in load_jsonl(RANKINGS)
    }
    corpus = {row["snippet_id"]: row for row in load_jsonl(CORPUS)}

    out_dir = ROOT / "outputs" / f"stage7_{args.mode}"
    checkpoint_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / f"stage7_{args.mode}_results.jsonl"
    flat_path = out_dir / f"stage7_{args.mode}_results.csv"
    summary_path = out_dir / f"stage7_{args.mode}_summary.json"

    client = groq_client()
    results = []

    for q_index, row in enumerate(sample, start=1):
        qid = row["question_id"]
        rank = rankings.get(qid)
        if not rank:
            raise RuntimeError(f"Missing rankings for {qid}.")

        print(
            f"Question {q_index}/{len(sample)}: "
            f"{row['stage7_id']} {row['question_type']}"
        )

        bge_ids = rank["bge_ranked_ids"]
        hybrid_ids = rank["hybrid_bge_ranked_ids"]
        bge_evidence = evidence_rows(bge_ids, corpus)
        hybrid_evidence = evidence_rows(hybrid_ids, corpus)

        graph_checkpoint_path = checkpoint_dir / "graph" / f"{qid}.json"
        graph_checkpoint = load_checkpoint(graph_checkpoint_path)
        if graph_checkpoint and graph_checkpoint.get("graph", {}).get(
            "ranked_snippet_ids"
        ):
            graph = graph_checkpoint["graph"]
            print("  graph: SKIP checkpoint")
        else:
            candidates = evidence_rows(
                hybrid_ids, corpus, count=GRAPH_CANDIDATE_COUNT
            )
            parsed, graph_call = call_json(
                client,
                graph_messages(row["question"], candidates),
                f"stage7_graph_{qid}",
                max_tokens=1800,
            )
            graph = normalized_graph(
                parsed, [item["snippet_id"] for item in candidates]
            )
            save_checkpoint(
                graph_checkpoint_path,
                {"graph": graph, "call": graph_call},
            )

        graph_evidence = evidence_rows(
            graph["ranked_snippet_ids"], corpus
        )
        hybrid_snippet_ids = [
            item["snippet_id"] for item in hybrid_evidence
        ]
        graph_snippet_ids = [
            item["snippet_id"] for item in graph_evidence
        ]
        graph_changed = graph_snippet_ids != hybrid_snippet_ids

        for condition in CONDITIONS:
            evidence, graph_for_condition, route = choose_condition_inputs(
                condition,
                row,
                bge_evidence,
                hybrid_evidence,
                graph_evidence,
                graph,
            )
            answer_path = (
                checkpoint_dir / "answers" / condition / f"{qid}.json"
            )
            answer_checkpoint = load_checkpoint(answer_path)

            if checkpoint_valid_answer(answer_checkpoint):
                answer_record = answer_checkpoint["answer_record"]
                answer_call = answer_checkpoint.get("call")
                print(f"  {condition}: answer SKIP")
            else:
                parsed, answer_call = call_json(
                    client,
                    generator_messages(
                        row["question_type"],
                        row["question"],
                        evidence,
                        graph_for_condition,
                        condition,
                    ),
                    f"stage7_answer_{condition}_{qid}",
                    max_tokens=1500,
                )
                answer_record = normalize_answer(parsed)
                save_checkpoint(
                    answer_path,
                    {
                        "answer_record": answer_record,
                        "call": answer_call,
                        "evidence_snippet_ids": [
                            item["snippet_id"] for item in evidence
                        ],
                        "route_selected": route,
                    },
                )

            verify_path = (
                checkpoint_dir
                / "verifications"
                / condition
                / f"{qid}.json"
            )
            verify_checkpoint = load_checkpoint(verify_path)
            claims = answer_record["claims"]

            if checkpoint_valid_verification(verify_checkpoint, claims):
                verifications = verify_checkpoint["verifications"]
                verify_call = verify_checkpoint.get("call")
                print(f"  {condition}: verifier SKIP")
            else:
                verifications, verify_call = complete_verifications(
                    client,
                    row["question"],
                    evidence,
                    graph_for_condition,
                    claims,
                    f"stage7_verify_{condition}_{qid}",
                )
                save_checkpoint(
                    verify_path,
                    {
                        "verifications": verifications,
                        "call": verify_call,
                    },
                )

            results.append(
                {
                    "run_utc": datetime.now(timezone.utc).isoformat(),
                    **row,
                    "condition": condition,
                    "route_selected": route,
                    "model": MODEL,
                    "evidence": evidence,
                    "evidence_snippet_ids": [
                        item["snippet_id"] for item in evidence
                    ],
                    "hybrid_reference_snippet_ids": hybrid_snippet_ids,
                    "graph_reranked_snippet_ids": graph_snippet_ids,
                    "evidence_changed_vs_hybrid": (
                        graph_changed
                        if route == "graph_reranked"
                        else [
                            item["snippet_id"] for item in evidence
                        ] != hybrid_snippet_ids
                    ),
                    "graph_sufficient": graph.get(
                        "graph_sufficient", False
                    ),
                    "graph_route_reason": graph.get(
                        "brief_route_reason", ""
                    ),
                    "graph": (
                        graph
                        if route == "graph_reranked"
                        else None
                    ),
                    "answer": answer_record["answer"],
                    "abstain": answer_record["abstain"],
                    "abstention_reason": answer_record[
                        "abstention_reason"
                    ],
                    "claims": claims,
                    "verifications": verifications,
                    "answer_call": answer_call,
                    "verification_call": verify_call,
                }
            )

    results.sort(
        key=lambda r: (
            r["stage7_id"],
            CONDITIONS.index(r["condition"]),
        )
    )
    write_jsonl(results_path, results)
    write_flat_csv(flat_path, results)

    by_condition = {}
    for condition in CONDITIONS:
        rows = [row for row in results if row["condition"] == condition]
        labels = [
            item["status"]
            for row in rows
            for item in row["verifications"]
        ]
        by_condition[condition] = {
            "answer_count": len(rows),
            "abstention_count": sum(row["abstain"] for row in rows),
            "claim_count": len(labels),
            "verifier_label_counts": dict(sorted(Counter(labels).items())),
            "mean_claims_per_answer": (
                statistics.mean(len(row["claims"]) for row in rows)
                if rows
                else 0
            ),
        }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "development_only": True,
        "sealed_test_accessed": False,
        "question_count": len(sample),
        "condition_count": len(CONDITIONS),
        "answer_count": len(results),
        "conditions": CONDITIONS,
        "by_condition": by_condition,
        "graph_suitable_graph_condition_evidence_change_count": sum(
            row["condition"] == "graph_reranked"
            and row["graph_role"] == "graph_suitable_candidate"
            and row["evidence_changed_vs_hybrid"]
            for row in results
        ),
        "scientific_note": (
            "Smoke outputs are technical feasibility results only. They are not "
            "final GraphRAG performance estimates."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
