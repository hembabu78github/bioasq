from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage7_common import (
    ROOT,
    call_json,
    groq_client,
    load_checkpoint,
    load_jsonl,
    save_checkpoint,
    write_jsonl,
)

SAMPLE = (
    ROOT / "outputs" / "stage7_graph_routing_v2"
    / "stage7_graph_routing_v2_sample.json"
)
RANKINGS = (
    ROOT / "data" / "processed" / "stage4"
    / "dev_retrieval_rankings_private.jsonl"
)
CORPUS = (
    ROOT / "data" / "processed" / "stage2" / "candidate_snippets.jsonl"
)
OUTPUT_DIR = ROOT / "outputs" / "stage7_graph_routing_v2"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_JSONL = OUTPUT_DIR / "stage7_graph_routing_v2_results.jsonl"
RESULTS_CSV = OUTPUT_DIR / "stage7_graph_routing_v2_results.csv"
SUMMARY = OUTPUT_DIR / "stage7_graph_routing_v2_summary.json"

CANDIDATE_COUNT = 20
SELECT_COUNT = 5
VALID_RELEVANCE = {"high", "medium", "low"}


def candidate_rows(
    ranked_ids: list[str],
    corpus: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for rank, snippet_id in enumerate(ranked_ids, start=1):
        if snippet_id in seen or snippet_id not in corpus:
            continue
        seen.add(snippet_id)
        source = corpus[snippet_id]
        rows.append(
            {
                "snippet_id": snippet_id,
                "base_rank": rank,
                "text": source["text"],
                "document_ids": source.get("document_ids", []),
            }
        )
        if len(rows) >= CANDIDATE_COUNT:
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
                "Use only the supplied biomedical snippets. Build a question-focused "
                "evidence graph; do not rank snippets and do not add outside facts. "
                "Return exactly one JSON object with:\n"
                "- entities: list of {entity_id, name, type}\n"
                "- relations: list of {edge_id, source, relation, target, "
                "evidence_snippet_ids, question_relevance}\n"
                "- answer_aspects: list of {aspect_id, description, "
                "evidence_snippet_ids, question_relevance}\n"
                "- graph_sufficient: boolean\n"
                "- route_reason: concise string\n\n"
                "question_relevance must be high, medium, or low. Every evidence "
                "snippet ID must come from the supplied candidates. An answer_aspect "
                "is a distinct item, mechanism, qualifier, comparison, or conclusion "
                "needed to answer the question. Set graph_sufficient=false when the "
                "candidate evidence lacks a material qualifier or answer component. "
                "Keep the JSON compact: include at most 12 entities, 16 relations, "
                "and 10 answer_aspects. Omit low-value background entities and "
                "relations. Use short names and descriptions."
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


def normalize_id_list(value: Any, allowed: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        snippet_id = str(item)
        if snippet_id in allowed and snippet_id not in result:
            result.append(snippet_id)
    return result


def normalize_graph(
    parsed: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    allowed = {row["snippet_id"] for row in candidates}

    entities = []
    for index, item in enumerate(parsed.get("entities", []), start=1):
        if isinstance(item, dict):
            entities.append(
                {
                    "entity_id": str(
                        item.get("entity_id") or item.get("id") or f"N{index}"
                    ),
                    "name": str(
                        item.get("name") or item.get("label") or ""
                    ).strip(),
                    "type": str(item.get("type", "")).strip(),
                }
            )

    relations = []
    for index, item in enumerate(parsed.get("relations", []), start=1):
        if not isinstance(item, dict):
            continue
        relevance = str(
            item.get("question_relevance", "low")
        ).strip().lower()
        if relevance not in VALID_RELEVANCE:
            relevance = "low"
        evidence_ids = normalize_id_list(
            item.get("evidence_snippet_ids"), allowed
        )
        if not evidence_ids:
            continue
        relations.append(
            {
                "edge_id": str(item.get("edge_id") or f"R{index}"),
                "source": str(item.get("source", "")).strip(),
                "relation": str(item.get("relation", "")).strip(),
                "target": str(item.get("target", "")).strip(),
                "evidence_snippet_ids": evidence_ids,
                "question_relevance": relevance,
            }
        )

    answer_aspects = []
    for index, item in enumerate(parsed.get("answer_aspects", []), start=1):
        if not isinstance(item, dict):
            continue
        relevance = str(
            item.get("question_relevance", "medium")
        ).strip().lower()
        if relevance not in VALID_RELEVANCE:
            relevance = "medium"
        evidence_ids = normalize_id_list(
            item.get("evidence_snippet_ids"), allowed
        )
        if not evidence_ids:
            continue
        answer_aspects.append(
            {
                "aspect_id": str(item.get("aspect_id") or f"A{index}"),
                "description": str(item.get("description", "")).strip(),
                "evidence_snippet_ids": evidence_ids,
                "question_relevance": relevance,
            }
        )

    return {
        "entities": entities,
        "relations": relations,
        "answer_aspects": answer_aspects,
        "graph_sufficient": bool(parsed.get("graph_sufficient")),
        "route_reason": str(
            parsed.get("route_reason")
            or parsed.get("brief_route_reason")
            or ""
        ).strip(),
    }


def build_coverage(
    graph: dict[str, Any],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    snippet_coverage: dict[str, dict[str, float]] = defaultdict(dict)
    item_weights: dict[str, float] = {}

    relation_weight = {"high": 3.0, "medium": 2.0, "low": 0.5}
    aspect_weight = {"high": 5.0, "medium": 3.0, "low": 1.0}

    for relation in graph["relations"]:
        key = f"relation:{relation['edge_id']}"
        weight = relation_weight[relation["question_relevance"]]
        item_weights[key] = weight
        for snippet_id in relation["evidence_snippet_ids"]:
            snippet_coverage[snippet_id][key] = weight

    for aspect in graph["answer_aspects"]:
        key = f"aspect:{aspect['aspect_id']}"
        weight = aspect_weight[aspect["question_relevance"]]
        item_weights[key] = weight
        for snippet_id in aspect["evidence_snippet_ids"]:
            snippet_coverage[snippet_id][key] = weight

    return snippet_coverage, item_weights


def deterministic_select(
    candidates: list[dict[str, Any]],
    graph: dict[str, Any],
    hybrid_top5: list[str],
) -> dict[str, Any]:
    by_id = {row["snippet_id"]: row for row in candidates}
    snippet_coverage, item_weights = build_coverage(graph)

    selected = []
    covered = set()
    selection_trace = []

    while len(selected) < SELECT_COUNT:
        best = None
        for row in candidates:
            snippet_id = row["snippet_id"]
            if snippet_id in selected:
                continue

            keys = set(snippet_coverage.get(snippet_id, {}))
            marginal_keys = keys - covered
            marginal_weight = sum(item_weights[key] for key in marginal_keys)
            novelty_bonus = 0.40 if snippet_id not in hybrid_top5 else 0.0
            base_rank_bonus = 0.25 / max(row["base_rank"], 1)
            total = marginal_weight + novelty_bonus + base_rank_bonus

            candidate = (
                total,
                marginal_weight,
                novelty_bonus,
                -row["base_rank"],
                snippet_id,
                marginal_keys,
            )
            if best is None or candidate[:5] > best[:5]:
                best = candidate

        if best is None:
            break

        total, marginal_weight, novelty_bonus, neg_rank, snippet_id, marginal_keys = best
        selected.append(snippet_id)
        covered.update(marginal_keys)
        selection_trace.append(
            {
                "selection_order": len(selected),
                "snippet_id": snippet_id,
                "base_rank": by_id[snippet_id]["base_rank"],
                "marginal_coverage_keys": sorted(marginal_keys),
                "marginal_coverage_weight": marginal_weight,
                "novelty_bonus": novelty_bonus,
                "total_selection_score": total,
            }
        )

    for row in candidates:
        if len(selected) >= SELECT_COUNT:
            break
        if row["snippet_id"] not in selected:
            selected.append(row["snippet_id"])
            selection_trace.append(
                {
                    "selection_order": len(selected),
                    "snippet_id": row["snippet_id"],
                    "base_rank": row["base_rank"],
                    "marginal_coverage_keys": [],
                    "marginal_coverage_weight": 0.0,
                    "novelty_bonus": 0.0,
                    "total_selection_score": 0.25 / max(row["base_rank"], 1),
                }
            )

    novel = [snippet_id for snippet_id in selected if snippet_id not in hybrid_top5]
    graph_supported_selected = [
        snippet_id
        for snippet_id in selected
        if snippet_coverage.get(snippet_id)
    ]

    return {
        "selected_snippet_ids": selected,
        "selection_trace": selection_trace,
        "covered_graph_items": sorted(covered),
        "covered_graph_item_count": len(covered),
        "total_graph_item_count": len(item_weights),
        "novel_selected_snippet_ids": novel,
        "graph_supported_selected_snippet_ids": graph_supported_selected,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "stage7_id",
        "question_id",
        "question_type",
        "retrieval_uncertainty_label",
        "structural_role",
        "graph_sufficient",
        "relation_count",
        "answer_aspect_count",
        "hybrid_top5",
        "graph_selected_top5",
        "evidence_set_changed",
        "novel_selected_count",
        "covered_graph_item_count",
        "graph_route_eligible",
        "risk_route_selected",
        "route_reason",
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
    required = [SAMPLE, RANKINGS, CORPUS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("ERROR: Missing Stage 7B inputs:")
        for path in missing:
            print("-", path)
        return 1

    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))["questions"]
    rankings = {
        row["question_id"]: row for row in load_jsonl(RANKINGS)
    }
    corpus = {row["snippet_id"]: row for row in load_jsonl(CORPUS)}
    client = groq_client()

    results = []
    for index, sample_row in enumerate(sample, start=1):
        qid = sample_row["question_id"]
        ranking = rankings.get(qid)
        if not ranking:
            raise RuntimeError(f"Missing rankings for {qid}.")

        candidates = candidate_rows(
            ranking["hybrid_bge_ranked_ids"], corpus
        )
        hybrid_top5 = [
            row["snippet_id"] for row in candidates[:SELECT_COUNT]
        ]

        checkpoint_path = CHECKPOINT_DIR / f"{qid}.json"
        checkpoint = load_checkpoint(checkpoint_path)
        if (
            checkpoint
            and isinstance(checkpoint.get("graph"), dict)
            and checkpoint["graph"].get("relations") is not None
        ):
            graph = checkpoint["graph"]
            call = checkpoint.get("call")
            print(
                f"{index}/8 {sample_row['stage7_id']}: "
                "graph SKIP checkpoint"
            )
        else:
            print(
                f"{index}/8 {sample_row['stage7_id']}: "
                f"graph call ({len(candidates)} candidates)"
            )
            base_messages = graph_messages(
                sample_row["question"], candidates
            )
            graph = None
            call = None
            semantic_issues = []

            for semantic_attempt in range(1, 4):
                messages = list(base_messages)
                if semantic_attempt > 1:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The previous response was structurally empty or "
                                "not useful. Return a compact JSON graph containing "
                                "at least one evidence-backed relation or one "
                                "evidence-backed answer_aspect when the snippets "
                                "support one. Otherwise explicitly set "
                                "graph_sufficient=false and explain why."
                            ),
                        }
                    )

                parsed, call = call_json(
                    client,
                    messages,
                    (
                        f"stage7_graph_routing_v2_{qid}"
                        f"_semantic_attempt_{semantic_attempt}"
                    ),
                    max_tokens=2600,
                )
                graph = normalize_graph(parsed, candidates)
                semantic_issues = []
                if (
                    not graph["relations"]
                    and not graph["answer_aspects"]
                    and graph["graph_sufficient"]
                ):
                    semantic_issues.append(
                        "graph_sufficient=true but no relations or answer aspects"
                    )
                if graph["graph_sufficient"] and not graph["route_reason"]:
                    semantic_issues.append(
                        "graph_sufficient=true but route_reason is empty"
                    )

                if not semantic_issues:
                    break

                print(
                    f"{sample_row['stage7_id']}: semantic graph retry "
                    f"{semantic_attempt}: {'; '.join(semantic_issues)}"
                )

            if graph is None or semantic_issues:
                raise RuntimeError(
                    f"{sample_row['stage7_id']} graph remained semantically "
                    f"incomplete: {semantic_issues}"
                )

            save_checkpoint(
                checkpoint_path,
                {
                    "graph": graph,
                    "call": call,
                    "candidate_snippet_ids": [
                        row["snippet_id"] for row in candidates
                    ],
                },
            )

        selection = deterministic_select(
            candidates, graph, hybrid_top5
        )
        graph_selected = selection["selected_snippet_ids"]
        changed = set(graph_selected) != set(hybrid_top5)

        relevant_relation_count = sum(
            relation["question_relevance"] in {"high", "medium"}
            for relation in graph["relations"]
        )
        relevant_aspect_count = sum(
            aspect["question_relevance"] in {"high", "medium"}
            for aspect in graph["answer_aspects"]
        )

        structural_graph = sample_row["structural_role"] == "graph_stress"
        graph_route_eligible = bool(
            structural_graph
            and graph["graph_sufficient"]
            and relevant_relation_count >= 1
            and relevant_aspect_count >= 1
            and selection["novel_selected_snippet_ids"]
            and changed
        )

        if sample_row["retrieval_uncertainty_label"] == "low":
            risk_route = "bge"
        elif graph_route_eligible:
            risk_route = "graph_coverage"
        else:
            risk_route = "hybrid"

        results.append(
            {
                **sample_row,
                "candidate_pool_size": len(candidates),
                "hybrid_top5": hybrid_top5,
                "graph": graph,
                "graph_sufficient": graph["graph_sufficient"],
                "relation_count": len(graph["relations"]),
                "answer_aspect_count": len(graph["answer_aspects"]),
                "relevant_relation_count": relevant_relation_count,
                "relevant_answer_aspect_count": relevant_aspect_count,
                "graph_selected_top5": graph_selected,
                "selection_trace": selection["selection_trace"],
                "covered_graph_items": selection["covered_graph_items"],
                "covered_graph_item_count": selection[
                    "covered_graph_item_count"
                ],
                "total_graph_item_count": selection[
                    "total_graph_item_count"
                ],
                "novel_selected_snippet_ids": selection[
                    "novel_selected_snippet_ids"
                ],
                "novel_selected_count": len(
                    selection["novel_selected_snippet_ids"]
                ),
                "graph_supported_selected_snippet_ids": selection[
                    "graph_supported_selected_snippet_ids"
                ],
                "evidence_set_changed": changed,
                "graph_route_eligible": graph_route_eligible,
                "risk_route_selected": risk_route,
                "route_reason": graph["route_reason"],
                "graph_call": call,
            }
        )

    write_jsonl(RESULTS_JSONL, results)
    write_csv(RESULTS_CSV, results)

    graph_stress = [
        row for row in results if row["structural_role"] == "graph_stress"
    ]
    controls = [
        row for row in results if row["structural_role"] == "direct_control"
    ]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "sealed_test_accessed": False,
        "purpose": "Graph-routing technical stress test only.",
        "question_count": len(results),
        "graph_call_count": len(results),
        "answer_generation_call_count": 0,
        "verification_call_count": 0,
        "graph_stress_route_eligible_count": sum(
            row["graph_route_eligible"] for row in graph_stress
        ),
        "graph_stress_evidence_set_change_count": sum(
            row["evidence_set_changed"] for row in graph_stress
        ),
        "control_graph_route_eligible_count": sum(
            row["graph_route_eligible"] for row in controls
        ),
        "risk_route_counts": dict(
            sorted(
                {
                    route: sum(
                        row["risk_route_selected"] == route
                        for row in results
                    )
                    for route in {"bge", "hybrid", "graph_coverage"}
                }.items()
            )
        ),
        "scientific_note": (
            "Graph selection is deterministic from evidence-provenance edges "
            "and answer aspects; the model no longer supplies final snippet rankings."
        ),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Results: {RESULTS_JSONL}")
    print(f"Summary: {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
