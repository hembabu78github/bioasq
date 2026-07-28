from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage7d_common import (
    ROOT,
    call_json,
    groq_client,
    load_checkpoint,
    load_jsonl,
    save_checkpoint,
    write_jsonl,
)

SAMPLE = (
    ROOT / "outputs" / "stage7d_route_freeze"
    / "stage7d_route_sample.json"
)
RANKINGS = (
    ROOT / "data" / "processed" / "stage4"
    / "dev_retrieval_rankings_private.jsonl"
)
CORPUS = (
    ROOT / "data" / "processed" / "stage2"
    / "candidate_snippets.jsonl"
)
V3_RESULTS = (
    ROOT / "outputs" / "stage7_graph_scope_v3"
    / "stage7_graph_scope_v3_results.jsonl"
)

OUTPUT_DIR = ROOT / "outputs" / "stage7d_route_freeze"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_JSONL = OUTPUT_DIR / "stage7d_route_results.jsonl"
RESULTS_CSV = OUTPUT_DIR / "stage7d_route_results.csv"
SUMMARY = OUTPUT_DIR / "stage7d_route_summary.json"

CANDIDATE_COUNT = 20
SELECT_COUNT = 5
RELEVANT = {"high", "medium"}


def candidates_from_rankings(
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
    if len(rows) < CANDIDATE_COUNT:
        raise RuntimeError(
            f"Only {len(rows)} unique candidates resolved; expected 20."
        )
    return rows


def graph_messages(
    question: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    payload = [
        {"snippet_id": row["snippet_id"], "text": row["text"]}
        for row in candidates
    ]
    return [
        {
            "role": "system",
            "content": (
                "Use only the supplied biomedical snippets. Extract a compact "
                "question-focused evidence graph; do not rank snippets and do "
                "not add outside facts. Return one JSON object with:\n"
                "- entities: at most 12 {entity_id, name, type}\n"
                "- relations: at most 16 {edge_id, source, relation, target, "
                "evidence_snippet_ids, question_relevance}\n"
                "- answer_aspects: at most 12 {aspect_id, description, "
                "evidence_snippet_ids, question_relevance}\n"
                "- graph_sufficient: boolean\n"
                "- route_reason: concise string\n\n"
                "question_relevance is high, medium or low. For a list "
                "question, create one answer_aspect for each distinct candidate "
                "answer item. Every evidence ID must come from the supplied "
                "snippets. Omit irrelevant background."
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


def normalize_ids(value: Any, allowed: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        snippet_id = str(item)
        if snippet_id in allowed and snippet_id not in output:
            output.append(snippet_id)
    return output


def normalize_graph(
    parsed: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    allowed = {row["snippet_id"] for row in candidates}

    entities = []
    for index, item in enumerate(parsed.get("entities", []), start=1):
        if not isinstance(item, dict):
            continue
        entities.append(
            {
                "entity_id": str(
                    item.get("entity_id")
                    or item.get("id")
                    or f"N{index}"
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
        if relevance not in {"high", "medium", "low"}:
            relevance = "low"
        evidence = normalize_ids(
            item.get("evidence_snippet_ids"), allowed
        )
        if not evidence:
            continue
        relations.append(
            {
                "edge_id": str(item.get("edge_id") or f"R{index}"),
                "source": str(item.get("source", "")).strip(),
                "relation": str(item.get("relation", "")).strip(),
                "target": str(item.get("target", "")).strip(),
                "evidence_snippet_ids": evidence,
                "question_relevance": relevance,
            }
        )

    aspects = []
    for index, item in enumerate(
        parsed.get("answer_aspects", []), start=1
    ):
        if not isinstance(item, dict):
            continue
        relevance = str(
            item.get("question_relevance", "medium")
        ).strip().lower()
        if relevance not in {"high", "medium", "low"}:
            relevance = "medium"
        evidence = normalize_ids(
            item.get("evidence_snippet_ids"), allowed
        )
        if not evidence:
            continue
        aspects.append(
            {
                "aspect_id": str(
                    item.get("aspect_id") or f"A{index}"
                ),
                "description": str(
                    item.get("description", "")
                ).strip(),
                "evidence_snippet_ids": evidence,
                "question_relevance": relevance,
            }
        )

    return {
        "entities": entities[:12],
        "relations": relations[:16],
        "answer_aspects": aspects[:12],
        "model_graph_sufficient": bool(
            parsed.get("graph_sufficient")
        ),
        "route_reason": str(
            parsed.get("route_reason")
            or parsed.get("brief_route_reason")
            or ""
        ).strip(),
    }


def graph_items(
    graph: dict[str, Any],
) -> tuple[dict[str, set[str]], dict[str, float]]:
    item_to_snippets: dict[str, set[str]] = {}
    weights: dict[str, float] = {}

    for relation in graph["relations"]:
        if relation["question_relevance"] not in RELEVANT:
            continue
        key = f"relation:{relation['edge_id']}"
        item_to_snippets[key] = set(
            relation["evidence_snippet_ids"]
        )
        weights[key] = (
            3.0
            if relation["question_relevance"] == "high"
            else 2.0
        )

    for aspect in graph["answer_aspects"]:
        if aspect["question_relevance"] not in RELEVANT:
            continue
        key = f"aspect:{aspect['aspect_id']}"
        item_to_snippets[key] = set(
            aspect["evidence_snippet_ids"]
        )
        weights[key] = (
            5.0
            if aspect["question_relevance"] == "high"
            else 3.0
        )

    return item_to_snippets, weights


def deterministic_select(
    candidates: list[dict[str, Any]],
    graph: dict[str, Any],
    hybrid_top5: list[str],
) -> dict[str, Any]:
    item_to_snippets, weights = graph_items(graph)
    snippet_to_items: dict[str, set[str]] = defaultdict(set)

    for key, snippet_ids in item_to_snippets.items():
        for snippet_id in snippet_ids:
            snippet_to_items[snippet_id].add(key)

    selected: list[str] = []
    covered: set[str] = set()
    trace: list[dict[str, Any]] = []

    while len(selected) < SELECT_COUNT:
        scored = []
        for row in candidates:
            snippet_id = row["snippet_id"]
            if snippet_id in selected:
                continue

            marginal = snippet_to_items[snippet_id] - covered
            marginal_weight = sum(
                weights[key] for key in marginal
            )
            if marginal_weight <= 0:
                continue

            novelty_bonus = (
                0.25 if snippet_id not in hybrid_top5 else 0.0
            )
            total = (
                marginal_weight
                + novelty_bonus
                + 0.20 / max(row["base_rank"], 1)
            )
            scored.append(
                (
                    total,
                    marginal_weight,
                    novelty_bonus,
                    -row["base_rank"],
                    snippet_id,
                    marginal,
                )
            )

        if not scored:
            break

        best = max(scored, key=lambda item: item[:5])
        (
            total,
            marginal_weight,
            novelty_bonus,
            negative_rank,
            snippet_id,
            marginal,
        ) = best
        selected.append(snippet_id)
        covered.update(marginal)
        trace.append(
            {
                "selection_order": len(selected),
                "snippet_id": snippet_id,
                "base_rank": -negative_rank,
                "selection_source": "positive_graph_coverage",
                "marginal_coverage_keys": sorted(marginal),
                "marginal_coverage_weight": marginal_weight,
                "novelty_bonus": novelty_bonus,
                "total_selection_score": total,
            }
        )

    by_id = {row["snippet_id"]: row for row in candidates}
    for snippet_id in hybrid_top5:
        if len(selected) >= SELECT_COUNT:
            break
        if snippet_id not in selected:
            selected.append(snippet_id)
            trace.append(
                {
                    "selection_order": len(selected),
                    "snippet_id": snippet_id,
                    "base_rank": by_id[snippet_id]["base_rank"],
                    "selection_source": "hybrid_top5_fill",
                    "marginal_coverage_keys": [],
                    "marginal_coverage_weight": 0.0,
                    "novelty_bonus": 0.0,
                    "total_selection_score": 0.0,
                }
            )

    if len(selected) != SELECT_COUNT:
        raise RuntimeError(
            f"Selector returned {len(selected)} snippets instead of five."
        )

    hybrid_set = set(hybrid_top5)
    selected_set = set(selected)
    hybrid_items = {
        key
        for key, snippets in item_to_snippets.items()
        if snippets & hybrid_set
    }
    selected_items = {
        key
        for key, snippets in item_to_snippets.items()
        if snippets & selected_set
    }
    exclusive_items = selected_items - hybrid_items

    useful_novel = [
        row
        for row in trace
        if row["snippet_id"] not in hybrid_set
        and row["marginal_coverage_weight"] > 0
    ]
    zero_coverage_novel = [
        row
        for row in trace
        if row["snippet_id"] not in hybrid_set
        and row["marginal_coverage_weight"] == 0
    ]

    return {
        "selected_snippet_ids": selected,
        "selection_trace": trace,
        "relevant_graph_item_count": len(item_to_snippets),
        "hybrid_covered_relevant_items": sorted(hybrid_items),
        "selected_covered_relevant_items": sorted(selected_items),
        "graph_exclusive_relevant_items": sorted(
            exclusive_items
        ),
        "graph_exclusive_relevant_item_count": len(
            exclusive_items
        ),
        "useful_novel_selected_snippet_ids": [
            row["snippet_id"] for row in useful_novel
        ],
        "useful_novel_selected_count": len(useful_novel),
        "zero_coverage_novel_selected_count": len(
            zero_coverage_novel
        ),
        "evidence_set_changed": selected_set != hybrid_set,
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fields = [
        "stage7_id",
        "question_type",
        "retrieval_uncertainty_label",
        "graph_role",
        "route_selected",
        "graph_route_eligible",
        "eligibility_source",
        "model_graph_sufficient",
        "relevant_relation_count",
        "relevant_answer_aspect_count",
        "graph_exclusive_relevant_item_count",
        "useful_novel_selected_count",
        "zero_coverage_novel_selected_count",
        "evidence_set_changed",
        "hybrid_top5",
        "selected_evidence_snippet_ids",
        "graph_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(
                            row[key], ensure_ascii=False
                        )
                        if isinstance(row.get(key), list)
                        else row.get(key)
                    )
                    for key in fields
                }
            )


def main() -> int:
    required = [SAMPLE, RANKINGS, CORPUS, V3_RESULTS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("ERROR: Missing Stage 7D routing inputs:")
        for path in missing:
            print("-", path)
        return 1

    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))[
        "questions"
    ]
    rankings = {
        row["question_id"]: row for row in load_jsonl(RANKINGS)
    }
    corpus = {
        row["snippet_id"]: row for row in load_jsonl(CORPUS)
    }
    v3_by_stage7 = {
        row["stage7_id"]: row for row in load_jsonl(V3_RESULTS)
    }

    client = None
    results = []
    reused_graph_count = 0
    new_graph_call_count = 0

    for index, sample_row in enumerate(sample, start=1):
        qid = sample_row["question_id"]
        ranking = rankings.get(qid)
        if not ranking:
            raise RuntimeError(f"Missing rankings for {qid}.")

        candidates = candidates_from_rankings(
            ranking["hybrid_bge_ranked_ids"], corpus
        )
        candidate_ids = [
            row["snippet_id"] for row in candidates
        ]
        hybrid_top5 = candidate_ids[:SELECT_COUNT]

        if sample_row["question_type"] != "list":
            results.append(
                {
                    **sample_row,
                    "candidate_pool_size": len(candidates),
                    "hybrid_top5": hybrid_top5,
                    "graph": None,
                    "model_graph_sufficient": False,
                    "relevant_relation_count": 0,
                    "relevant_answer_aspect_count": 0,
                    "selection_trace": [],
                    "relevant_graph_item_count": 0,
                    "graph_exclusive_relevant_items": [],
                    "graph_exclusive_relevant_item_count": 0,
                    "useful_novel_selected_snippet_ids": [],
                    "useful_novel_selected_count": 0,
                    "zero_coverage_novel_selected_count": 0,
                    "evidence_set_changed": False,
                    "deterministic_scope_sufficient": False,
                    "graph_route_eligible": False,
                    "eligibility_source": "not_list_question",
                    "route_selected": "hybrid_text_only",
                    "selected_evidence_snippet_ids": hybrid_top5,
                    "graph_source": "not_applicable",
                    "graph_hidden_from_downstream": True,
                }
            )
            print(
                f"{index}/24 {sample_row['stage7_id']}: "
                "HYBRID (non-list)"
            )
            continue

        checkpoint_path = CHECKPOINT_DIR / f"{qid}.json"
        checkpoint = load_checkpoint(checkpoint_path)
        v3 = v3_by_stage7.get(sample_row["stage7_id"])

        if checkpoint and isinstance(
            checkpoint.get("graph"), dict
        ):
            graph = checkpoint["graph"]
            graph_call = checkpoint.get("call")
            graph_source = checkpoint.get(
                "source", "stage7d_checkpoint"
            )
            print(
                f"{index}/24 {sample_row['stage7_id']}: "
                "graph SKIP Stage 7D checkpoint"
            )
        elif v3 and isinstance(v3.get("graph"), dict):
            graph = v3["graph"]
            graph_call = v3.get("graph_call")
            graph_source = "reused_stage7b_v3"
            reused_graph_count += 1
            save_checkpoint(
                checkpoint_path,
                {
                    "graph": graph,
                    "call": graph_call,
                    "source": graph_source,
                },
            )
            print(
                f"{index}/24 {sample_row['stage7_id']}: "
                "graph reused from V3"
            )
        else:
            if client is None:
                client = groq_client()
            print(
                f"{index}/24 {sample_row['stage7_id']}: "
                "new graph call"
            )
            parsed, graph_call = call_json(
                client,
                graph_messages(
                    sample_row["question"], candidates
                ),
                f"stage7d_route_graph_{qid}",
                max_tokens=2800,
            )
            graph = normalize_graph(parsed, candidates)
            graph_source = "stage7d_new_call"
            new_graph_call_count += 1
            save_checkpoint(
                checkpoint_path,
                {
                    "graph": graph,
                    "call": graph_call,
                    "source": graph_source,
                },
            )

        selection = deterministic_select(
            candidates, graph, hybrid_top5
        )
        relevant_relations = sum(
            item["question_relevance"] in RELEVANT
            for item in graph["relations"]
        )
        relevant_aspects = sum(
            item["question_relevance"] in RELEVANT
            for item in graph["answer_aspects"]
        )
        deterministic_sufficient = bool(
            relevant_aspects >= 2
            and selection["relevant_graph_item_count"] >= 2
        )
        eligible = bool(
            deterministic_sufficient
            and selection[
                "graph_exclusive_relevant_item_count"
            ] >= 1
            and selection["useful_novel_selected_count"] >= 1
            and selection["evidence_set_changed"]
            and selection[
                "zero_coverage_novel_selected_count"
            ] == 0
        )

        if eligible and graph["model_graph_sufficient"]:
            eligibility_source = "model_and_deterministic"
        elif eligible:
            eligibility_source = (
                "deterministic_override_of_model_flag"
            )
        else:
            eligibility_source = "not_eligible"

        route = (
            "graph_selected_text_only"
            if eligible
            else "hybrid_text_only"
        )
        selected_ids = (
            selection["selected_snippet_ids"]
            if eligible
            else hybrid_top5
        )

        results.append(
            {
                **sample_row,
                "candidate_pool_size": len(candidates),
                "hybrid_top5": hybrid_top5,
                "graph": graph,
                "model_graph_sufficient": graph[
                    "model_graph_sufficient"
                ],
                "relevant_relation_count": relevant_relations,
                "relevant_answer_aspect_count": relevant_aspects,
                "selection_trace": selection[
                    "selection_trace"
                ],
                "relevant_graph_item_count": selection[
                    "relevant_graph_item_count"
                ],
                "graph_exclusive_relevant_items": selection[
                    "graph_exclusive_relevant_items"
                ],
                "graph_exclusive_relevant_item_count": (
                    selection[
                        "graph_exclusive_relevant_item_count"
                    ]
                ),
                "useful_novel_selected_snippet_ids": (
                    selection[
                        "useful_novel_selected_snippet_ids"
                    ]
                ),
                "useful_novel_selected_count": selection[
                    "useful_novel_selected_count"
                ],
                "zero_coverage_novel_selected_count": (
                    selection[
                        "zero_coverage_novel_selected_count"
                    ]
                ),
                "evidence_set_changed": selection[
                    "evidence_set_changed"
                ],
                "deterministic_scope_sufficient": (
                    deterministic_sufficient
                ),
                "graph_route_eligible": eligible,
                "eligibility_source": eligibility_source,
                "route_selected": route,
                "selected_evidence_snippet_ids": selected_ids,
                "graph_source": graph_source,
                "graph_call": graph_call,
                "graph_hidden_from_downstream": True,
            }
        )
        print(
            f"  route: {'GRAPH' if eligible else 'HYBRID'}"
        )

    results.sort(key=lambda row: row["stage7_id"])
    write_jsonl(RESULTS_JSONL, results)
    write_csv(RESULTS_CSV, results)

    route_counts = Counter(
        row["route_selected"] for row in results
    )
    summary = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "development_only": True,
        "sealed_test_accessed": False,
        "purpose": (
            "Freeze Stage 7D risk-adaptive routes before answer generation."
        ),
        "question_count": len(results),
        "list_question_count": sum(
            row["question_type"] == "list"
            for row in results
        ),
        "graph_route_eligible_count": sum(
            row["graph_route_eligible"] for row in results
        ),
        "graph_route_stage7_ids": [
            row["stage7_id"]
            for row in results
            if row["graph_route_eligible"]
        ],
        "route_counts": dict(sorted(route_counts.items())),
        "reused_stage7b_v3_graph_count": reused_graph_count,
        "new_graph_call_count": new_graph_call_count,
        "answer_generation_call_count": 0,
        "verification_call_count": 0,
        "graph_payload_supplied_to_downstream_count": 0,
        "scientific_note": (
            "The graph is frozen as an evidence selector only. Downstream "
            "generation and verification will receive text evidence only."
        ),
    }
    SUMMARY.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Results: {RESULTS_JSONL}")
    print(f"Summary: {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
