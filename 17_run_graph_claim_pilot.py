from __future__ import annotations

import csv
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from json_repair import repair_json

ROOT = Path(__file__).resolve().parent
SELECTED_PATH = ROOT / "data" / "processed" / "stage4" / "graph_claim_pilot_ids.json"
RANKINGS_PATH = (
    ROOT / "data" / "processed" / "stage4" / "dev_retrieval_rankings_private.jsonl"
)
CORPUS_PATH = ROOT / "data" / "processed" / "stage2" / "candidate_snippets.jsonl"
OUTPUT_DIR = ROOT / "outputs" / "stage4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "graph_claim_pilot_results.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "graph_claim_pilot_summary.json"
BY_TYPE_PATH = OUTPUT_DIR / "graph_claim_pilot_by_type.csv"
EXAMPLES_PATH = OUTPUT_DIR / "graph_claim_examples.jsonl"

MODEL = "openai/gpt-oss-20b"
SEED = 20260725
TEMPERATURE = 0
MAX_TOKENS = 1800
EVIDENCE_COUNT = 6
CALL_SPACING_SECONDS = 2.0
ALLOWED_ENTITY_TYPES = [
    "Drug",
    "Disease",
    "Gene",
    "Protein",
    "Symptom",
    "Treatment",
    "Outcome",
    "BiologicalProcess",
    "Population",
    "Other",
]
ALLOWED_RELATIONS = [
    "treats",
    "associated_with",
    "causes",
    "inhibits",
    "activates",
    "interacts_with",
    "part_of",
    "affects",
    "increases",
    "decreases",
    "used_for",
    "contraindicated_with",
    "has_side_effect",
    "marker_of",
    "mechanism_of",
    "other",
]
ALLOWED_STATUS = {"supported", "contradicted", "insufficient_evidence"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_json_object(raw: str) -> tuple[dict[str, Any], bool]:
    """Parse strict JSON, then repair minor model-generated JSON syntax errors."""
    try:
        parsed = json.loads(raw)
        repaired = False
    except Exception:
        parsed = repair_json(raw, return_objects=True)
        repaired = True

    if not isinstance(parsed, dict):
        raise ValueError("Top-level response is not a JSON object.")
    return parsed, repaired


def extract_failed_generation(exc: Exception) -> str | None:
    """Recover Groq's rejected JSON payload when json_validate_failed is returned."""
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None

    error = body.get("error", body)
    if not isinstance(error, dict):
        return None

    failed = error.get("failed_generation")
    return failed if isinstance(failed, str) and failed.strip() else None


def call_json(
    client: Groq,
    messages: list[dict[str, str]],
    call_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Try Groq JSON mode first. If the provider repeatedly returns
    json_validate_failed with an empty failed_generation payload, fall back to
    ordinary text generation with an explicit JSON-only instruction and repair
    minor syntax errors locally.
    """
    last_error = None

    structured_delays = (0, 5, 15)
    for attempt, delay in enumerate(structured_delays, start=1):
        if delay:
            time.sleep(delay)

        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                seed=SEED,
                max_completion_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            raw = response.choices[0].message.content or ""
            parsed, repaired = parse_json_object(raw)

            usage = (
                response.usage.model_dump()
                if getattr(response, "usage", None) is not None
                and hasattr(response.usage, "model_dump")
                else None
            )

            time.sleep(CALL_SPACING_SECONDS)
            return parsed, {
                "call_name": call_name,
                "attempt": attempt,
                "latency_ms": latency_ms,
                "usage": usage,
                "raw_response": raw,
                "response_mode": "groq_json_object",
                "json_repaired": repaired,
                "recovered_from_failed_generation": False,
            }

        except Exception as exc:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)

            failed_generation = extract_failed_generation(exc)
            if failed_generation:
                try:
                    parsed, repaired = parse_json_object(failed_generation)
                    print(
                        f"Recovered malformed JSON for {call_name} "
                        f"on attempt {attempt} using json-repair."
                    )
                    time.sleep(CALL_SPACING_SECONDS)
                    return parsed, {
                        "call_name": call_name,
                        "attempt": attempt,
                        "latency_ms": latency_ms,
                        "usage": None,
                        "raw_response": failed_generation,
                        "response_mode": "groq_failed_generation_repair",
                        "json_repaired": repaired,
                        "recovered_from_failed_generation": True,
                        "original_error": f"{type(exc).__name__}: {exc}",
                    }
                except Exception as repair_exc:
                    last_error = (
                        f"{type(exc).__name__}: {exc}; "
                        f"failed_generation repair also failed: "
                        f"{type(repair_exc).__name__}: {repair_exc}"
                    )
                    continue

            last_error = f"{type(exc).__name__}: {exc}"
            print(
                f"{call_name} JSON mode attempt {attempt} failed. "
                "Will retry or use plain-text JSON fallback."
            )

    # Provider-side JSON validation occasionally fails with an empty
    # failed_generation. In that case, request JSON as ordinary text and repair
    # minor syntax issues locally. The semantic prompt is unchanged.
    fallback_messages = list(messages)
    fallback_messages[0] = {
        "role": "system",
        "content": (
            messages[0]["content"]
            + "\nReturn exactly one JSON object. Do not use markdown fences, "
              "comments, trailing prose, or explanatory text outside the JSON."
        ),
    }

    for fallback_attempt, delay in enumerate((0, 10, 30), start=1):
        if delay:
            time.sleep(delay)

        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=fallback_messages,
                temperature=TEMPERATURE,
                seed=SEED,
                max_completion_tokens=MAX_TOKENS,
            )
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            raw = response.choices[0].message.content or ""
            if not raw.strip():
                raise ValueError("Plain-text fallback returned an empty response.")

            parsed, repaired = parse_json_object(raw)
            usage = (
                response.usage.model_dump()
                if getattr(response, "usage", None) is not None
                and hasattr(response.usage, "model_dump")
                else None
            )

            print(
                f"{call_name} completed using plain-text JSON fallback "
                f"(attempt {fallback_attempt})."
            )
            time.sleep(CALL_SPACING_SECONDS)
            return parsed, {
                "call_name": call_name,
                "attempt": len(structured_delays) + fallback_attempt,
                "latency_ms": latency_ms,
                "usage": usage,
                "raw_response": raw,
                "response_mode": "plain_text_json_fallback",
                "json_repaired": repaired,
                "recovered_from_failed_generation": False,
                "structured_mode_last_error": last_error,
            }

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(
                f"{call_name} plain-text fallback attempt "
                f"{fallback_attempt} failed: {last_error}"
            )

    raise RuntimeError(
        f"{call_name} failed after structured JSON and plain-text fallback retries: "
        f"{last_error}"
    )


def format_evidence(evidence: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"[{item['evidence_id']}] {item['text']}" for item in evidence
    )


def graph_messages(question: str, evidence: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Extract a small biomedical knowledge graph using only the supplied evidence. "
                "Do not add external facts. Return JSON with entities, relations, and "
                "graph_sufficient. Each entity must have entity_id, name, and entity_type. "
                "Entity type must be one of: "
                + ", ".join(ALLOWED_ENTITY_TYPES)
                + ". Each relation must have edge_id, source_entity_id, target_entity_id, "
                "relation_type, evidence_ids, evidence_quote, and confidence. relation_type "
                "must be one of: "
                + ", ".join(ALLOWED_RELATIONS)
                + ". evidence_quote must be copied from the supplied evidence and must support "
                "the edge. If no explicit relation is supported, return an empty relations list."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nEvidence:\n{format_evidence(evidence)}",
        },
    ]


def answer_messages(
    question: str,
    evidence: list[dict[str, str]],
    graph: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Produce two biomedical QA variants in one JSON object: text_only and "
                "graph_assisted. Use only the supplied evidence; graph_assisted may also use "
                "the supplied graph, whose edges are evidence-derived. Each variant must have "
                "answerable (boolean), answer (string), and claims (list). Every claim must "
                "have claim_id, text, importance ('critical' or 'supporting'), evidence_ids, "
                "and graph_edge_ids. text_only graph_edge_ids must be empty. Claims must be "
                "atomic and independently verifiable. Do not add medical advice or external facts."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\nEvidence:\n{format_evidence(evidence)}\n\n"
                f"Evidence-derived graph:\n{json.dumps(graph, ensure_ascii=False)}"
            ),
        },
    ]


def verification_messages(
    question: str,
    evidence: list[dict[str, str]],
    graph: dict[str, Any],
    answers: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Verify every claim from both answer variants using only the supplied evidence "
                "and evidence-derived graph. Return JSON with text_only and graph_assisted. "
                "Each must contain verifications, a list of objects with claim_id, status, "
                "evidence_ids, graph_edge_ids, and brief_rationale. status must be exactly one "
                "of supported, contradicted, insufficient_evidence. Do not treat the answer's "
                "own cited IDs as proof; independently check them."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\nEvidence:\n{format_evidence(evidence)}\n\n"
                f"Graph:\n{json.dumps(graph, ensure_ascii=False)}\n\n"
                f"Answer variants:\n{json.dumps(answers, ensure_ascii=False)}"
            ),
        },
    ]


def sanitize_graph(graph: dict[str, Any], valid_evidence_ids: set[str]) -> dict[str, Any]:
    entities = graph.get("entities", [])
    relations = graph.get("relations", [])
    if not isinstance(entities, list):
        entities = []
    if not isinstance(relations, list):
        relations = []

    clean_entities = []
    valid_entity_ids = set()
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        entity_id = str(entity.get("entity_id", "")).strip()
        name = str(entity.get("name", "")).strip()
        entity_type = str(entity.get("entity_type", "Other")).strip()
        if not entity_id or not name:
            continue
        if entity_type not in ALLOWED_ENTITY_TYPES:
            entity_type = "Other"
        clean_entities.append(
            {"entity_id": entity_id, "name": name, "entity_type": entity_type}
        )
        valid_entity_ids.add(entity_id)

    clean_relations = []
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        source = str(relation.get("source_entity_id", "")).strip()
        target = str(relation.get("target_entity_id", "")).strip()
        relation_type = str(relation.get("relation_type", "other")).strip()
        if source not in valid_entity_ids or target not in valid_entity_ids:
            continue
        if relation_type not in ALLOWED_RELATIONS:
            relation_type = "other"
        evidence_ids = [
            str(x)
            for x in relation.get("evidence_ids", [])
            if str(x) in valid_evidence_ids
        ]
        if not evidence_ids:
            continue
        clean_relations.append(
            {
                "edge_id": str(
                    relation.get("edge_id", f"R{len(clean_relations)+1}")
                ),
                "source_entity_id": source,
                "target_entity_id": target,
                "relation_type": relation_type,
                "evidence_ids": sorted(set(evidence_ids)),
                "evidence_quote": str(relation.get("evidence_quote", "")).strip(),
                "confidence": relation.get("confidence"),
            }
        )

    return {
        "entities": clean_entities,
        "relations": clean_relations,
        "graph_sufficient": bool(graph.get("graph_sufficient", bool(clean_relations))),
    }


def sanitize_answers(
    answers: dict[str, Any],
    valid_evidence_ids: set[str],
    valid_edge_ids: set[str],
) -> dict[str, Any]:
    clean = {}
    for variant in ("text_only", "graph_assisted"):
        value = answers.get(variant, {})
        claims = value.get("claims", []) if isinstance(value, dict) else []
        clean_claims = []
        for index, claim in enumerate(claims if isinstance(claims, list) else [], start=1):
            if not isinstance(claim, dict):
                continue
            text = str(claim.get("text", "")).strip()
            if not text:
                continue
            clean_claims.append(
                {
                    "claim_id": str(claim.get("claim_id", f"C{index}")),
                    "text": text,
                    "importance": (
                        "critical"
                        if str(claim.get("importance", "")).lower() == "critical"
                        else "supporting"
                    ),
                    "evidence_ids": sorted(
                        {
                            str(x)
                            for x in claim.get("evidence_ids", [])
                            if str(x) in valid_evidence_ids
                        }
                    ),
                    "graph_edge_ids": (
                        []
                        if variant == "text_only"
                        else sorted(
                            {
                                str(x)
                                for x in claim.get("graph_edge_ids", [])
                                if str(x) in valid_edge_ids
                            }
                        )
                    ),
                }
            )
        clean[variant] = {
            "answerable": bool(value.get("answerable", bool(clean_claims)))
            if isinstance(value, dict)
            else False,
            "answer": str(value.get("answer", "")).strip()
            if isinstance(value, dict)
            else "",
            "claims": clean_claims,
        }
    return clean


def sanitize_verifications(
    verification: dict[str, Any],
    answers: dict[str, Any],
    valid_evidence_ids: set[str],
    valid_edge_ids: set[str],
) -> dict[str, Any]:
    clean = {}
    for variant in ("text_only", "graph_assisted"):
        claim_ids = {claim["claim_id"] for claim in answers[variant]["claims"]}
        raw = verification.get(variant, {})
        items = raw.get("verifications", []) if isinstance(raw, dict) else []
        by_claim = {}
        for item in items if isinstance(items, list) else []:
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
        # Missing verification becomes insufficient evidence.
        clean_items = [
            by_claim.get(
                claim_id,
                {
                    "claim_id": claim_id,
                    "status": "insufficient_evidence",
                    "evidence_ids": [],
                    "graph_edge_ids": [],
                    "brief_rationale": "No valid verification record was returned.",
                },
            )
            for claim_id in sorted(claim_ids)
        ]
        clean[variant] = {"verifications": clean_items}
    return clean


def adjudicate(
    answer: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    claim_by_id = {claim["claim_id"]: claim for claim in answer["claims"]}
    items = verification["verifications"]
    supported = [item for item in items if item["status"] == "supported"]
    contradicted = [item for item in items if item["status"] == "contradicted"]
    critical_ids = {
        claim["claim_id"]
        for claim in answer["claims"]
        if claim["importance"] == "critical"
    }
    critical_supported = [
        item for item in supported if item["claim_id"] in critical_ids
    ]
    critical_contradicted = [
        item for item in contradicted if item["claim_id"] in critical_ids
    ]

    supported_ratio = len(supported) / max(len(items), 1)
    abstain = (
        not answer["answerable"]
        or bool(critical_contradicted)
        or (bool(critical_ids) and not critical_supported)
        or supported_ratio < 0.5
    )

    supported_claim_texts = [
        claim_by_id[item["claim_id"]]["text"]
        for item in supported
        if item["claim_id"] in claim_by_id
    ]

    return {
        "abstain": abstain,
        "decision_reason": (
            "critical_contradiction"
            if critical_contradicted
            else "no_supported_critical_claim"
            if critical_ids and not critical_supported
            else "insufficient_supported_claim_ratio"
            if supported_ratio < 0.5
            else "answer_supported"
        ),
        "supported_claim_ratio": round(supported_ratio, 6),
        "supported_claim_count": len(supported),
        "contradicted_claim_count": len(contradicted),
        "insufficient_claim_count": sum(
            item["status"] == "insufficient_evidence" for item in items
        ),
        "final_supported_claims": supported_claim_texts,
    }


def main() -> int:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "PASTE_YOUR_KEY_HERE":
        print("ERROR: GROQ_API_KEY is missing from .env.")
        return 2

    required = [SELECTED_PATH, RANKINGS_PATH, CORPUS_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("ERROR: Missing Stage 4 inputs:")
        for path in missing:
            print(f"- {path}")
        return 1

    selected = json.loads(SELECTED_PATH.read_text(encoding="utf-8")).get("questions", [])
    rankings = {row["question_id"]: row for row in load_jsonl(RANKINGS_PATH)}
    corpus = {row["snippet_id"]: row for row in load_jsonl(CORPUS_PATH)}

    client = Groq(api_key=api_key)

    # Resume safely from a partially completed JSONL file.
    records: list[dict[str, Any]] = []
    if RESULTS_PATH.exists():
        try:
            records = load_jsonl(RESULTS_PATH)
        except Exception as exc:
            print(f"ERROR: Existing results file cannot be read safely: {exc}")
            print(f"Move or rename this file before retrying: {RESULTS_PATH}")
            return 2

    completed_ids = {str(record.get("question_id", "")) for record in records}
    if completed_ids:
        print(
            f"Resuming Stage 4: {len(completed_ids)} completed question(s) "
            f"found in {RESULTS_PATH.name}."
        )

    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        for index, q in enumerate(selected, start=1):
            qid = q["question_id"]
            if qid in completed_ids:
                print(
                    f"[{index}/{len(selected)}] SKIP completed — "
                    f"{q['question_type']} — {q['question']}"
                )
                continue
            route = q["provisional_route"]
            ranking_key = (
                "hybrid_bge_ranked_ids" if route == "hybrid_bge" else "bge_ranked_ids"
            )
            ranked_ids = rankings[qid][ranking_key]
            evidence = []
            for rank, snippet_id in enumerate(ranked_ids[:EVIDENCE_COUNT], start=1):
                if snippet_id not in corpus:
                    continue
                evidence.append(
                    {
                        "evidence_id": f"E{rank}",
                        "snippet_id": snippet_id,
                        "document_ids": corpus[snippet_id].get("document_ids", []),
                        "text": corpus[snippet_id]["text"],
                    }
                )

            valid_evidence_ids = {item["evidence_id"] for item in evidence}
            print(f"[{index}/{len(selected)}] {q['question_type']} — {q['question']}")

            graph_raw, graph_call = call_json(
                client,
                graph_messages(q["question"], evidence),
                "graph_extraction",
            )
            graph = sanitize_graph(graph_raw, valid_evidence_ids)
            valid_edge_ids = {edge["edge_id"] for edge in graph["relations"]}

            answers_raw, answer_call = call_json(
                client,
                answer_messages(q["question"], evidence, graph),
                "dual_answer_generation",
            )
            answers = sanitize_answers(
                answers_raw,
                valid_evidence_ids,
                valid_edge_ids,
            )

            verification_raw, verification_call = call_json(
                client,
                verification_messages(q["question"], evidence, graph, answers),
                "claim_verification",
            )
            verifications = sanitize_verifications(
                verification_raw,
                answers,
                valid_evidence_ids,
                valid_edge_ids,
            )

            adjudication = {
                variant: adjudicate(answers[variant], verifications[variant])
                for variant in ("text_only", "graph_assisted")
            }

            record = {
                "run_utc": datetime.now(timezone.utc).isoformat(),
                "question_id": qid,
                "question_type": q["question_type"],
                "question": q["question"],
                "selection_reason": q["selection_reason"],
                "retrieval_uncertainty_score": q["retrieval_uncertainty_score"],
                "retrieval_uncertainty_label": q["retrieval_uncertainty_label"],
                "retrieval_route": route,
                "model": MODEL,
                "seed": SEED,
                "temperature": TEMPERATURE,
                "evidence": evidence,
                "graph": graph,
                "answers": answers,
                "verifications": verifications,
                "adjudication": adjudication,
                "calls": [graph_call, answer_call, verification_call],
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    def variant_summary(variant: str) -> dict[str, Any]:
        statuses = Counter()
        abstentions = 0
        ratios = []
        claim_count = 0
        for record in records:
            statuses.update(
                item["status"]
                for item in record["verifications"][variant]["verifications"]
            )
            abstentions += int(record["adjudication"][variant]["abstain"])
            ratios.append(record["adjudication"][variant]["supported_claim_ratio"])
            claim_count += len(record["answers"][variant]["claims"])
        return {
            "question_count": len(records),
            "claim_count": claim_count,
            "verification_status_counts": dict(sorted(statuses.items())),
            "abstention_count": abstentions,
            "abstention_rate": round(abstentions / max(len(records), 1), 6),
            "mean_supported_claim_ratio": round(statistics.mean(ratios), 6)
            if ratios
            else None,
        }

    total_calls = [call for record in records for call in record["calls"]]
    token_usage = Counter()
    for call in total_calls:
        usage = call.get("usage") or {}
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                token_usage[key] += value

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "question_count": len(records),
        "completed_call_count": len(total_calls),
        "expected_call_count": len(records) * 3,
        "mean_call_latency_ms": round(
            statistics.mean(call["latency_ms"] for call in total_calls), 2
        )
        if total_calls
        else None,
        "token_usage_totals": dict(token_usage),
        "graph": {
            "mean_entity_count": round(
                statistics.mean(len(record["graph"]["entities"]) for record in records),
                3,
            ),
            "mean_relation_count": round(
                statistics.mean(len(record["graph"]["relations"]) for record in records),
                3,
            ),
            "questions_with_no_supported_graph_edge": sum(
                len(record["graph"]["relations"]) == 0 for record in records
            ),
        },
        "text_only": variant_summary("text_only"),
        "graph_assisted": variant_summary("graph_assisted"),
        "interpretation": (
            "Feasibility and structural pilot only. Verification labels are model-generated "
            "and are not treated as gold accuracy labels. Independent manual audit is required."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with BY_TYPE_PATH.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "question_type",
            "question_count",
            "text_only_abstention_rate",
            "text_only_mean_supported_claim_ratio",
            "graph_assisted_abstention_rate",
            "graph_assisted_mean_supported_claim_ratio",
            "mean_graph_entities",
            "mean_graph_relations",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for qtype in sorted({record["question_type"] for record in records}):
            subset = [record for record in records if record["question_type"] == qtype]
            writer.writerow(
                {
                    "question_type": qtype,
                    "question_count": len(subset),
                    "text_only_abstention_rate": sum(
                        r["adjudication"]["text_only"]["abstain"] for r in subset
                    )
                    / len(subset),
                    "text_only_mean_supported_claim_ratio": statistics.mean(
                        r["adjudication"]["text_only"]["supported_claim_ratio"]
                        for r in subset
                    ),
                    "graph_assisted_abstention_rate": sum(
                        r["adjudication"]["graph_assisted"]["abstain"] for r in subset
                    )
                    / len(subset),
                    "graph_assisted_mean_supported_claim_ratio": statistics.mean(
                        r["adjudication"]["graph_assisted"]["supported_claim_ratio"]
                        for r in subset
                    ),
                    "mean_graph_entities": statistics.mean(
                        len(r["graph"]["entities"]) for r in subset
                    ),
                    "mean_graph_relations": statistics.mean(
                        len(r["graph"]["relations"]) for r in subset
                    ),
                }
            )

    example_order = sorted(
        records,
        key=lambda r: (
            -abs(
                r["adjudication"]["graph_assisted"]["supported_claim_ratio"]
                - r["adjudication"]["text_only"]["supported_claim_ratio"]
            ),
            r["question_id"],
        ),
    )[:6]
    with EXAMPLES_PATH.open("w", encoding="utf-8") as handle:
        for record in example_order:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Pilot summary saved: {SUMMARY_PATH}")
    return 0 if len(records) == 12 and len(total_calls) == 36 else 2


if __name__ == "__main__":
    raise SystemExit(main())
