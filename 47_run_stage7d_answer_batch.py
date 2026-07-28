from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage7d_common import (
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

OUTPUT_DIR = ROOT / "outputs" / "stage7d_answer_eval"
JOBS_JSON = OUTPUT_DIR / "stage7d_execution_jobs.json"
BATCH_PLAN = OUTPUT_DIR / "stage7d_batch_plan.json"
CORPUS = (
    ROOT / "data" / "processed" / "stage2"
    / "candidate_snippets.jsonl"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
COMPLETED_JOB_DIR = CHECKPOINT_DIR / "completed_jobs"
BATCH_OUTPUT_DIR = OUTPUT_DIR / "batches"


def evidence_rows(
    snippet_ids: list[str],
    corpus: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for snippet_id in snippet_ids:
        source = corpus.get(snippet_id)
        if source is None:
            raise RuntimeError(
                f"Unknown evidence snippet: {snippet_id}"
            )
        rows.append(
            {
                "evidence_id": f"E{len(rows) + 1}",
                "snippet_id": snippet_id,
                "document_ids": source.get("document_ids", []),
                "text": source["text"],
            }
        )
    if len(rows) != 5 or len(
        {row["snippet_id"] for row in rows}
    ) != 5:
        raise RuntimeError(
            "Every job must contain five unique evidence snippets."
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

    # A valid abstention may legitimately omit answer text. Record a
    # standardized answer rather than treating that as a failed API call.
    if abstain and not answer:
        answer = (
            "Insufficient supported evidence to provide a reliable answer."
        )

    if not answer:
        raise ValueError(
            "Generator returned an empty answer without abstaining."
        )
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


def checkpoint_valid_answer(
    value: dict[str, Any] | None,
    evidence_ids: list[str],
) -> bool:
    if not value:
        return False
    if value.get("evidence_snippet_ids") != evidence_ids:
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
    allowed_evidence = {"E1", "E2", "E3", "E4", "E5"}
    return (
        {
            str(item.get("claim_id", "")) for item in items
        }
        == expected
        and all(
            item.get("status") in ALLOWED_LABELS
            and str(item.get("brief_rationale", "")).strip()
            and isinstance(
                item.get("material_qualifiers_checked"), list
            )
            and item.get("material_qualifiers_checked")
            and isinstance(item.get("evidence_ids"), list)
            and set(item.get("evidence_ids", []))
            <= allowed_evidence
            and not item.get("graph_edge_ids")
            for item in items
        )
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
            "No graph is supplied. Verify only against displayed text evidence "
            "E1-E5. graph_edge_ids must be an empty list."
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


def completed_job_valid(
    value: dict[str, Any] | None,
    job: dict[str, Any],
) -> bool:
    if not value:
        return False
    return (
        value.get("job_id") == job["job_id"]
        and value.get("evidence_snippet_ids")
        == job["evidence_snippet_ids"]
        and bool(str(value.get("answer", "")).strip())
        and len(value.get("claims", []))
        == len(value.get("verifications", []))
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fields = [
        "job_id",
        "stage7_id",
        "question_type",
        "conditions",
        "logical_arm_count",
        "generator_abstained",
        "claim_count",
        "supported_claim_count",
        "contradicted_claim_count",
        "insufficient_claim_count",
        "final_disposition",
        "answer",
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
                    "job_id": row["job_id"],
                    "stage7_id": row["stage7_id"],
                    "question_type": row["question_type"],
                    "conditions": json.dumps(row["conditions"]),
                    "logical_arm_count": row["logical_arm_count"],
                    "generator_abstained": row[
                        "generator_abstained"
                    ],
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
                    "evidence_snippet_ids": json.dumps(
                        row["evidence_snippet_ids"]
                    ),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-index", type=int, required=True
    )
    args = parser.parse_args()

    missing = [
        str(path)
        for path in (JOBS_JSON, BATCH_PLAN, CORPUS)
        if not path.exists()
    ]
    if missing:
        print("ERROR: Missing Stage 7D-B inputs:")
        for path in missing:
            print("-", path)
        return 1

    jobs_payload = json.loads(
        JOBS_JSON.read_text(encoding="utf-8")
    )
    plan = json.loads(
        BATCH_PLAN.read_text(encoding="utf-8")
    )
    batch_count = plan["batch_count"]
    if args.batch_index < 1 or args.batch_index > batch_count:
        raise RuntimeError(
            f"Batch index must be between 1 and {batch_count}."
        )

    jobs = [
        job for job in jobs_payload["jobs"]
        if job["batch_index"] == args.batch_index
    ]
    corpus = {
        row["snippet_id"]: row for row in load_jsonl(CORPUS)
    }

    BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPLETED_JOB_DIR.mkdir(parents=True, exist_ok=True)

    client = None
    batch_results = []
    answer_calls_this_run = 0
    verifier_calls_this_run = 0
    completed_job_skips = 0

    for index, job in enumerate(jobs, start=1):
        print(
            f"{index}/{len(jobs)} {job['job_id']} "
            f"{job['stage7_id']} {job['conditions']}"
        )
        completed_path = (
            COMPLETED_JOB_DIR / f"{job['job_id']}.json"
        )
        completed = load_checkpoint(completed_path)
        if completed_job_valid(completed, job):
            print("  job: SKIP completed checkpoint")
            batch_results.append(completed)
            completed_job_skips += 1
            continue

        evidence = evidence_rows(
            job["evidence_snippet_ids"], corpus
        )
        if client is None:
            client = groq_client()

        answer_path = (
            CHECKPOINT_DIR / "answers" / f"{job['job_id']}.json"
        )
        answer_checkpoint = load_checkpoint(answer_path)
        if checkpoint_valid_answer(
            answer_checkpoint,
            job["evidence_snippet_ids"],
        ):
            answer_record = answer_checkpoint["answer_record"]
            answer_call = answer_checkpoint.get("call")
            print("  answer: SKIP checkpoint")
        else:
            base_messages = generator_messages(
                job["question_type"],
                job["question"],
                evidence,
            )
            answer_record = None
            answer_call = None
            last_answer_issue = None

            # Structured JSON can still be semantically incomplete. Retry a
            # bounded number of times without changing the frozen evidence,
            # model, temperature or scientific prompt.
            for semantic_attempt in range(1, 4):
                messages = list(base_messages)
                if last_answer_issue:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Correct the prior output. Return a non-empty "
                                "answer, or set abstain=true with an "
                                "abstention_reason. A non-abstaining answer "
                                "must include at least one atomic claim. "
                                f"Problem: {last_answer_issue}"
                            ),
                        }
                    )

                parsed, answer_call = call_json(
                    client,
                    messages,
                    (
                        f"stage7d_answer_{job['job_id']}"
                        f"_attempt_{semantic_attempt}"
                    ),
                    max_tokens=1800,
                )
                answer_calls_this_run += 1

                try:
                    answer_record = normalize_answer(parsed)
                    break
                except ValueError as exc:
                    last_answer_issue = str(exc)

            if answer_record is None:
                raise RuntimeError(
                    f"Generator remained semantically incomplete after "
                    f"three attempts for {job['job_id']}: "
                    f"{last_answer_issue}"
                )

            save_checkpoint(
                answer_path,
                {
                    "answer_record": answer_record,
                    "call": answer_call,
                    "evidence_snippet_ids": job[
                        "evidence_snippet_ids"
                    ],
                },
            )

        claims = answer_record["claims"]
        verify_path = (
            CHECKPOINT_DIR
            / "verifications"
            / f"{job['job_id']}.json"
        )
        verify_checkpoint = load_checkpoint(verify_path)
        if verification_complete(verify_checkpoint, claims):
            verifications = verify_checkpoint["verifications"]
            verification_call = verify_checkpoint.get("call")
            print("  verifier: SKIP checkpoint")
        else:
            verifications, verification_call = complete_verifications(
                client,
                job["question"],
                evidence,
                claims,
                f"stage7d_verify_{job['job_id']}",
            )
            save_checkpoint(
                verify_path,
                {
                    "verifications": verifications,
                    "call": verification_call,
                },
            )
            if claims:
                verifier_calls_this_run += 1

        disposition = final_disposition(
            answer_record["abstain"],
            claims,
            verifications,
            answer_record["answer"],
        )

        result = {
            "run_utc": datetime.now(timezone.utc).isoformat(),
            **job,
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
        save_checkpoint(completed_path, result)
        batch_results.append(result)

    batch_results.sort(key=lambda row: row["job_order"])
    jsonl_path = (
        BATCH_OUTPUT_DIR
        / f"stage7d_batch_{args.batch_index:02d}_results.jsonl"
    )
    csv_path = (
        BATCH_OUTPUT_DIR
        / f"stage7d_batch_{args.batch_index:02d}_results.csv"
    )
    summary_path = (
        BATCH_OUTPUT_DIR
        / f"stage7d_batch_{args.batch_index:02d}_summary.json"
    )

    write_jsonl(jsonl_path, batch_results)
    write_csv(csv_path, batch_results)

    labels = [
        item["status"]
        for row in batch_results
        for item in row["verifications"]
    ]
    summary = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "development_only": True,
        "sealed_test_accessed": False,
        "batch_index": args.batch_index,
        "batch_count": batch_count,
        "job_count": len(jobs),
        "completed_job_count": len(batch_results),
        "completed_job_checkpoint_skips": completed_job_skips,
        "answer_calls_this_run": answer_calls_this_run,
        "verifier_calls_this_run": verifier_calls_this_run,
        "claim_count": len(labels),
        "verifier_label_counts": dict(
            sorted(Counter(labels).items())
        ),
        "release_count": sum(
            row["final_disposition"] == "release"
            for row in batch_results
        ),
        "abstention_count": sum(
            row["final_disposition"] == "abstain"
            for row in batch_results
        ),
        "unsupported_claims_released": sum(
            row["final_disposition"] == "release"
            and any(
                item["status"] != "supported"
                for item in row["verifications"]
            )
            for row in batch_results
        ),
        "graph_payload_supplied_count": sum(
            row["graph_supplied_to_generator"]
            or row["graph_supplied_to_verifier"]
            for row in batch_results
        ),
        "scientific_note": (
            "Batch results are automated same-model verifier outputs and are "
            "not human gold."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Batch results: {jsonl_path}")
    print(f"Batch summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
