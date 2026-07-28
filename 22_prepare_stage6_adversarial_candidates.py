from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage6_common import (
    BASELINE_MODEL,
    ROOT,
    call_json,
    final_human_label,
    groq_client,
    read_csv,
    write_jsonl,
)

BLINDED_AUDIT = (
    ROOT / "outputs" / "stage4_correction" / "claim_manual_audit_blinded.csv"
)
HUMAN_REFERENCE = (
    ROOT / "outputs" / "stage5_annotation_review"
    / "claim_audit_merged_for_adjudication.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "stage6_prep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRIVATE_CANDIDATES = OUTPUT_DIR / "adversarial_claim_candidates_private.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "adversarial_generation_summary.json"
RAW_CALLS_PATH = OUTPUT_DIR / "adversarial_generation_calls_private.jsonl"

EXPECTED_SUPPORTED_SOURCE_COUNT = 24
VARIANTS_PER_SOURCE = 2


def messages_for_source(source: dict[str, str]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Create exactly two controlled biomedical claim perturbations from one "
                "human-verified supported claim. Use only the supplied question, evidence "
                "and evidence-derived graph.\n\n"
                "Return JSON with exactly these keys:\n"
                '{"contradicted_candidate": {...}, '
                '"insufficient_candidate": {...}}\n\n'
                "Each candidate object must contain:\n"
                "- text\n"
                "- perturbation_type\n"
                "- changed_span\n"
                "- source_evidence_ids\n"
                "- generation_rationale\n\n"
                "CONTRADICTED candidate rules:\n"
                "- It must be directly opposed by the displayed evidence, not merely absent.\n"
                "- Prefer one controlled change: entity substitution, relation reversal, "
                "negation flip, number change, or date change.\n"
                "- Do not create unsafe medical advice.\n\n"
                "INSUFFICIENT candidate rules:\n"
                "- It must be plausible but not fully supported or contradicted by the "
                "displayed evidence.\n"
                "- Prefer one controlled material qualifier: market-status claim, "
                "regulatory leap, causality leap, effectiveness leap, population "
                "expansion, comparative superlative, date qualifier, or numerical detail.\n"
                "- Do not simply make the sentence vague.\n\n"
                "Both candidate texts must differ materially from the source claim. "
                "Do not include labels other than the requested object names."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question type: {source['question_type']}\n"
                f"Question: {source['question']}\n"
                f"Human-supported source claim: {source['claim']}\n\n"
                f"Retrieved evidence:\n{source['retrieved_evidence']}\n\n"
                f"Evidence-derived graph:\n{source['local_graph']}"
            ),
        },
    ]


def clean_candidate(
    value: Any,
    source: dict[str, str],
    expected_label: str,
    audit_id: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{audit_id}: candidate is not an object.")
    text = str(value.get("text", "")).strip()
    if not text:
        raise ValueError(f"{audit_id}: candidate text is empty.")
    if text.casefold() == source["claim"].strip().casefold():
        raise ValueError(f"{audit_id}: candidate is identical to source claim.")

    evidence_ids = value.get("source_evidence_ids", [])
    if not isinstance(evidence_ids, list):
        evidence_ids = []

    return {
        "audit_id": audit_id,
        "source_audit_id": source["audit_id"],
        "source_question_id": hashlib.sha256(
            " ".join(source["question"].split()).casefold().encode("utf-8")
        ).hexdigest()[:16],
        "question_type": source["question_type"],
        "question": source["question"],
        "original_supported_claim": source["claim"],
        "adversarial_claim": text,
        "provisional_target_label": expected_label,
        "perturbation_type": str(value.get("perturbation_type", "")).strip(),
        "changed_span": str(value.get("changed_span", "")).strip(),
        "source_evidence_ids": [str(x) for x in evidence_ids],
        "generation_rationale": str(
            value.get("generation_rationale", "")
        ).strip(),
        "retrieved_evidence": source["retrieved_evidence"],
        "local_graph": source["local_graph"],
    }


def main() -> int:
    for path in (BLINDED_AUDIT, HUMAN_REFERENCE):
        if not path.exists():
            print(f"ERROR: Missing required file: {path}")
            return 1

    audit_rows = read_csv(BLINDED_AUDIT)
    reference_rows = read_csv(HUMAN_REFERENCE)
    reference = {
        row["audit_id"]: final_human_label(row)
        for row in reference_rows
    }

    source_rows = [
        row for row in audit_rows
        if reference.get(row["audit_id"]) == "supported"
    ]
    source_rows.sort(key=lambda row: row["audit_id"])

    if len(source_rows) != EXPECTED_SUPPORTED_SOURCE_COUNT:
        print(
            "ERROR: Expected 24 human-supported source claims, "
            f"found {len(source_rows)}."
        )
        return 2

    existing: dict[str, dict[str, Any]] = {}
    if PRIVATE_CANDIDATES.exists():
        for row in PRIVATE_CANDIDATES.read_text(encoding="utf-8").splitlines():
            if row.strip():
                parsed = json.loads(row)
                existing[parsed["source_audit_id"]] = parsed

    client = groq_client()
    candidates: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []

    for index, source in enumerate(source_rows, start=1):
        source_id = source["audit_id"]
        cached = existing.get(source_id)
        if cached and isinstance(cached.get("variants"), list):
            print(f"[{index}/{len(source_rows)}] SKIP completed {source_id}")
            candidates.extend(cached["variants"])
            continue

        print(f"[{index}/{len(source_rows)}] Generating variants for {source_id}")
        parsed, call = call_json(
            client,
            BASELINE_MODEL,
            messages_for_source(source),
            f"adversarial_generation_{source_id}",
            max_tokens=1800,
        )

        contradicted = clean_candidate(
            parsed.get("contradicted_candidate"),
            source,
            "contradicted",
            f"ADV-{(index-1)*2+1:03d}",
        )
        insufficient = clean_candidate(
            parsed.get("insufficient_candidate"),
            source,
            "insufficient_evidence",
            f"ADV-{(index-1)*2+2:03d}",
        )
        variants = [contradicted, insufficient]
        candidates.extend(variants)
        call_rows.append(
            {
                "source_audit_id": source_id,
                "variants": variants,
                "call": call,
            }
        )

        # Crash-safe checkpoint.
        grouped = []
        for completed_source in source_rows[:index]:
            sid = completed_source["audit_id"]
            own = [x for x in candidates if x["source_audit_id"] == sid]
            if own:
                grouped.append({"source_audit_id": sid, "variants": own})
        with PRIVATE_CANDIDATES.open("w", encoding="utf-8") as handle:
            for row in grouped:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        with RAW_CALLS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"source_audit_id": source_id, "call": call},
                    ensure_ascii=False,
                )
                + "\n"
            )

    flat_candidates = []
    for line in PRIVATE_CANDIDATES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        flat_candidates.extend(json.loads(line)["variants"])

    flat_candidates.sort(key=lambda row: row["audit_id"])
    write_jsonl(
        OUTPUT_DIR / "adversarial_claim_candidates_flat_private.jsonl",
        flat_candidates,
    )

    label_counts = Counter(row["provisional_target_label"] for row in flat_candidates)
    perturbation_counts = Counter(
        row["perturbation_type"] or "unspecified" for row in flat_candidates
    )
    digest = hashlib.sha256(
        (
            OUTPUT_DIR / "adversarial_claim_candidates_flat_private.jsonl"
        ).read_bytes()
    ).hexdigest()

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_supported_claim_count": len(source_rows),
        "variants_per_source": VARIANTS_PER_SOURCE,
        "candidate_count": len(flat_candidates),
        "provisional_target_label_counts": dict(sorted(label_counts.items())),
        "perturbation_type_counts": dict(sorted(perturbation_counts.items())),
        "candidate_sha256": digest,
        "model": BASELINE_MODEL,
        "human_validation_required": True,
        "scientific_note": (
            "Generated target labels are provisional. Human evidence-only labels "
            "determine the reference standard."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Adversarial candidates: {len(flat_candidates)}")
    print(f"Summary saved: {SUMMARY_PATH}")
    return 0 if len(flat_candidates) == 48 else 3


if __name__ == "__main__":
    raise SystemExit(main())
