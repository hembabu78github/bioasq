from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "bioasq11" / "training11b.json"
DERIVED_DIR = ROOT / "data" / "processed" / "stage2"
OUTPUT_DIR = ROOT / "outputs" / "stage2"
DERIVED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = DERIVED_DIR / "candidate_snippets.jsonl"
GOLD_PATH = DERIVED_DIR / "gold_relevance.jsonl"
PROVENANCE_PATH = DERIVED_DIR / "snippet_provenance_private.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "closed_corpus_manifest.json"


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if not DATA_PATH.exists():
        print(f"ERROR: Missing source dataset: {DATA_PATH}")
        return 1

    root = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    questions = root.get("questions", [])
    if not isinstance(questions, list):
        print("ERROR: Top-level questions field is not a list.")
        return 1

    snippets: dict[str, dict[str, Any]] = {}
    provenance: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"question_ids": set(), "document_ids": set()}
    )
    question_relevance: dict[str, set[str]] = defaultdict(set)
    empty_or_invalid = 0

    for q in questions:
        if not isinstance(q, dict):
            continue
        qid = str(q.get("id", ""))
        for snippet in q.get("snippets", []):
            if not isinstance(snippet, dict):
                empty_or_invalid += 1
                continue
            text = snippet.get("text")
            if not isinstance(text, str) or not text.strip():
                empty_or_invalid += 1
                continue

            normalized = normalize_text(text)
            snippet_id = sha256_text(normalized)
            document_id = str(snippet.get("document", ""))

            if snippet_id not in snippets:
                snippets[snippet_id] = {
                    "snippet_id": snippet_id,
                    "text": text.strip(),
                    "document_ids": set(),
                }

            if document_id:
                snippets[snippet_id]["document_ids"].add(document_id)
                provenance[snippet_id]["document_ids"].add(document_id)
            if qid:
                provenance[snippet_id]["question_ids"].add(qid)
                question_relevance[qid].add(snippet_id)

    with CORPUS_PATH.open("w", encoding="utf-8") as handle:
        for snippet_id in sorted(snippets):
            record = snippets[snippet_id]
            public_record = {
                "snippet_id": snippet_id,
                "text": record["text"],
                "document_ids": sorted(record["document_ids"]),
            }
            handle.write(json.dumps(public_record, ensure_ascii=False) + "\n")

    with GOLD_PATH.open("w", encoding="utf-8") as handle:
        for qid in sorted(question_relevance):
            handle.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "relevant_snippet_ids": sorted(question_relevance[qid]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with PROVENANCE_PATH.open("w", encoding="utf-8") as handle:
        for snippet_id in sorted(provenance):
            handle.write(
                json.dumps(
                    {
                        "snippet_id": snippet_id,
                        "question_ids": sorted(provenance[snippet_id]["question_ids"]),
                        "document_ids": sorted(provenance[snippet_id]["document_ids"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": "global deduplicated BioASQ11 gold-snippet bank",
        "scientific_scope": (
            "Closed-corpus snippet retrieval benchmark. This is not full-PubMed retrieval."
        ),
        "leakage_control": {
            "candidate_corpus_contains_question_ids": False,
            "question_to_gold_mapping_stored_separately": True,
            "retrieval_code_must_not_read_gold_relevance_before_ranking": True,
        },
        "source_question_count": len(questions),
        "candidate_snippet_count": len(snippets),
        "question_relevance_record_count": len(question_relevance),
        "empty_or_invalid_snippet_count": empty_or_invalid,
        "files": {
            "candidate_corpus": {
                "relative_path": str(CORPUS_PATH.relative_to(ROOT)),
                "sha256": sha256_file(CORPUS_PATH),
                "size_bytes": CORPUS_PATH.stat().st_size,
            },
            "gold_relevance": {
                "relative_path": str(GOLD_PATH.relative_to(ROOT)),
                "sha256": sha256_file(GOLD_PATH),
                "size_bytes": GOLD_PATH.stat().st_size,
            },
            "private_provenance": {
                "relative_path": str(PROVENANCE_PATH.relative_to(ROOT)),
                "sha256": sha256_file(PROVENANCE_PATH),
                "size_bytes": PROVENANCE_PATH.stat().st_size,
                "note": "Not consumed by the retriever.",
            },
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Candidate snippets: {len(snippets)}")
    print(f"Question relevance records: {len(question_relevance)}")
    print(f"Manifest saved: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
