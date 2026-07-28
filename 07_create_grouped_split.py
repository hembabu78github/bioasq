from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "bioasq11" / "training11b.json"
DERIVED_DIR = ROOT / "data" / "processed" / "stage2"
OUTPUT_DIR = ROOT / "outputs" / "stage2"
DERIVED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260725
RATIOS = {"train": 0.70, "dev": 0.15, "test": 0.15}


def normalize_body(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def allocate_grouped(
    groups: list[list[dict[str, Any]]],
    rng: random.Random,
) -> dict[str, list[dict[str, Any]]]:
    groups = list(groups)
    rng.shuffle(groups)
    total = sum(len(group) for group in groups)
    targets = {
        "train": round(total * RATIOS["train"]),
        "dev": round(total * RATIOS["dev"]),
    }
    targets["test"] = total - targets["train"] - targets["dev"]

    assigned = {"train": [], "dev": [], "test": []}
    counts = Counter()

    for group in groups:
        deficits = {
            split: targets[split] - counts[split]
            for split in ("train", "dev", "test")
        }
        # Prefer the split with the largest remaining relative deficit.
        split = max(
            deficits,
            key=lambda name: (
                deficits[name] / max(targets[name], 1),
                deficits[name],
                {"train": 2, "dev": 1, "test": 0}[name],
            ),
        )
        assigned[split].extend(group)
        counts[split] += len(group)

    return assigned


def main() -> int:
    if not DATA_PATH.exists():
        print(f"ERROR: Missing source dataset: {DATA_PATH}")
        return 1

    questions = json.loads(DATA_PATH.read_text(encoding="utf-8")).get("questions", [])
    by_type_and_body: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for q in questions:
        if not isinstance(q, dict):
            continue
        qtype = str(q.get("type", "missing")).strip().lower()
        body_key = normalize_body(str(q.get("body", "")))
        by_type_and_body[qtype][body_key].append(q)

    rng = random.Random(SEED)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}

    for qtype in sorted(by_type_and_body):
        groups = list(by_type_and_body[qtype].values())
        assigned = allocate_grouped(groups, rng)
        for split in splits:
            splits[split].extend(assigned[split])

    for split in splits:
        splits[split].sort(key=lambda q: str(q.get("id", "")))

    paths = {}
    id_sets = {}
    for split, records in splits.items():
        path = DERIVED_DIR / f"{split}_ids.txt"
        ids = [str(q.get("id")) for q in records]
        path.write_text("\n".join(ids) + "\n", encoding="utf-8")
        paths[split] = path
        id_sets[split] = set(ids)

    overlap = {
        "train_dev": len(id_sets["train"] & id_sets["dev"]),
        "train_test": len(id_sets["train"] & id_sets["test"]),
        "dev_test": len(id_sets["dev"] & id_sets["test"]),
    }

    body_to_splits: dict[str, set[str]] = defaultdict(set)
    for split, records in splits.items():
        for q in records:
            body_to_splits[normalize_body(str(q.get("body", "")))].add(split)
    duplicate_group_leakage = {
        body: sorted(split_names)
        for body, split_names in body_to_splits.items()
        if len(split_names) > 1
    }

    type_counts = {}
    for split, records in splits.items():
        type_counts[split] = dict(
            sorted(Counter(str(q.get("type", "missing")).lower() for q in records).items())
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "ratios": RATIOS,
        "method": (
            "Stratify by BioASQ question type; group exact normalized duplicate bodies; "
            "assign complete groups deterministically to train/dev/test."
        ),
        "counts": {split: len(records) for split, records in splits.items()},
        "question_type_counts": type_counts,
        "id_overlap_counts": overlap,
        "duplicate_body_group_leakage_count": len(duplicate_group_leakage),
        "duplicate_body_group_leakage_examples": list(duplicate_group_leakage.items())[:20],
        "files": {
            split: {
                "relative_path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for split, path in paths.items()
        },
        "sealed_test_rule": (
            "All threshold tuning, routing-policy development and prompt/model selection "
            "must use train/dev data only. Test IDs and labels are not to be used for tuning."
        ),
    }

    manifest_path = OUTPUT_DIR / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Split counts: {manifest['counts']}")
    print(f"Duplicate-body leakage groups: {manifest['duplicate_body_group_leakage_count']}")
    print(f"Manifest saved: {manifest_path}")
    return 0 if not any(overlap.values()) and not duplicate_group_leakage else 2


if __name__ == "__main__":
    raise SystemExit(main())
