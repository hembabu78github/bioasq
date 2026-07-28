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
PILOT_PATH = ROOT / "data" / "processed" / "pilot" / "bioasq11_pilot_80.json"
DERIVED_DIR = ROOT / "data" / "processed" / "stage3"
OUTPUT_DIR = ROOT / "outputs" / "stage3"
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


def choose_split(
    group_size: int,
    current: Counter[str],
    targets: dict[str, int],
    allowed: tuple[str, ...],
) -> str:
    deficits = {name: targets[name] - current[name] for name in allowed}
    return max(
        allowed,
        key=lambda name: (
            deficits[name] / max(targets[name], 1),
            deficits[name],
            {"train": 2, "dev": 1, "test": 0}[name],
        ),
    )


def main() -> int:
    if not DATA_PATH.exists() or not PILOT_PATH.exists():
        print("ERROR: Source dataset or pilot file is missing.")
        return 1

    questions = json.loads(DATA_PATH.read_text(encoding="utf-8")).get("questions", [])
    pilot_questions = json.loads(PILOT_PATH.read_text(encoding="utf-8")).get("questions", [])
    pilot_ids = {str(q.get("id", "")) for q in pilot_questions}

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
    pinned_group_count = 0
    pinned_question_count = 0

    for qtype in sorted(by_type_and_body):
        groups = list(by_type_and_body[qtype].values())
        pinned = [
            group
            for group in groups
            if any(str(q.get("id", "")) in pilot_ids for q in group)
        ]
        remaining = [
            group
            for group in groups
            if not any(str(q.get("id", "")) in pilot_ids for q in group)
        ]

        pinned_group_count += len(pinned)
        pinned_question_count += sum(len(group) for group in pinned)

        type_total = sum(len(group) for group in groups)
        targets = {
            "train": round(type_total * RATIOS["train"]),
            "dev": round(type_total * RATIOS["dev"]),
        }
        targets["test"] = type_total - targets["train"] - targets["dev"]

        current = Counter()
        for group in pinned:
            splits["dev"].extend(group)
            current["dev"] += len(group)

        rng.shuffle(remaining)
        for group in remaining:
            split = choose_split(
                group_size=len(group),
                current=current,
                targets=targets,
                allowed=("train", "dev", "test"),
            )
            splits[split].extend(group)
            current[split] += len(group)

    for split in splits:
        splits[split].sort(key=lambda q: str(q.get("id", "")))

    paths: dict[str, Path] = {}
    id_sets: dict[str, set[str]] = {}
    for split, records in splits.items():
        path = DERIVED_DIR / f"{split}_ids_v2.txt"
        ids = [str(q.get("id", "")) for q in records]
        path.write_text("\n".join(ids) + "\n", encoding="utf-8")
        paths[split] = path
        id_sets[split] = set(ids)

    pilot_membership = {
        split: len(pilot_ids & id_sets[split])
        for split in ("train", "dev", "test")
    }

    overlap = {
        "train_dev": len(id_sets["train"] & id_sets["dev"]),
        "train_test": len(id_sets["train"] & id_sets["test"]),
        "dev_test": len(id_sets["dev"] & id_sets["test"]),
    }

    body_splits: dict[str, set[str]] = defaultdict(set)
    for split, records in splits.items():
        for q in records:
            body_splits[normalize_body(str(q.get("body", "")))].add(split)
    body_leakage = {
        body: sorted(names)
        for body, names in body_splits.items()
        if len(names) > 1
    }

    counts_by_type = {}
    for split, records in splits.items():
        counts_by_type[split] = dict(
            sorted(Counter(str(q.get("type", "missing")).lower() for q in records).items())
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "v2",
        "supersedes": "outputs/stage2/split_manifest.json",
        "reason_for_repair": (
            "The 80-question pilot was selected before the provisional Stage 2 split. "
            "All pilot duplicate-groups are now pinned to development, and a new untouched "
            "test partition is created."
        ),
        "seed": SEED,
        "ratios": RATIOS,
        "method": (
            "Stratify by question type; group exact normalized duplicate bodies; pin all "
            "groups containing a pilot question to development; allocate all remaining "
            "groups deterministically toward 70/15/15 targets."
        ),
        "counts": {split: len(records) for split, records in splits.items()},
        "question_type_counts": counts_by_type,
        "pinned_pilot": {
            "pilot_id_count": len(pilot_ids),
            "pinned_duplicate_group_count": pinned_group_count,
            "pinned_question_count_including_exact_duplicates": pinned_question_count,
            "membership_after_repair": pilot_membership,
        },
        "id_overlap_counts": overlap,
        "duplicate_body_group_leakage_count": len(body_leakage),
        "duplicate_body_group_leakage_examples": list(body_leakage.items())[:20],
        "files": {
            split: {
                "relative_path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for split, path in paths.items()
        },
        "sealed_test_rule": (
            "The v2 test partition is untouched. No test IDs, answers, relevance labels or "
            "performance may be used for model, prompt, threshold, routing or graph-policy "
            "development."
        ),
    }

    out = OUTPUT_DIR / "split_manifest_v2.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    valid = (
        pilot_membership == {"train": 0, "dev": len(pilot_ids), "test": 0}
        and not any(overlap.values())
        and not body_leakage
        and sum(manifest["counts"].values()) == len(questions)
    )

    print(f"Split counts: {manifest['counts']}")
    print(f"Pilot membership: {pilot_membership}")
    print(f"Duplicate-body leakage groups: {len(body_leakage)}")
    print(f"Manifest saved: {out}")
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
