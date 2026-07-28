from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "bioasq11" / "training11b.json"
PILOT_DIR = ROOT / "data" / "processed" / "pilot"
OUTPUT_DIR = ROOT / "outputs" / "stage1"
PILOT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260725
PER_TYPE = 20


def norm_body(q: dict[str, Any]) -> str:
    return " ".join(str(q.get("body", "")).lower().split())


def main() -> int:
    if not DATA_PATH.exists():
        print(f"ERROR: Missing dataset: {DATA_PATH}")
        return 1

    questions = json.loads(DATA_PATH.read_text(encoding="utf-8")).get("questions", [])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()

    for q in questions:
        if not isinstance(q, dict):
            continue
        body = norm_body(q)
        if not body or body in seen:
            continue
        seen.add(body)
        grouped[str(q.get("type", "missing")).lower().strip()].append(q)

    rng = random.Random(SEED)
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for qtype in sorted(grouped):
        candidates = list(grouped[qtype])
        rng.shuffle(candidates)
        chosen = candidates[: min(PER_TYPE, len(candidates))]
        selected.extend(chosen)
        counts[qtype] = len(chosen)

    selected.sort(key=lambda q: (str(q.get("type", "")), str(q.get("id", ""))))

    pilot_path = PILOT_DIR / "bioasq11_pilot_80.json"
    pilot_path.write_text(
        json.dumps({"questions": selected}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "selection_rule": (
            "Keep one instance per exact normalised question body, shuffle "
            f"deterministically within each type, select up to {PER_TYPE} per type."
        ),
        "selected_counts": counts,
        "total_selected": len(selected),
        "pilot_relative_path": str(pilot_path.relative_to(ROOT)),
        "pilot_sha256": hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
        "note": "Development pilot only; not a final evaluation split.",
    }

    manifest_path = OUTPUT_DIR / "bioasq11_pilot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Pilot questions: {len(selected)}")
    print(f"Counts: {counts}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
