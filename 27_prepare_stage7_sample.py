from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage7_common import ROOT, load_jsonl

DIAGNOSTICS = (
    ROOT / "data" / "processed" / "stage4"
    / "dev_retrieval_diagnostics_private.jsonl"
)
PILOT_MANIFEST = ROOT / "outputs" / "stage4" / "graph_claim_pilot_manifest.json"
PILOT_IDS = (
    ROOT / "data" / "processed" / "stage4" / "graph_claim_pilot_ids.json"
)
OUTPUT_DIR = ROOT / "outputs" / "stage7_sampling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_CSV = OUTPUT_DIR / "stage7_question_sample.csv"
SAMPLE_JSON = OUTPUT_DIR / "stage7_question_sample.json"
SUMMARY_JSON = OUTPUT_DIR / "stage7_sampling_summary.json"
EXCLUSION_JSON = OUTPUT_DIR / "stage7_exclusion_audit.json"
SMOKE_CSV = OUTPUT_DIR / "stage7_smoke_sample.csv"
SMOKE_JSON = OUTPUT_DIR / "stage7_smoke_sample.json"
SMOKE_MANIFEST = OUTPUT_DIR / "stage7_smoke_manifest.json"

QUESTION_TYPES = ["factoid", "list", "summary", "yesno"]
UNCERTAINTY_LEVELS = ["low", "medium", "high"]
ROLE_GRAPH = "graph_suitable_candidate"
ROLE_CONTROL = "non_graph_control_candidate"

GRAPH_TERMS = {
    "mechanism": 4,
    "pathway": 4,
    "interact": 4,
    "interaction": 4,
    "inhibit": 4,
    "activate": 4,
    "target": 3,
    "receptor": 3,
    "gene": 3,
    "protein": 3,
    "mutation": 3,
    "mutated": 3,
    "associated": 3,
    "association": 3,
    "cause": 3,
    "causes": 3,
    "role": 2,
    "effect": 2,
    "affect": 2,
    "regulate": 3,
    "relationship": 4,
    "linked": 3,
    "biomarker": 3,
    "mediated": 3,
    "derived": 2,
    "responsible": 2,
}


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def graph_score(question: str) -> int:
    normalized = normalize_question(question)
    tokens = normalized.split()
    score = 0
    for term, weight in GRAPH_TERMS.items():
        if term in normalized:
            score += weight
    if "how does" in normalized or "what is the role" in normalized:
        score += 4
    if "which" in tokens and any(
        term in normalized
        for term in ("target", "gene", "protein", "receptor", "pathway")
    ):
        score += 2
    if len(tokens) >= 18:
        score += 1
    return score


def prior_pilot_ids() -> set[str]:
    if PILOT_MANIFEST.exists():
        data = json.loads(PILOT_MANIFEST.read_text(encoding="utf-8"))
        return {
            str(row["question_id"])
            for row in data.get("selected_questions", [])
            if row.get("question_id")
        }
    if PILOT_IDS.exists():
        data = json.loads(PILOT_IDS.read_text(encoding="utf-8"))
        return {
            str(row["question_id"])
            for row in data.get("questions", [])
            if row.get("question_id")
        }
    return set()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_cell(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(candidates) < 2:
        raise RuntimeError("Fewer than two eligible questions in a quota cell.")

    graph_pick = sorted(
        candidates,
        key=lambda r: (
            -r["_graph_score"],
            -int(r.get("hybrid_improves_rank_by_3_or_more", 0)),
            -int(r.get("bge_failure_at_5", 0)),
            -float(r.get("retrieval_uncertainty_score", 0)),
            r["question_id"],
        ),
    )[0]

    control_pool = [
        row for row in candidates if row["question_id"] != graph_pick["question_id"]
    ]
    control_pick = sorted(
        control_pool,
        key=lambda r: (
            r["_graph_score"],
            int(r.get("bge_failure_at_5", 0)),
            -int(r.get("bge_hit_at_5", 0)),
            r["question_id"],
        ),
    )[0]
    return graph_pick, control_pick


def select_smoke(sample: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_type_role_uncertainty = {
        (
            row["question_type"],
            row["graph_role"],
            row["retrieval_uncertainty_label"],
        ): row
        for row in sample
    }
    target = {"low": 3, "medium": 2, "high": 3}

    options_by_type = {}
    for qtype in QUESTION_TYPES:
        options = []
        for g_unc in UNCERTAINTY_LEVELS:
            for c_unc in UNCERTAINTY_LEVELS:
                if g_unc == c_unc:
                    continue
                graph_row = by_type_role_uncertainty.get(
                    (qtype, ROLE_GRAPH, g_unc)
                )
                control_row = by_type_role_uncertainty.get(
                    (qtype, ROLE_CONTROL, c_unc)
                )
                if graph_row and control_row:
                    options.append((graph_row, control_row))
        if not options:
            raise RuntimeError(f"No smoke pair available for {qtype}.")
        options_by_type[qtype] = options

    exact = []
    fallback = []
    for combo in itertools.product(
        *(options_by_type[qtype] for qtype in QUESTION_TYPES)
    ):
        rows = [row for pair in combo for row in pair]
        counts = Counter(row["retrieval_uncertainty_label"] for row in rows)
        distance = sum(abs(counts[level] - target[level]) for level in target)
        key = tuple(row["question_id"] for row in rows)
        fallback.append((distance, key, rows))
        if distance == 0:
            exact.append((key, rows))

    if exact:
        return sorted(exact, key=lambda item: item[0])[0][1]
    return sorted(fallback, key=lambda item: (item[0], item[1]))[0][2]


def main() -> int:
    if not DIAGNOSTICS.exists():
        print(f"ERROR: Missing development diagnostics: {DIAGNOSTICS}")
        return 1

    excluded_ids = prior_pilot_ids()
    diagnostics = load_jsonl(DIAGNOSTICS)

    eligible = []
    seen_bodies = set()
    exclusions = Counter()
    excluded_examples = []

    for row in diagnostics:
        qid = str(row.get("question_id", ""))
        qtype = str(row.get("question_type", "")).lower()
        uncertainty = str(row.get("retrieval_uncertainty_label", "")).lower()
        question = str(row.get("question", "")).strip()
        body_key = normalize_question(question)

        if qid in excluded_ids:
            exclusions["prior_graph_pilot"] += 1
            excluded_examples.append(
                {"question_id": qid, "reason": "prior_graph_pilot"}
            )
            continue
        if qtype not in QUESTION_TYPES or uncertainty not in UNCERTAINTY_LEVELS:
            exclusions["invalid_type_or_uncertainty"] += 1
            continue
        if not body_key or body_key in seen_bodies:
            exclusions["duplicate_normalized_body"] += 1
            continue

        seen_bodies.add(body_key)
        enriched = dict(row)
        enriched["_graph_score"] = graph_score(question)
        eligible.append(enriched)

    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        cells[
            (row["question_type"], row["retrieval_uncertainty_label"])
        ].append(row)

    selected = []
    for qtype in QUESTION_TYPES:
        for uncertainty in UNCERTAINTY_LEVELS:
            graph_pick, control_pick = choose_cell(cells[(qtype, uncertainty)])
            for role, row in (
                (ROLE_GRAPH, graph_pick),
                (ROLE_CONTROL, control_pick),
            ):
                selected.append(
                    {
                        "stage7_id": "",
                        "question_id": row["question_id"],
                        "question_type": row["question_type"],
                        "question": row["question"],
                        "retrieval_uncertainty_label": row[
                            "retrieval_uncertainty_label"
                        ],
                        "retrieval_uncertainty_score": row[
                            "retrieval_uncertainty_score"
                        ],
                        "graph_role": role,
                        "graph_suitability_heuristic_score": row["_graph_score"],
                        "selection_reason": (
                            f"predeclared_{role}; balanced 2-per-type-by-"
                            "uncertainty cell; selected before Stage 7 generation"
                        ),
                        "bge_hit_at_5": row.get("bge_hit_at_5"),
                        "hybrid_hit_at_5": row.get("hybrid_hit_at_5"),
                        "bge_first_relevant_rank_at_20": row.get(
                            "bge_first_relevant_rank_at_20"
                        ),
                        "hybrid_first_relevant_rank_at_20": row.get(
                            "hybrid_first_relevant_rank_at_20"
                        ),
                    }
                )

    selected.sort(
        key=lambda r: (
            QUESTION_TYPES.index(r["question_type"]),
            UNCERTAINTY_LEVELS.index(r["retrieval_uncertainty_label"]),
            0 if r["graph_role"] == ROLE_GRAPH else 1,
            r["question_id"],
        )
    )
    for index, row in enumerate(selected, start=1):
        row["stage7_id"] = f"S7Q-{index:03d}"

    if len(selected) != 24:
        raise RuntimeError(f"Expected 24 questions, selected {len(selected)}.")

    sample_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "sealed_test_accessed": False,
        "selection_frozen_before_generation": True,
        "questions": selected,
    }
    SAMPLE_JSON.write_text(
        json.dumps(sample_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(SAMPLE_CSV, selected)

    smoke_rows = select_smoke(selected)
    smoke_rows.sort(
        key=lambda r: (
            QUESTION_TYPES.index(r["question_type"]),
            0 if r["graph_role"] == ROLE_GRAPH else 1,
        )
    )
    smoke_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "sealed_test_accessed": False,
        "question_count": len(smoke_rows),
        "questions": smoke_rows,
    }
    SMOKE_JSON.write_text(
        json.dumps(smoke_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(SMOKE_CSV, smoke_rows)

    type_counts = Counter(row["question_type"] for row in selected)
    uncertainty_counts = Counter(
        row["retrieval_uncertainty_label"] for row in selected
    )
    role_counts = Counter(row["graph_role"] for row in selected)
    smoke_type_counts = Counter(row["question_type"] for row in smoke_rows)
    smoke_uncertainty_counts = Counter(
        row["retrieval_uncertainty_label"] for row in smoke_rows
    )
    smoke_role_counts = Counter(row["graph_role"] for row in smoke_rows)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_question_count": len(selected),
        "question_type_counts": dict(sorted(type_counts.items())),
        "uncertainty_counts": dict(sorted(uncertainty_counts.items())),
        "graph_role_counts": dict(sorted(role_counts.items())),
        "prior_graph_pilot_question_count_excluded": len(excluded_ids),
        "prior_graph_pilot_overlap_after_selection": len(
            {row["question_id"] for row in selected} & excluded_ids
        ),
        "sample_sha256": hashlib.sha256(SAMPLE_JSON.read_bytes()).hexdigest(),
        "scientific_note": (
            "Graph-suitable and control roles are deterministic pre-experiment "
            "sampling strata based on question wording, not observed GraphRAG performance."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    exclusion_audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_diagnostic_count": len(diagnostics),
        "eligible_count_after_exclusions": len(eligible),
        "exclusion_counts": dict(sorted(exclusions.items())),
        "excluded_examples": excluded_examples[:30],
        "prior_pilot_ids": sorted(excluded_ids),
        "sealed_test_accessed": False,
    }
    EXCLUSION_JSON.write_text(
        json.dumps(exclusion_audit, indent=2), encoding="utf-8"
    )

    smoke_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "question_count": len(smoke_rows),
        "question_type_counts": dict(sorted(smoke_type_counts.items())),
        "uncertainty_counts": dict(sorted(smoke_uncertainty_counts.items())),
        "graph_role_counts": dict(sorted(smoke_role_counts.items())),
        "four_conditions": [
            "bge_text_only",
            "hybrid_text_only",
            "graph_reranked",
            "risk_adaptive_agentic",
        ],
        "expected_answer_count": len(smoke_rows) * 4,
        "smoke_sample_sha256": hashlib.sha256(
            SMOKE_JSON.read_bytes()
        ).hexdigest(),
        "do_not_run_until_sample_reviewed": True,
    }
    SMOKE_MANIFEST.write_text(
        json.dumps(smoke_manifest, indent=2), encoding="utf-8"
    )

    print("Stage 7 development sample prepared.")
    print(f"- Full sample: {len(selected)} questions")
    print(f"- Smoke sample: {len(smoke_rows)} questions")
    print(f"- Full sample CSV: {SAMPLE_CSV}")
    print(f"- Sampling summary: {SUMMARY_JSON}")
    print(f"- Smoke manifest: {SMOKE_MANIFEST}")
    print("")
    print("STOP HERE. Upload the three requested sampling files for review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
