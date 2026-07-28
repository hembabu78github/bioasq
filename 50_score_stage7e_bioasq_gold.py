from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "stage7e_gold_scoring"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = (
    ROOT / "outputs" / "stage7d_answer_eval"
    / "stage7d_answer_eval_results.jsonl"
)
PROTOCOL_PATH = ROOT / "Stage7E_Gold_Scoring_Protocol.json"

PER_QUESTION_JSONL = OUTPUT_DIR / "stage7e_gold_per_question.jsonl"
PER_QUESTION_CSV = OUTPUT_DIR / "stage7e_gold_per_question.csv"
SUMMARY_PATH = OUTPUT_DIR / "stage7e_gold_summary.json"
ROUTE_DECISION_PATH = OUTPUT_DIR / "stage7e_text_route_decision.json"

ACCEPTED_RESULTS_SHA256 = "d28008c3cec4ef81bd6e2e98a0f7708f9c9606dd65c7607dd487e1ad8f972a66"
CONDITIONS = ("bge_text_only", "hybrid_text_only")
TYPE_ORDER = ("factoid", "list", "summary", "yesno")
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20260726

GENERIC_SUFFIXES = {
    "cancer", "disease", "disorder", "syndrome", "carcinoma",
    "tumor", "tumour", "protein", "gene", "mutation", "mutations",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_bioasq_json(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            ROOT / "data" / "raw" / "bioasq" / "training11b.json",
            ROOT / "data" / "raw" / "training11b.json",
            ROOT / "training11b.json",
            ROOT / "data" / "BioASQ" / "training11b.json",
        ]
    )
    for path in candidates:
        if path.exists():
            return path.resolve()

    matches = list(ROOT.rglob("training11b.json"))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        print("ERROR: Multiple training11b.json files found:")
        for path in matches:
            print("-", path)
        print("Pass the intended file using --bioasq-json.")
        raise SystemExit(2)

    print("ERROR: BioASQ training11b.json was not found.")
    print("Run again with:")
    print(
        r"python 50_score_stage7e_bioasq_gold.py "
        r'--bioasq-json "D:\path\to\training11b.json"'
    )
    raise SystemExit(2)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = (
        text.replace("α", " alpha ")
        .replace("β", " beta ")
        .replace("γ", " gamma ")
        .replace("δ", " delta ")
        .replace("κ", " kappa ")
    )
    text = text.lower()
    text = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def aliases(value: Any, lenient: bool) -> set[str]:
    raw = str(value).strip()
    output = {normalize_text(raw)}

    for outside, inside in re.findall(
        r"([^()]*)\(([^()]*)\)", raw
    ):
        if outside.strip():
            output.add(normalize_text(outside))
        if inside.strip():
            output.add(normalize_text(inside))

    if lenient:
        expanded = set(output)
        for item in list(output):
            tokens = item.split()
            trimmed = [
                token for token in tokens
                if token not in GENERIC_SUFFIXES
            ]
            if trimmed:
                expanded.add(" ".join(trimmed))
        output = expanded

    return {item for item in output if item}


def token_f1(left: str, right: str) -> float:
    a = left.split()
    b = right.split()
    if not a or not b:
        return 0.0
    common = Counter(a) & Counter(b)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(a)
    recall = overlap / len(b)
    return 2 * precision * recall / (precision + recall)


def values_match(predicted: Any, gold: Any, lenient: bool) -> bool:
    pred_aliases = aliases(predicted, lenient)
    gold_aliases = aliases(gold, lenient)
    if pred_aliases & gold_aliases:
        return True
    if not lenient:
        return False

    for pred in pred_aliases:
        for target in gold_aliases:
            if token_f1(pred, target) >= 0.85:
                return True
    return False


def parse_answer_items(answer: Any) -> list[str]:
    if isinstance(answer, list):
        items = answer
    else:
        text = str(answer).strip()
        if not text:
            return []
        items = None
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    items = list(parsed)
            except Exception:
                items = None
        if items is None:
            cleaned = re.sub(r"^\s*[-*•]\s*", "", text)
            if ";" in cleaned or "\n" in cleaned:
                items = re.split(r";|\n", cleaned)
            elif re.search(r"\b\d+[.)]\s+", cleaned):
                items = re.split(r"\b\d+[.)]\s+", cleaned)
            elif "," in cleaned and len(cleaned.split()) < 40:
                items = cleaned.split(",")
            else:
                items = [cleaned]

    output = []
    seen = set()
    for item in items:
        value = str(item).strip().strip("[]'\" ")
        if not value:
            continue
        key = normalize_text(value)
        if key and key not in seen:
            seen.add(key)
            output.append(value)
    return output


def flatten_factoid_gold(exact_answer: Any) -> list[str]:
    output = []
    if isinstance(exact_answer, list):
        for item in exact_answer:
            if isinstance(item, list):
                output.extend(str(value) for value in item)
            else:
                output.append(str(item))
    elif exact_answer not in (None, ""):
        output.append(str(exact_answer))
    return output


def list_gold_groups(exact_answer: Any) -> list[list[str]]:
    groups = []
    if not isinstance(exact_answer, list):
        return groups
    for item in exact_answer:
        if isinstance(item, list):
            groups.append([str(value) for value in item])
        else:
            groups.append([str(item)])
    return groups


def maximum_matching(
    predictions: list[str],
    gold_groups: list[list[str]],
    lenient: bool,
) -> int:
    edges = {
        index: [
            gold_index
            for gold_index, group in enumerate(gold_groups)
            if any(
                values_match(prediction, synonym, lenient)
                for synonym in group
            )
        ]
        for index, prediction in enumerate(predictions)
    }
    matched_gold: dict[int, int] = {}

    def augment(pred_index: int, visited: set[int]) -> bool:
        for gold_index in edges[pred_index]:
            if gold_index in visited:
                continue
            visited.add(gold_index)
            if (
                gold_index not in matched_gold
                or augment(matched_gold[gold_index], visited)
            ):
                matched_gold[gold_index] = pred_index
                return True
        return False

    matches = 0
    for pred_index in range(len(predictions)):
        if augment(pred_index, set()):
            matches += 1
    return matches


def factoid_scores(answer: str, exact_answer: Any) -> dict[str, float]:
    predictions = parse_answer_items(answer)
    gold = flatten_factoid_gold(exact_answer)

    def reciprocal_rank(lenient: bool) -> float:
        for rank, prediction in enumerate(predictions, start=1):
            if any(
                values_match(prediction, target, lenient)
                for target in gold
            ):
                return 1.0 / rank
        return 0.0

    return {
        "strict_score": reciprocal_rank(False),
        "primary_score": reciprocal_rank(True),
    }


def list_scores(answer: str, exact_answer: Any) -> dict[str, float]:
    predictions = parse_answer_items(answer)
    gold_groups = list_gold_groups(exact_answer)

    def f1(lenient: bool) -> float:
        matches = maximum_matching(
            predictions, gold_groups, lenient
        )
        precision = matches / len(predictions) if predictions else 0.0
        recall = matches / len(gold_groups) if gold_groups else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    return {
        "strict_score": f1(False),
        "primary_score": f1(True),
    }


def yesno_scores(answer: str, exact_answer: Any) -> dict[str, float]:
    normalized = normalize_text(answer)
    predicted = None
    if re.search(r"\byes\b", normalized):
        predicted = "yes"
    elif re.search(r"\bno\b", normalized):
        predicted = "no"
    gold = normalize_text(exact_answer)
    score = 1.0 if predicted and predicted == gold else 0.0
    return {"strict_score": score, "primary_score": score}


def lcs_length(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def summary_scores(answer: str, ideal_answer: Any) -> dict[str, float]:
    references = (
        [str(item) for item in ideal_answer]
        if isinstance(ideal_answer, list)
        else [str(ideal_answer)]
    )
    score = max(
        (rouge_l_f1(answer, reference) for reference in references),
        default=0.0,
    )
    return {"strict_score": score, "primary_score": score}


def score_answer(
    question_type: str,
    answer: str,
    exact_answer: Any,
    ideal_answer: Any,
) -> dict[str, float]:
    if question_type == "factoid":
        return factoid_scores(answer, exact_answer)
    if question_type == "list":
        return list_scores(answer, exact_answer)
    if question_type == "yesno":
        return yesno_scores(answer, exact_answer)
    if question_type == "summary":
        return summary_scores(answer, ideal_answer)
    raise ValueError(f"Unsupported question type: {question_type}")


def bootstrap_ci(
    paired_differences: list[float],
) -> tuple[float, float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    count = len(paired_differences)
    means = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [
            paired_differences[rng.randrange(count)]
            for _ in range(count)
        ]
        means.append(sum(sample) / count)
    means.sort()
    lower = means[int(0.025 * BOOTSTRAP_SAMPLES)]
    upper = means[int(0.975 * BOOTSTRAP_SAMPLES)]
    win_probability = sum(value > 0 for value in means) / len(means)
    return lower, upper, win_probability


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "stage7_id", "question_id", "question_type", "condition",
        "final_disposition", "deployed_answer", "raw_answer",
        "deployed_primary_score", "deployed_strict_score",
        "raw_primary_score", "raw_strict_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bioasq-json")
    args = parser.parse_args()

    missing = [
        str(path)
        for path in (RESULTS_PATH, PROTOCOL_PATH)
        if not path.exists()
    ]
    if missing:
        print("ERROR: Missing Stage 7E inputs:")
        for path in missing:
            print("-", path)
        return 1

    if sha256(RESULTS_PATH) != ACCEPTED_RESULTS_SHA256:
        raise RuntimeError(
            "Stage 7D results do not match the accepted SHA-256. "
            "Do not regenerate or alter the frozen results."
        )

    bioasq_path = find_bioasq_json(args.bioasq_json)
    bioasq_payload = json.loads(
        bioasq_path.read_text(encoding="utf-8")
    )
    gold_by_id = {
        str(question["id"]): question
        for question in bioasq_payload["questions"]
    }

    stage7_results = [
        row for row in load_jsonl(RESULTS_PATH)
        if row["condition"] in CONDITIONS
    ]
    stage7_results.sort(
        key=lambda row: (
            row["stage7_id"],
            CONDITIONS.index(row["condition"]),
        )
    )

    output_rows = []
    missing_gold = []
    for row in stage7_results:
        gold = gold_by_id.get(str(row["question_id"]))
        if not gold:
            missing_gold.append(row["question_id"])
            continue

        raw_answer = row["answer"]
        deployed_answer = (
            raw_answer
            if row["final_disposition"] == "release"
            else ""
        )
        deployed = score_answer(
            row["question_type"],
            deployed_answer,
            gold.get("exact_answer"),
            gold.get("ideal_answer"),
        )
        raw = score_answer(
            row["question_type"],
            raw_answer,
            gold.get("exact_answer"),
            gold.get("ideal_answer"),
        )

        output_rows.append(
            {
                "stage7_id": row["stage7_id"],
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "condition": row["condition"],
                "question": row["question"],
                "final_disposition": row["final_disposition"],
                "deployed_answer": deployed_answer,
                "raw_answer": raw_answer,
                "gold_exact_answer": gold.get("exact_answer"),
                "gold_ideal_answer": gold.get("ideal_answer"),
                "deployed_primary_score": deployed["primary_score"],
                "deployed_strict_score": deployed["strict_score"],
                "raw_primary_score": raw["primary_score"],
                "raw_strict_score": raw["strict_score"],
            }
        )

    if missing_gold:
        raise RuntimeError(
            "Missing BioASQ gold records for: "
            + ", ".join(sorted(set(missing_gold)))
        )

    grouped = defaultdict(lambda: defaultdict(list))
    for row in output_rows:
        grouped[row["condition"]][row["question_type"]].append(row)

    condition_summary = {}
    for condition in CONDITIONS:
        by_type = {}
        for question_type in TYPE_ORDER:
            rows = grouped[condition][question_type]
            by_type[question_type] = {
                "question_count": len(rows),
                "release_count": sum(
                    row["final_disposition"] == "release"
                    for row in rows
                ),
                "coverage": mean([
                    1.0 if row["final_disposition"] == "release" else 0.0
                    for row in rows
                ]),
                "deployed_primary_mean": mean([
                    row["deployed_primary_score"] for row in rows
                ]),
                "deployed_strict_mean": mean([
                    row["deployed_strict_score"] for row in rows
                ]),
                "raw_primary_mean": mean([
                    row["raw_primary_score"] for row in rows
                ]),
                "raw_strict_mean": mean([
                    row["raw_strict_score"] for row in rows
                ]),
            }

        condition_summary[condition] = {
            "by_type": by_type,
            "deployed_primary_composite": mean([
                by_type[question_type]["deployed_primary_mean"]
                for question_type in TYPE_ORDER
            ]),
            "deployed_strict_composite": mean([
                by_type[question_type]["deployed_strict_mean"]
                for question_type in TYPE_ORDER
            ]),
            "raw_primary_composite": mean([
                by_type[question_type]["raw_primary_mean"]
                for question_type in TYPE_ORDER
            ]),
            "raw_strict_composite": mean([
                by_type[question_type]["raw_strict_mean"]
                for question_type in TYPE_ORDER
            ]),
            "overall_coverage": mean([
                by_type[question_type]["coverage"]
                for question_type in TYPE_ORDER
            ]),
        }

    rows_by_key = {
        (row["stage7_id"], row["condition"]): row
        for row in output_rows
    }
    paired_differences = []
    paired_rows = []
    for stage7_id in sorted({row["stage7_id"] for row in output_rows}):
        bge = rows_by_key[(stage7_id, "bge_text_only")]
        hybrid = rows_by_key[(stage7_id, "hybrid_text_only")]
        difference = (
            hybrid["deployed_primary_score"]
            - bge["deployed_primary_score"]
        )
        paired_differences.append(difference)
        paired_rows.append(
            {
                "stage7_id": stage7_id,
                "question_type": bge["question_type"],
                "bge_deployed_primary_score": bge[
                    "deployed_primary_score"
                ],
                "hybrid_deployed_primary_score": hybrid[
                    "deployed_primary_score"
                ],
                "hybrid_minus_bge": difference,
            }
        )

    lower, upper, win_probability = bootstrap_ci(
        paired_differences
    )
    observed_difference = mean(paired_differences)

    if lower > 0:
        selected_route = "hybrid_text_only"
        decision_basis = (
            "Hybrid selected because the 95% paired-bootstrap CI "
            "for hybrid-minus-BGE is entirely above zero."
        )
    elif upper < 0:
        selected_route = "bge_text_only"
        decision_basis = (
            "BGE selected because the 95% paired-bootstrap CI "
            "for hybrid-minus-BGE is entirely below zero."
        )
    else:
        selected_route = "bge_text_only"
        decision_basis = (
            "The 95% paired-bootstrap CI includes zero; BGE is "
            "selected by the frozen parsimony tie rule."
        )

    decision = {
        "development_only": True,
        "sealed_test_accessed": False,
        "selected_final_text_route": selected_route,
        "decision_basis": decision_basis,
        "hybrid_minus_bge_observed_mean": observed_difference,
        "hybrid_minus_bge_bootstrap_95_ci": [lower, upper],
        "bootstrap_hybrid_win_probability": win_probability,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "graph_route_status": "excluded_from_main_architecture",
        "next_gate": (
            "Review and accept the gold-scoring outputs, freeze the "
            "selected text-only route, and only then prepare the "
            "one-time sealed-test evaluation."
        ),
    }

    with PER_QUESTION_JSONL.open(
        "w", encoding="utf-8"
    ) as handle:
        for row in output_rows:
            handle.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )
    write_csv(PER_QUESTION_CSV, output_rows)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "sealed_test_accessed": False,
        "bioasq_json_path": str(bioasq_path),
        "bioasq_json_sha256": sha256(bioasq_path),
        "accepted_stage7d_results_sha256": ACCEPTED_RESULTS_SHA256,
        "question_count": 24,
        "logical_arm_count": len(output_rows),
        "conditions": condition_summary,
        "paired_comparison": {
            "hybrid_minus_bge_observed_mean": observed_difference,
            "bootstrap_95_ci": [lower, upper],
            "bootstrap_hybrid_win_probability": win_probability,
            "paired_questions": paired_rows,
        },
        "route_decision": decision,
        "scientific_note": (
            "The development sample is balanced but small. The "
            "predeclared paired-bootstrap and parsimony rule select "
            "the final text-only route without accessing the test set."
        ),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    ROUTE_DECISION_PATH.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Stage 7E BioASQ gold scoring completed.")
    print(f"- BioASQ file: {bioasq_path}")
    print(f"- Scored logical arms: {len(output_rows)}")
    print(
        "- BGE deployed composite: "
        f"{condition_summary['bge_text_only']['deployed_primary_composite']:.6f}"
    )
    print(
        "- Hybrid deployed composite: "
        f"{condition_summary['hybrid_text_only']['deployed_primary_composite']:.6f}"
    )
    print(
        "- Hybrid-minus-BGE 95% CI: "
        f"[{lower:.6f}, {upper:.6f}]"
    )
    print(f"- Selected route: {selected_route}")
    print(f"- Summary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
