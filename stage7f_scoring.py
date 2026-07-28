from __future__ import annotations

import ast
import re
import unicodedata
from collections import Counter
from typing import Any

GENERIC_SUFFIXES = {
    "cancer", "disease", "disorder", "syndrome", "carcinoma",
    "tumor", "tumour", "protein", "gene", "mutation", "mutations",
}


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
    for outside, inside in re.findall(r"([^()]*)\(([^()]*)\)", raw):
        if outside.strip():
            output.add(normalize_text(outside))
        if inside.strip():
            output.add(normalize_text(inside))
    if lenient:
        expanded = set(output)
        for item in list(output):
            trimmed = [
                token for token in item.split()
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
    overlap = sum((Counter(a) & Counter(b)).values())
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
    return any(
        token_f1(pred, target) >= 0.85
        for pred in pred_aliases
        for target in gold_aliases
    )


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
        key = normalize_text(value)
        if value and key and key not in seen:
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
    if not isinstance(exact_answer, list):
        return []
    return [
        [str(value) for value in item]
        if isinstance(item, list)
        else [str(item)]
        for item in exact_answer
    ]


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
    matched_gold = {}

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

    return sum(
        augment(pred_index, set())
        for pred_index in range(len(predictions))
    )


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
        if precision + recall else 0.0
    )


def score_answer(
    question_type: str,
    answer: str,
    exact_answer: Any,
    ideal_answer: Any,
) -> dict[str, float]:
    if question_type == "factoid":
        predictions = parse_answer_items(answer)
        gold = flatten_factoid_gold(exact_answer)

        def rr(lenient: bool) -> float:
            for rank, prediction in enumerate(predictions, start=1):
                if any(
                    values_match(prediction, target, lenient)
                    for target in gold
                ):
                    return 1.0 / rank
            return 0.0

        return {"primary_score": rr(True), "strict_score": rr(False)}

    if question_type == "list":
        predictions = parse_answer_items(answer)
        gold_groups = list_gold_groups(exact_answer)

        def f1(lenient: bool) -> float:
            matches = maximum_matching(
                predictions, gold_groups, lenient
            )
            precision = matches / len(predictions) if predictions else 0.0
            recall = matches / len(gold_groups) if gold_groups else 0.0
            return (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )

        return {"primary_score": f1(True), "strict_score": f1(False)}

    if question_type == "yesno":
        normalized = normalize_text(answer)
        predicted = (
            "yes" if re.search(r"\byes\b", normalized)
            else "no" if re.search(r"\bno\b", normalized)
            else None
        )
        gold = normalize_text(exact_answer)
        score = 1.0 if predicted and predicted == gold else 0.0
        return {"primary_score": score, "strict_score": score}

    if question_type == "summary":
        references = (
            [str(item) for item in ideal_answer]
            if isinstance(ideal_answer, list)
            else [str(ideal_answer)]
        )
        score = max(
            (
                rouge_l_f1(answer, reference)
                for reference in references
            ),
            default=0.0,
        )
        return {"primary_score": score, "strict_score": score}

    raise ValueError(f"Unsupported question type: {question_type}")
