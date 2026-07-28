# Stage 7D-A Protocol Note

## Purpose

Freeze the deployment-available routing decision for every question in the
balanced 24-question development sample before generation begins.

## Candidate evidence

- Hybrid top 20 supplies the graph-extraction candidate pool.
- Hybrid top five is the default text evidence.
- Graph-selected evidence contains exactly five snippets.

## Deterministic list-question eligibility

A list question is graph-routed only when all conditions hold:

1. At least two relevant answer aspects are extracted.
2. At least two relevant graph items are present.
3. At least one relevant item is not covered by hybrid top five.
4. At least one selected novel snippet has positive marginal graph coverage.
5. The selected evidence set changes.
6. No zero-coverage novel filler is selected.

The model's `graph_sufficient` flag is retained for audit but is not the sole
routing gate.

## Evidence closure

The route manifest may store the extracted graph for provenance. Downstream
generation and verification will receive only the selected five text snippets.

## Integrity

- Development sample only.
- No answer or verifier calls.
- No sealed-test access.
- No threshold or prompt changes after results are observed.
