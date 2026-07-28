# Stage 4 — Retrieval Diagnostics, Local GraphRAG, and Claim-Verification Pilot

Stage 3 established the retrieval foundation:

- BGE-small achieved 100% Hit@20 and 0.893 MRR@20 on the 80-question
  development pilot.
- BM25+BGE reciprocal-rank fusion produced the best first-rank performance
  (88.75% Hit@1 and 0.918 MRR@20), but with higher latency.
- PubMedBERT underperformed BGE-small and will not be used in the proposed
  pipeline.
- The repaired v2 test partition is untouched.

Stage 4 has two purposes:

1. Evaluate retrieval-diagnostic uncertainty on the complete 709-question
   development partition, without accessing the test set.
2. Run a small structured GraphRAG and claim-verification feasibility pilot on
   12 development questions.

## Stage 4 workflow

### Part A — Development retrieval diagnostics

The script evaluates BM25, BGE-small and BM25+BGE fusion on all 709 development
questions. It records deployment-available features such as:

- BM25 and BGE score margins;
- lexical/dense top-k overlap;
- agreement on the top-ranked snippet;
- query length and question type;
- the existing query-only evidence-risk prior.

A new **retrieval uncertainty score** is calculated without using gold answers
or relevance labels. Gold relevance is used only after ranking to assess
whether the score predicts retrieval difficulty.

### Part B — GraphRAG and claim verification

A deterministic 12-question development pilot is selected, with three questions
from each BioASQ type and a mix of high-uncertainty, difficult and control
questions.

For each question, the pipeline:

1. Selects BGE or hybrid BGE using the provisional uncertainty rule.
2. Takes the top six retrieved snippets.
3. Extracts a query-focused local biomedical knowledge graph.
4. Produces both:
   - a text-only evidence-grounded answer;
   - a graph-assisted answer.
5. Splits each answer into atomic claims.
6. Verifies every claim as:
   - supported;
   - contradicted;
   - insufficient evidence.
7. Applies a deterministic abstention rule.
8. Saves the complete audit trail.

This is a feasibility pilot, not the final GraphRAG evaluation.

## Groq calls

Stage 4 uses:

```text
openai/gpt-oss-20b
```

There are three Groq calls per question:

1. graph extraction;
2. dual answer and atomic-claim generation;
3. claim verification.

For 12 questions, the expected maximum is 36 calls, excluding retries.

## Installation location

Copy all files from this ZIP into:

```text
D:\Prog\JMS_RAG
```

Keep all existing project files, cached BGE embeddings and `.env`.

## Before running

Close memory-heavy applications. Do not delete:

```text
models\stage3_embeddings\bge_small_corpus_embeddings.npy
models\stage3_embeddings\bge_small_corpus_ids.json
```

## Run

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\RUN_STAGE4.ps1
```

Or:

```powershell
python 15_analyze_dev_retrieval_diagnostics.py
python 16_select_graph_claim_pilot.py
python 17_run_graph_claim_pilot.py
python 18_validate_stage4.py
```

## Upload after completion

Upload these eight files:

```text
outputs\stage4\dev_retrieval_diagnostics_summary.json
outputs\stage4\dev_retrieval_by_type.csv
outputs\stage4\graph_claim_pilot_manifest.json
outputs\stage4\graph_claim_pilot_summary.json
outputs\stage4\graph_claim_pilot_by_type.csv
outputs\stage4\graph_claim_pilot_results.jsonl
outputs\stage4\graph_claim_examples.jsonl
outputs\stage4\stage4_validation_report.json
```

Do not upload `.env`, the BGE embeddings, split-ID files, the complete candidate
corpus, or the private 709-question ranking file unless requested.

## Interpretation limits

Stage 4 does not establish final claim-verification accuracy because an
independent manually labelled verification set has not yet been created. It
tests:

- pipeline feasibility;
- schema compliance;
- provenance completeness;
- graph/claim generation behavior;
- abstention mechanics;
- whether graph-assisted answers appear to change support outcomes.

A later stage will create a blinded human-audit sample before final evaluation.
