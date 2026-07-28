# Stage 2 — Leakage-Safe Corpus, Grouped Split, Evidence-Risk Prior, and BM25 Pilot

This stage converts the audited BioASQ11 file into a reproducible closed-corpus
retrieval benchmark for development.

It performs four tasks:

1. Builds a deduplicated candidate snippet corpus.
2. Stores question-to-gold-snippet relevance separately from the retriever input.
3. Creates a deterministic, question-type-stratified, duplicate-group-safe
   train/development/test split.
4. Runs a BM25 retrieval pilot on the 80-question development pilot.

## Important scientific rule

The BioASQ JSON contains question-linked gold snippets. The retriever must never
receive the gold snippets selected specifically for the current question.

This stage therefore creates:

- `candidate_snippets.jsonl`: global corpus available to every query;
- `gold_relevance.jsonl`: hidden relevance mapping used only for evaluation.

The candidate corpus does not include question IDs. This prevents the retrieval
code from using question-to-gold associations as input.

This remains a **closed snippet-bank benchmark**, not full-PubMed retrieval. The
manuscript will state that limitation explicitly.

## Installation location

Copy all files from this ZIP into the existing project folder:

```text
D:\Prog\JMS_RAG
```

Keep the existing `.env`, `.venv`, `data`, and `outputs` folders.

## Run

Open PowerShell:

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
```

Run:

```powershell
python 06_build_closed_snippet_corpus.py
python 07_create_grouped_split.py
python 08_create_evidence_risk_prior.py
python 09_run_bm25_pilot.py
python 10_validate_stage2.py
```

Or run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\RUN_STAGE2.ps1
```

No Groq call is made during Stage 2.

## Expected runtime

Approximate CPU time: several minutes, depending on the laptop and antivirus
scanning. The BM25 pilot searches about 52,000 candidate snippets for 80 pilot
questions.

## Upload after completion

Upload these seven files:

```text
outputs\stage2\closed_corpus_manifest.json
outputs\stage2\split_manifest.json
outputs\stage2\evidence_risk_summary.json
outputs\stage2\bm25_pilot_summary.json
outputs\stage2\bm25_pilot_by_type.csv
outputs\stage2\bm25_pilot_results.jsonl
outputs\stage2\stage2_validation_report.json
```

Do not upload the large candidate corpus unless requested.

## Do not inspect or modify

Please do not manually edit:

- `train_ids.txt`
- `dev_ids.txt`
- `test_ids.txt`
- `candidate_snippets.jsonl`
- `gold_relevance.jsonl`

The final test set is now being sealed. Later tuning will use development data
only.
