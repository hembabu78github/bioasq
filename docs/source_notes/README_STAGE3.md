# Stage 3 — Repair the Development/Test Boundary and Compare Dense/Hybrid Retrieval

Stage 2 passed all structural leakage checks and produced a strong BM25
development baseline. One protocol issue must now be corrected:

The original 80-question pilot was selected before the train/development/test
split was created. Some pilot questions could therefore have entered the
provisional test split. Stage 3 replaces that split and pins the complete pilot
duplicate-groups to the development set. A new test partition is then created
and sealed before any dense-retrieval tuning.

Stage 3 then compares:

1. BM25 lexical retrieval
2. BGE-small dense retrieval
3. PubMedBERT biomedical dense retrieval
4. BM25 + BGE reciprocal-rank fusion
5. BM25 + PubMedBERT reciprocal-rank fusion

The comparison is performed only on the 80-question development pilot.

## Models

- `BAAI/bge-small-en-v1.5`
- `NeuML/pubmedbert-base-embeddings`

The scripts record the exact Hugging Face revision visible at run time,
embedding dimension, sequence length, package versions, encoding time, index
size and search latency.

## Installation location

Copy all files from this ZIP into the existing project folder:

```text
D:\Prog\JMS_RAG
```

Keep the existing `.venv`, `.env`, `data`, and `outputs` directories.

## Before running

Close browsers and other memory-heavy applications. The environment report
previously showed only about 3.5 GiB available RAM while many applications were
open.

The first execution downloads two embedding models and installs PyTorch /
Sentence Transformers. It may take time.

## Run

Open PowerShell:

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\RUN_STAGE3.ps1
```

Or run the commands individually:

```powershell
pip install -r requirements_stage3.txt
python 11_repair_split_pin_pilot_to_dev.py
python 12_run_dense_retrieval_pilot.py
python 13_run_hybrid_rrf_and_route_analysis.py
python 14_validate_stage3.py
```

## Expected runtime

The BGE model should be faster. PubMedBERT is larger and may require
substantially more CPU time. Both corpus embedding files are cached, so an
interrupted later analysis does not require re-encoding completed models.

The scripts print progress and save partial model-specific results after each
model finishes.

## Upload after completion

Upload these eight files:

```text
outputs\stage3\split_manifest_v2.json
outputs\stage3\dense_models_summary.json
outputs\stage3\dense_pilot_by_model_type.csv
outputs\stage3\dense_pilot_results.jsonl
outputs\stage3\hybrid_rrf_summary.json
outputs\stage3\hybrid_rrf_by_type.csv
outputs\stage3\route_utility_analysis.json
outputs\stage3\stage3_validation_report.json
```

Do not upload:

- `.env`
- model files
- `.npy` embedding files
- train/dev/test ID files
- the large candidate corpus

## Scientific limits

Stage 3 evaluates retrieval only. It does not yet evaluate:

- knowledge-graph construction
- agentic routing
- answer generation
- claim verification
- selective abstention
- final test performance

The evidence-risk prior remains provisional. Stage 3 tests whether it actually
separates easy and difficult retrieval cases. It will be revised or removed if
it does not add predictive value beyond question type.
