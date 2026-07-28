# Stage 1 — Acquire, validate, and profile BioASQ11

This stage downloads the archived BioASQ11 training dataset from the official
Zenodo record, verifies its published MD5 checksum, audits the JSON structure,
creates a deterministic pilot subset, and performs a four-question Groq
JSON-output smoke test.

## Installation location

Copy this package's contents into the existing project folder:

```text
D:\Prog\JMS_RAG
```

Keep the existing `.env`, `.venv`, and `outputs\stage0` folders.

## Run

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1

python 02_download_bioasq11.py
python 03_audit_bioasq11.py
python 04_make_pilot_subset.py
python 05_groq_structured_smoke_test.py
```

Or run:

```powershell
.\RUN_STAGE1.ps1
```

## Verified source used by the scripts

- Dataset: BioASQ-QA, version BioASQ11
- Zenodo record: 7655130
- File: `training11b.json`
- Published MD5: `fc1fe03831b69157c82a746337c00712`

The downloader stops if the checksum is wrong.

## Upload after completion

Upload these six files:

```text
outputs\stage1\download_manifest.json
outputs\stage1\bioasq11_audit.json
outputs\stage1\bioasq11_question_type_counts.csv
outputs\stage1\bioasq11_pilot_manifest.json
outputs\stage1\groq_smoke_test_summary.json
outputs\stage1\groq_smoke_test_results.jsonl
```

Do not upload `.env`, your Groq key, the full dataset, or `.venv`.

No final train/development/test split is created in this stage. We will choose
the split only after reviewing duplicates and the evidence-corpus structure.
