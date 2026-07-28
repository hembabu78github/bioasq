$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python environment not found. Run this from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1

python 06_build_closed_snippet_corpus.py
if ($LASTEXITCODE -ne 0) { throw "Closed-corpus construction failed." }

python 07_create_grouped_split.py
if ($LASTEXITCODE -ne 0) { throw "Grouped split failed." }

python 08_create_evidence_risk_prior.py
if ($LASTEXITCODE -ne 0) { throw "Evidence-risk prior generation failed." }

python 09_run_bm25_pilot.py
if ($LASTEXITCODE -ne 0) { throw "BM25 pilot failed." }

python 10_validate_stage2.py
if ($LASTEXITCODE -ne 0) { throw "Stage 2 validation failed." }

Write-Host ""
Write-Host "Stage 2 completed. Upload the seven files listed in README_STAGE2.md."
