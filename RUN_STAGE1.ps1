$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Stage 0 virtual environment not found. Run this from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1

python 02_download_bioasq11.py
if ($LASTEXITCODE -ne 0) { throw "Dataset download failed." }

python 03_audit_bioasq11.py
if ($LASTEXITCODE -ne 0) { throw "Dataset audit failed." }

python 04_make_pilot_subset.py
if ($LASTEXITCODE -ne 0) { throw "Pilot creation failed." }

python 05_groq_structured_smoke_test.py
if ($LASTEXITCODE -ne 0) { throw "Groq smoke test failed." }

Write-Host ""
Write-Host "Stage 1 completed. Upload the six files listed in README_STAGE1.md."
