$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python environment not found. Run this from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1

python 28_run_stage7_experiment.py --mode smoke
if ($LASTEXITCODE -ne 0) { throw "Stage 7 smoke experiment failed." }

python 29_validate_stage7_smoke.py
if ($LASTEXITCODE -ne 0) { throw "Stage 7 smoke validation failed." }

Write-Host ""
Write-Host "Stage 7 smoke experiment completed."
