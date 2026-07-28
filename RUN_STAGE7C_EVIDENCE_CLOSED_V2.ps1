$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python environment not found. Run from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1

$env:STAGE7_MAX_RATE_WAIT_SECONDS = "3600"
$env:STAGE7_MAX_RATE_WAIT_EVENTS = "30"

python 40_prepare_stage7c_evidence_closed_v2.py
if ($LASTEXITCODE -ne 0) { throw "Stage 7C V2 preparation failed." }

python 41_run_stage7c_evidence_closed_v2.py
if ($LASTEXITCODE -ne 0) { throw "Stage 7C V2 run failed." }

python 42_validate_stage7c_evidence_closed_v2.py
if ($LASTEXITCODE -ne 0) { throw "Stage 7C V2 validation failed." }

Write-Host ""
Write-Host "Stage 7C evidence-closed V2 completed."
