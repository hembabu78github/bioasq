$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python environment not found. Run from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1

$env:STAGE7_MAX_RATE_WAIT_SECONDS = "3600"
$env:STAGE7_MAX_RATE_WAIT_EVENTS = "30"

python 33_prepare_stage7_graph_scope_v3.py
if ($LASTEXITCODE -ne 0) { throw "Stage 7B V3 sample preparation failed." }

python 34_run_stage7_graph_scope_v3.py
if ($LASTEXITCODE -ne 0) { throw "Stage 7B V3 routing-scope run failed." }

python 35_validate_stage7_graph_scope_v3.py
if ($LASTEXITCODE -ne 0) { throw "Stage 7B V3 routing-scope validation failed." }

Write-Host ""
Write-Host "Stage 7B V3 routing-scope gate completed."
