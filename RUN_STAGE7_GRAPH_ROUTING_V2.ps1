$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python environment not found. Run from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1

$env:STAGE7_MAX_RATE_WAIT_SECONDS = "3600"
$env:STAGE7_MAX_RATE_WAIT_EVENTS = "30"

python 30_prepare_stage7_graph_routing_v2.py
if ($LASTEXITCODE -ne 0) { throw "Stage 7B sample preparation failed." }

python 31_run_stage7_graph_routing_v2.py
if ($LASTEXITCODE -ne 0) { throw "Stage 7B graph-routing run failed." }

python 32_validate_stage7_graph_routing_v2.py
if ($LASTEXITCODE -ne 0) { throw "Stage 7B graph-routing validation failed." }

Write-Host ""
Write-Host "Stage 7B graph-routing technical gate completed."
