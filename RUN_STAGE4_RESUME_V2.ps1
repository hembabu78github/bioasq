$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python environment not found. Run from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1

Write-Host "Resuming Stage 4 with JSON-mode fallback support..."
python 17_run_graph_claim_pilot.py
if ($LASTEXITCODE -ne 0) { throw "Graph/claim pilot resume failed." }

Write-Host "Running Stage 4 validation..."
python 18_validate_stage4.py
if ($LASTEXITCODE -ne 0) { throw "Stage 4 validation failed." }

Write-Host ""
Write-Host "Stage 4 resume V2 completed."
