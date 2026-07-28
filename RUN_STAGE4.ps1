$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python environment not found. Run from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1

python 15_analyze_dev_retrieval_diagnostics.py
if ($LASTEXITCODE -ne 0) { throw "Development retrieval diagnostics failed." }

python 16_select_graph_claim_pilot.py
if ($LASTEXITCODE -ne 0) { throw "Graph/claim pilot selection failed." }

python 17_run_graph_claim_pilot.py
if ($LASTEXITCODE -ne 0) { throw "Graph/claim pilot failed." }

python 18_validate_stage4.py
if ($LASTEXITCODE -ne 0) { throw "Stage 4 validation failed." }

Write-Host ""
Write-Host "Stage 4 completed. Upload the eight files listed in README_STAGE4.md."
