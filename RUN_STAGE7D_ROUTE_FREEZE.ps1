$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python environment not found. Run from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1

$env:STAGE7_MAX_RATE_WAIT_SECONDS = "3600"
$env:STAGE7_MAX_RATE_WAIT_EVENTS = "30"

python 43_prepare_stage7d_route_freeze.py
if ($LASTEXITCODE -ne 0) {
    throw "Stage 7D route sample preparation failed."
}

python 44_run_stage7d_route_freeze.py
if ($LASTEXITCODE -ne 0) {
    throw "Stage 7D route freeze failed."
}

python 45_validate_stage7d_route_freeze.py
if ($LASTEXITCODE -ne 0) {
    throw "Stage 7D route validation failed."
}

Write-Host ""
Write-Host "Stage 7D route freeze completed."
Write-Host "Upload the four route-freeze review files before answer generation."
