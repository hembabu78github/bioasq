$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python environment not found. Run from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1

$env:STAGE7_MAX_RATE_WAIT_SECONDS = "3600"
$env:STAGE7_MAX_RATE_WAIT_EVENTS = "30"

python 36_prepare_stage7c_answer_smoke.py
if ($LASTEXITCODE -ne 0) { throw "Stage 7C sample preparation failed." }

python 37_run_stage7c_answer_smoke.py
if ($LASTEXITCODE -ne 0) { throw "Stage 7C answer smoke failed." }

python 38_validate_stage7c_answer_smoke.py
if ($LASTEXITCODE -ne 0) { throw "Stage 7C validation failed." }

python 39_build_stage7c_blinded_packets.py
if ($LASTEXITCODE -ne 0) { throw "Stage 7C annotation packet build failed." }

Write-Host ""
Write-Host "Stage 7C selective answer smoke completed."
