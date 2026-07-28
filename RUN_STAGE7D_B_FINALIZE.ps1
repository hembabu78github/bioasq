$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python environment not found. Run from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1

python 48_finalize_stage7d_answer_eval.py
if ($LASTEXITCODE -ne 0) {
    throw "Stage 7D-B finalization failed. Complete all batches first."
}

python 49_validate_stage7d_answer_eval.py
if ($LASTEXITCODE -ne 0) {
    throw "Stage 7D-B final validation failed."
}

Write-Host ""
Write-Host "Stage 7D-B finalized and validated."
