$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python environment not found. Run from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1

python 46_prepare_stage7d_answer_batches.py
if ($LASTEXITCODE -ne 0) {
    throw "Stage 7D-B batch preparation failed."
}

Write-Host ""
Write-Host "Stage 7D-B preparation completed."
Write-Host "Read the printed batch count, then run Batch 1 only."
