param(
    [Parameter(Mandatory = $true)]
    [int]$Batch
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python environment not found. Run from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1

$env:STAGE7_MAX_RATE_WAIT_SECONDS = "3600"
$env:STAGE7_MAX_RATE_WAIT_EVENTS = "30"

python 47_run_stage7d_answer_batch.py --batch-index $Batch
if ($LASTEXITCODE -ne 0) {
    throw "Stage 7D-B batch $Batch failed."
}

Write-Host ""
Write-Host "Stage 7D-B batch $Batch completed."
