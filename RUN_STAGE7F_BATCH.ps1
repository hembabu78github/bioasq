param(
    [Parameter(Mandatory=$true)]
    [int]$Batch
)
$ErrorActionPreference = "Stop"
cd D:\Prog\JMS_RAG
. .\.venv\Scripts\Activate.ps1
python 53_run_stage7f_test_batch.py --batch-index $Batch
if ($LASTEXITCODE -ne 0) {
    throw "Stage 7F batch $Batch failed."
}
Write-Host "Stage 7F batch $Batch completed."
