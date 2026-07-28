$ErrorActionPreference = "Stop"
cd D:\Prog\JMS_RAG
. .\.venv\Scripts\Activate.ps1

$planPath = "outputs\stage7f_test_eval\stage7f_test_batch_plan.json"
if (-not (Test-Path $planPath)) {
    throw "Run RUN_STAGE7F_PREP.ps1 first."
}
$plan = Get-Content $planPath -Raw | ConvertFrom-Json

1..$plan.batch_count | ForEach-Object {
    .\RUN_STAGE7F_BATCH.ps1 -Batch $_
}
Write-Host ""
Write-Host "All Stage 7F batches completed."
