$ErrorActionPreference = "Stop"
cd D:\Prog\JMS_RAG
. .\.venv\Scripts\Activate.ps1

python 54_finalize_stage7f_test_eval.py
if ($LASTEXITCODE -ne 0) {
    throw "Stage 7F finalization failed."
}

python 55_validate_stage7f_test_eval.py
if ($LASTEXITCODE -ne 0) {
    throw "Stage 7F validation failed."
}

Write-Host ""
Write-Host "Stage 7F sealed-test evaluation completed and validated."
Write-Host "Do not retune or rerun based on the test results."
