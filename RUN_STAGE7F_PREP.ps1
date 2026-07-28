$ErrorActionPreference = "Stop"
cd D:\Prog\JMS_RAG
. .\.venv\Scripts\Activate.ps1
python 52_prepare_stage7f_test_eval.py
if ($LASTEXITCODE -ne 0) {
    throw "Stage 7F sealed-test preparation failed."
}
Write-Host ""
Write-Host "Stage 7F preparation completed."
Write-Host "STOP and upload the manifest and batch plan before API execution."
