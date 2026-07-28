$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python environment not found. Run this from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1
python 27_prepare_stage7_sample.py
if ($LASTEXITCODE -ne 0) { throw "Stage 7 sampling failed." }

Write-Host ""
Write-Host "Stage 7 sampling completed."
Write-Host "STOP: Do not run the smoke experiment until the sample is reviewed."
