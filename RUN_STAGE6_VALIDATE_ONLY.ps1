$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python environment not found. Run from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1

python 26_validate_stage6.py
if ($LASTEXITCODE -ne 0) { throw "Stage 6 validation failed." }

Write-Host ""
Write-Host "Stage 6 validation completed successfully."
