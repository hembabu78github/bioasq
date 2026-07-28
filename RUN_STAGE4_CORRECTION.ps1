$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python environment not found. Run from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1
python 19_reaudit_stage4_outputs.py
if ($LASTEXITCODE -ne 0) { throw "Stage 4 correction failed." }

Write-Host ""
Write-Host "Stage 4 correction completed. See outputs\stage4_correction."
