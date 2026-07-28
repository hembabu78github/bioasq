$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python environment not found. Run this from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1

Write-Host "Installing Stage 3 dependencies..."
python -m pip install -r requirements_stage3.txt
if ($LASTEXITCODE -ne 0) { throw "Stage 3 dependency installation failed." }

python 11_repair_split_pin_pilot_to_dev.py
if ($LASTEXITCODE -ne 0) { throw "Split repair failed." }

python 12_run_dense_retrieval_pilot.py
if ($LASTEXITCODE -ne 0) { throw "Dense retrieval pilot failed." }

python 13_run_hybrid_rrf_and_route_analysis.py
if ($LASTEXITCODE -ne 0) { throw "Hybrid retrieval / route analysis failed." }

python 14_validate_stage3.py
if ($LASTEXITCODE -ne 0) { throw "Stage 3 validation failed." }

Write-Host ""
Write-Host "Stage 3 completed. Upload the eight files listed in README_STAGE3.md."
