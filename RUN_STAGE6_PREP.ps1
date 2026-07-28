$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python environment not found. Run from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1

python -m pip install -r requirements_stage6.txt
if ($LASTEXITCODE -ne 0) { throw "Stage 6 dependency installation failed." }

python 22_prepare_stage6_adversarial_candidates.py
if ($LASTEXITCODE -ne 0) { throw "Adversarial candidate generation failed." }

python 23_prepare_stage6_annotation_packets.py
if ($LASTEXITCODE -ne 0) { throw "Annotation packet creation failed." }

Write-Host ""
Write-Host "Stage 6 preparation completed."
Write-Host "Give each annotator only their own CSV and the Stage 6 annotation protocol."
