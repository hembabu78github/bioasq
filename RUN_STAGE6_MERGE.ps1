$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python environment not found. Run from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1

python 24_merge_stage6_annotations.py
if ($LASTEXITCODE -ne 0) { throw "Stage 6 annotation merge failed." }

Write-Host ""
Write-Host "Review outputs\stage6_annotation_review\stage6_agreement_summary.json."
Write-Host "If disagreements exist, complete adjudication in stage6_merged_for_adjudication.csv."
