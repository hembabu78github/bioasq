param(
    [string]$BioASQJson = ""
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python environment not found. Run from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1

if ([string]::IsNullOrWhiteSpace($BioASQJson)) {
    python 50_score_stage7e_bioasq_gold.py
} else {
    python 50_score_stage7e_bioasq_gold.py --bioasq-json "$BioASQJson"
}

if ($LASTEXITCODE -ne 0) {
    throw "Stage 7E BioASQ gold scoring failed."
}

python 51_validate_stage7e_gold_scores.py
if ($LASTEXITCODE -ne 0) {
    throw "Stage 7E validation failed."
}

Write-Host ""
Write-Host "Stage 7E BioASQ gold scoring completed and validated."
Write-Host "Upload the four review files. Do not access the sealed test."
