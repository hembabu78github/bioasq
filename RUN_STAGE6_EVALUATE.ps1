$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python environment not found. Run from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1

$merged = "outputs\stage6_annotation_review\stage6_merged_for_adjudication.csv"
if (-not (Test-Path $merged)) {
    throw "Merged human-reference file is missing. Run RUN_STAGE6_MERGE.ps1 first."
}

$unresolved = Import-Csv $merged | Where-Object {
    [string]::IsNullOrWhiteSpace($_.adjudicated_label)
}
if ($unresolved.Count -gt 0) {
    throw "There are unresolved human labels. Complete adjudication before verifier evaluation."
}

Write-Host "Checkpointing is enabled. Completed groups will be skipped on rerun."
Write-Host "Groq retry windows up to 30 minutes will be waited automatically."
python 25_run_stage6_verifier_comparison.py
if ($LASTEXITCODE -ne 0) { throw "Verifier comparison failed." }

python 26_validate_stage6.py
if ($LASTEXITCODE -ne 0) { throw "Stage 6 validation failed." }

Write-Host ""
Write-Host "Stage 6 evaluation completed."
