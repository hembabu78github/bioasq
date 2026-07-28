$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python environment not found. Run from D:\Prog\JMS_RAG."
}

. .\.venv\Scripts\Activate.ps1

$env:STAGE6_ALT_VERIFIER_MODEL = "llama-3.3-70b-versatile"

$backup = "outputs\stage6_evaluation\pre_completeness_repair_v1"
New-Item -ItemType Directory -Force $backup | Out-Null
$files = @(
    "stage6_verifier_comparison_summary.json",
    "stage6_verifier_by_class.csv",
    "stage6_verifier_predictions.jsonl",
    "stage6_verifier_error_cases.csv",
    "stage6_model_discovery.json",
    "stage6_validation_report.json"
)
foreach ($file in $files) {
    $source = Join-Path "outputs\stage6_evaluation" $file
    if (Test-Path $source) {
        Copy-Item $source (Join-Path $backup $file) -Force
    }
}

Write-Host "Stage 6 semantic completeness repair started."
Write-Host "Complete checkpoints will be skipped."
Write-Host "Only checkpoint groups with omitted claim records will be rerun."

python 25_run_stage6_verifier_comparison.py
if ($LASTEXITCODE -ne 0) { throw "Stage 6 completeness repair failed." }

python 26_validate_stage6.py
if ($LASTEXITCODE -ne 0) { throw "Stage 6 completeness validation failed." }

Write-Host ""
Write-Host "Stage 6 completeness repair and validation completed."
