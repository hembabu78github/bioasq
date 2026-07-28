$ErrorActionPreference = "Stop"

Write-Host "Creating Python 3.11 virtual environment..."
py -3.11 -m venv .venv

Write-Host "Activating environment..."
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing Stage 0 requirements..."
pip install -r requirements_stage0.txt

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host ""
    Write-Host "A .env file has been created."
    Write-Host "Open it and insert the Groq key before running the model probe:"
    Write-Host "notepad .env"
}

Write-Host ""
Write-Host "Running environment check..."
python 00_environment_check.py

Write-Host ""
Write-Host "After adding the Groq key to .env, run:"
Write-Host "python 01_groq_model_probe.py"
