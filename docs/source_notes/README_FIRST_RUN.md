# Journal of Medical Systems — Auditable Biomedical RAG
## Stage 0: Windows environment and Groq verification

This starter package is designed for:

- Windows PowerShell
- Python 3.11
- 16 GB RAM
- Approximately 50 GB free disk space
- CPU-only execution
- Groq API for generation
- No Google Colab

The first run does **not** download biomedical datasets or large transformer models. It only verifies the computer, creates a reproducibility report, and checks which Groq models are currently available to your account.

## 1. Extract the ZIP

Extract the package to a short Windows path, for example:

```powershell
D:\Prog\JMS_RAG
```

Avoid OneDrive-synced folders and paths containing very long names.

## 2. Open PowerShell in the extracted folder

Example:

```powershell
cd D:\Prog\JMS_RAG
```

## 3. Create and activate the Python 3.11 environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

When activation succeeds, the PowerShell prompt will begin with `(.venv)`.

## 4. Install the Stage 0 packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements_stage0.txt
```

## 5. Create the local `.env` file

```powershell
Copy-Item .env.example .env
notepad .env
```

Replace:

```text
GROQ_API_KEY=PASTE_YOUR_KEY_HERE
```

with your actual Groq key.

Do **not** upload the `.env` file or paste the key into ChatGPT.

## 6. Run the environment check

```powershell
python 00_environment_check.py
```

Expected output files:

```text
outputs\stage0\environment_report.json
outputs\stage0\pip_freeze.txt
```

## 7. Check the Groq connection and available models

```powershell
python 01_groq_model_probe.py
```

Expected output file:

```text
outputs\stage0\groq_models.json
```

## 8. Upload only these three files

Upload:

- `environment_report.json`
- `pip_freeze.txt`
- `groq_models.json`

Do not upload `.env`.

## Troubleshooting

### PowerShell blocks environment activation

Run once in the same PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### `py -3.11` is not found

Confirm installation:

```powershell
py -0p
```

Use the displayed Python 3.11 executable.

### Groq authentication error

Check that `.env` contains exactly one line beginning with:

```text
GROQ_API_KEY=
```

Do not add quotation marks around the key unless the key itself contains spaces.

### Package installation error

Save the full PowerShell error and share it. Do not install alternative package versions independently, because the environment record must remain reproducible.
