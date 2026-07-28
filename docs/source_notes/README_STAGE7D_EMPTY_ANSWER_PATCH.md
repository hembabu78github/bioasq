# Stage 7D empty-answer semantic retry patch

## Cause

The generator returned syntactically valid JSON but the `answer` field was
empty. The runner treated every empty answer as fatal, even when the model had
set `abstain=true`.

## Fix

The patched runner:

1. accepts a valid abstention with an empty answer field and records the frozen
   standardized abstention text;
2. retries up to three times when a non-abstaining output has an empty answer
   or no atomic claims;
3. keeps the same model, evidence, temperature and frozen scientific prompt;
4. checkpoints only a semantically valid answer.

## Apply

Replace:

`D:\Prog\JMS_RAG\47_run_stage7d_answer_batch.py`

with the patched file.

Do not delete any checkpoint folders.

## Resume Batch 3

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\RUN_STAGE7D_B_BATCH.ps1 -Batch 3
```

Completed jobs should be skipped. The interrupted `S7Q-009` BGE job should
resume.

After Batch 3 succeeds:

```powershell
4..9 | ForEach-Object {
    .\RUN_STAGE7D_B_BATCH.ps1 -Batch $_
}
```
