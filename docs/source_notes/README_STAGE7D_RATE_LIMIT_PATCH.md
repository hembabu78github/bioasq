# Stage 7D Groq milliseconds rate-limit patch

## Cause

Groq returned:

`Please try again in 472.5ms`

The previous parser interpreted the `m` in `ms` as **minutes**, producing a
false wait of 28,350 seconds.

## Fix

The patched `stage7d_common.py` explicitly parses milliseconds before minutes
and also supports seconds, compound minute/second durations and hour-based
quota waits.

## Apply

Replace:

`D:\Prog\JMS_RAG\stage7d_common.py`

with the patched file in this package.

Do not delete any Stage 7D checkpoints.

## Resume

Run Batch 2 again:

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\RUN_STAGE7D_B_BATCH.ps1 -Batch 2
```

The five completed Batch 2 jobs should be skipped. Only the interrupted sixth
job should resume.

After Batch 2 succeeds, continue with Batches 3-9.
