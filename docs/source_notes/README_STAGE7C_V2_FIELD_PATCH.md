# Stage 7C V2 Field-Name Patch

The run completed the eight arms and then failed only during CSV export.

Cause:
- The sample and result records use `reuse_v1_result`.
- The CSV writer mistakenly requested `reused_v1_result`.

This patch changes the CSV writer to use the existing field name.

## Install

Copy `41_run_stage7c_evidence_closed_v2.py` into:

`D:\Prog\JMS_RAG`

Replace the existing file.

Do not delete:

`outputs\stage7c_evidence_closed_v2\checkpoints`

## Resume

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
.\RUN_STAGE7C_EVIDENCE_CLOSED_V2.ps1
```

The two new answer/verifier checkpoints should be reused, so no additional
Groq calls should normally be needed.
