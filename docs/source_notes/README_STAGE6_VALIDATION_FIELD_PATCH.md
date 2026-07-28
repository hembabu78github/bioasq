# Stage 6 Validation Field Patch V1

The verifier comparison completed successfully. Validation failed only because
the Stage 6 candidate quality patch renamed the summary field:

- original schema: `candidate_count`
- quality-patch schema: `corrected_candidate_count`

The previous validator checked only `candidate_count`.

This patch accepts either field. It does not rerun Groq calls or alter the
experimental results.

## Install

Copy these files into:

```text
D:\Prog\JMS_RAG
```

Replace:

```text
26_validate_stage6.py
```

Also copy:

```text
RUN_STAGE6_VALIDATE_ONLY.ps1
```

## Run

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
.\RUN_STAGE6_VALIDATE_ONLY.ps1
```

Do not rerun the full Stage 6 evaluation.
