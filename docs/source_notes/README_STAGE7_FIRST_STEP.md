# Stage 7A — Sampling and Smoke-Test Package

You have already completed the Stage 6 freeze. You do not need to interpret the
long Stage 7 design notes or manually select questions.

## Your immediate task

1. Copy this package into:

   `D:\Prog\JMS_RAG`

2. Run only:

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\RUN_STAGE7_PREP.ps1
```

3. Upload the three files listed below.

Do not run `RUN_STAGE7_SMOKE.ps1` until the generated sample has been reviewed.

## What the preparation script does

It selects 24 development questions before any Stage 7 generation:

- 6 factoid, 6 list, 6 summary and 6 yes/no;
- 8 low-, 8 medium- and 8 high-uncertainty questions;
- one graph-suitable candidate and one non-graph control from every
  question-type × uncertainty cell;
- no questions from the earlier 12-question graph pilot;
- no sealed-test access.

It also creates a balanced eight-question smoke subset.

## Files to upload after preparation

- `outputs\stage7_sampling\stage7_question_sample.csv`
- `outputs\stage7_sampling\stage7_sampling_summary.json`
- `outputs\stage7_sampling\stage7_smoke_manifest.json`

The package includes the resumable four-condition smoke runner, but it is held
until the sample review is complete.
