# Stage 7F — One-Time Sealed-Test Evaluation

## Frozen architecture

- Route: BGE text-only
- Evidence: top five BGE snippets
- Generator: openai/gpt-oss-20b
- Prompt: stage7d_evidence_closed_v1
- Verifier: frozen hardened same-model verifier
- Release: only when every atomic claim is verifier-supported
- Graph route: excluded

The held-out test contains 706 questions:
- 212 factoid
- 135 list
- 169 summary
- 190 yes/no

## Irreversible point

Running `RUN_STAGE7F_PREP.ps1` opens the sealed test questions. From that
moment, no model, route, prompt, evidence count, verifier rule, threshold or
metric may be changed.

Preparation does not read test gold answers or relevance labels. Gold is read
only during finalization after all answer jobs are complete.

## Install

Copy every file from this package into:

`D:\Prog\JMS_RAG`

## First action only

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\RUN_STAGE7F_PREP.ps1
```

Then upload:

- `outputs\stage7f_test_eval\stage7f_test_manifest.json`
- `outputs\stage7f_test_eval\stage7f_test_batch_plan.json`

Do not start API batches until the preparation manifest is reviewed.

## Later execution

After preparation acceptance:

```powershell
.\RUN_STAGE7F_ALL_BATCHES.ps1
```

The runner is checkpointed. A rate-limit or daily-quota interruption is safe:
rerun the same command later and completed jobs will be skipped.

After all 118 batches:

```powershell
.\RUN_STAGE7F_FINALIZE.ps1
```

No post-test tuning or rerunning is permitted.
