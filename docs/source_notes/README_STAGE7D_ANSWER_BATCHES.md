# Stage 7D-B — Checkpointed Development Answer Evaluation

The Stage 7D-A route manifest has been accepted and frozen by SHA-256 hash.

## Logical comparison

Each of the 24 development questions has three logical conditions:

1. BGE text-only
2. Hybrid text-only
3. Risk-adaptive

This produces 72 logical arms.

When two conditions use the same question and the same ordered five evidence
snippets, they share one deterministic execution job. This preserves all
logical arms while avoiding unnecessary duplicate API calls.

## Frozen evidence rule

- Generator and verifier receive exactly five text snippets.
- No graph nodes, edges, answer aspects or hidden snippets are supplied.
- Verifier citations are restricted to E1-E5.
- The graph-selected risk route is frozen only for S7Q-009, S7Q-010 and S7Q-011.
- The sealed test is not accessed.

## Install

Copy these files into `D:\Prog\JMS_RAG`:

- `stage7d_common.py`
- `46_prepare_stage7d_answer_batches.py`
- `47_run_stage7d_answer_batch.py`
- `48_finalize_stage7d_answer_eval.py`
- `49_validate_stage7d_answer_eval.py`
- `RUN_STAGE7D_B_PREP.ps1`
- `RUN_STAGE7D_B_BATCH.ps1`
- `RUN_STAGE7D_B_FINALIZE.ps1`

## Immediate run

Prepare the batch plan:

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\RUN_STAGE7D_B_PREP.ps1
```

The script prints the number of unique execution jobs and batches.

Then run **Batch 1 only**:

```powershell
.\RUN_STAGE7D_B_BATCH.ps1 -Batch 1
```

Upload the Batch 1 files for technical review before running the remaining
batches.

## Later

After all batches have been approved and completed:

```powershell
.\RUN_STAGE7D_B_FINALIZE.ps1
```
