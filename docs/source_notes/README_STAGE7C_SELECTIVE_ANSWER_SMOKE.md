# Stage 7C — Selective GraphRAG Answer Smoke

Stage 7B V3 passed with exactly two eligible questions:

- S7Q-009
- S7Q-011

Both are multi-entity list questions. None of the three summary questions
qualified. The frozen graph scope is therefore narrower than the generic V3
validator sentence:

**GraphRAG is used only for eligible multi-entity list questions.**

All other questions fall back to hybrid retrieval.

## What Stage 7C runs

- six hybrid-baseline answers, one for every V3 question;
- two selective-graph answers, only for S7Q-009 and S7Q-011;
- eight frozen hardened-verifier calls;
- deterministic post-verifier final disposition.

No new graph calls are made.

## Final disposition policy

- Release only when the generator did not abstain and every atomic claim is
  verifier-supported.
- Otherwise return an abstention.

This is intentionally conservative and prevents a contradicted or
insufficient-evidence claim from being released.

## Human review

The automated verifier is not human gold and uses the same model family as the
generator. The package creates two blinded annotator CSVs. Route names,
automated labels and automated final dispositions are hidden.

Annotators should independently use only the displayed evidence and enter:

- `supported`
- `contradicted`
- `insufficient_evidence`

They should also complete the answer-completeness and utility fields.

## Install

Copy into:

`D:\Prog\JMS_RAG`

- `36_prepare_stage7c_answer_smoke.py`
- `37_run_stage7c_answer_smoke.py`
- `38_validate_stage7c_answer_smoke.py`
- `39_build_stage7c_blinded_packets.py`
- `RUN_STAGE7C_ANSWER_SMOKE.ps1`

Keep the patched `stage7_common.py`.

## Run

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\RUN_STAGE7C_ANSWER_SMOKE.ps1
```

The run is checkpointed. Re-run the same command after any quota interruption.

## Stop after completion

Upload the core result files for review before annotators begin. Do not run the
full 24-question experiment or the sealed test.
