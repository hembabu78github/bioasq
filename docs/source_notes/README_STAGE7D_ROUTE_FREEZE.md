# Stage 7D-A — Frozen Routing Manifest

This is the immediate next step after the valid Annotator A–C human audit.

Do **not** run the full answer evaluation yet. Stage 7D-A first freezes the
risk-adaptive route for all 24 development questions.

## Frozen policy

- Factoid, summary and yes/no questions use hybrid text evidence.
- Each of the six list questions is evaluated by the deterministic graph
  evidence selector.
- A list question uses graph-selected text only when the frozen
  counterfactual-coverage rule passes.
- The graph is never supplied to the generator or verifier.

## API usage

Three Stage 7B V3 graph results are reused:

- S7Q-007
- S7Q-009
- S7Q-011

At most three new graph calls are required for:

- S7Q-008
- S7Q-010
- S7Q-012

No answer-generation or verifier calls are made.

## Install

Extract the package and copy these files into:

`D:\Prog\JMS_RAG`

- `stage7d_common.py`
- `43_prepare_stage7d_route_freeze.py`
- `44_run_stage7d_route_freeze.py`
- `45_validate_stage7d_route_freeze.py`
- `RUN_STAGE7D_ROUTE_FREEZE.ps1`

The package does not overwrite `stage7_common.py`.

## Run

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\RUN_STAGE7D_ROUTE_FREEZE.ps1
```

The run is checkpointed. Re-run the same command after a quota interruption.

## Stop after routing

Upload the route files for review. Do not start the 72-arm answer evaluation
and do not access the sealed test.
