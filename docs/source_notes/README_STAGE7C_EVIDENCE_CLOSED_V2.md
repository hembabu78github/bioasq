# Stage 7C Evidence-Closed V2

Stage 7C V1 passed its structural validator, but scientific review found a
provenance-closure defect.

The graph was built from 20 candidate snippets while only five selected snippets
were displayed to the generator and verifier. Graph edges referenced hidden
candidate snippets, and the same-model verifier accepted graph assertions as
support.

V1 is retained as an auditable failed pilot. Its graph-versus-hybrid comparison
must not be reported as evidence of answer-quality improvement.

## V2 correction

- The graph is used only to select the five evidence snippets.
- No graph entities, edges, answer aspects or hidden candidate snippets are sent
  to the generator.
- The verifier receives the same five displayed snippets and no graph.
- Verifier citations must use only E1, E2, E3, E4 or E5.
- `graph_edge_ids` must be empty.
- The six hybrid baselines are reused because they already used text evidence
  without a graph.
- Only the two graph-selected answers and their verifications are regenerated.

This is a protocol-integrity correction, not answer-level tuning.

## Install

Copy into `D:\Prog\JMS_RAG`:

- `40_prepare_stage7c_evidence_closed_v2.py`
- `41_run_stage7c_evidence_closed_v2.py`
- `42_validate_stage7c_evidence_closed_v2.py`
- `RUN_STAGE7C_EVIDENCE_CLOSED_V2.ps1`

Keep the patched `stage7_common.py`.

## Run

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\RUN_STAGE7C_EVIDENCE_CLOSED_V2.ps1
```

Expected new calls: two generation calls and two verifier calls.

Do not use the old Stage 7C blinded annotation packets. New packets will be
created only after the evidence-closed outputs are reviewed.
