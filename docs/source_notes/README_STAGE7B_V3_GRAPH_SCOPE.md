# Stage 7B V3 — Final Graph-Routing Scope Gate

The V2 gate cannot be accepted as evidence of graph value.

The only V2-eligible question, S7Q-005, already had all relevant graph items
covered by hybrid top five. Four novel snippets were selected with zero
marginal graph coverage because of the novelty bonus.

In contrast, S7Q-009 contained genuine graph-exclusive evidence but was blocked
only by the model's inconsistent `graph_sufficient=false` flag.

V3 makes one final development-only routing test.

## Scope

The graph route is limited to:

- multi-entity list questions;
- multi-aspect summary questions.

Factoid and yes/no direct-relation questions are outside the graph route.

## Counterfactual eligibility

A route qualifies only when:

- at least two relevant answer aspects are extracted;
- at least one relevant graph item is absent from hybrid top five but covered
  by graph-selected evidence;
- at least one novel selected snippet has positive marginal graph coverage;
- the evidence set changes;
- no zero-coverage novel filler is selected.

The model's `graph_sufficient` flag is recorded but is advisory rather than a
hard gate because V2 showed that it was internally inconsistent.

## Final decision rule

- At least 2 of 6 qualify: retain GraphRAG as a selective list/summary route.
- Fewer than 2 qualify: remove GraphRAG as a central architectural claim and
  continue with hybrid retrieval, the frozen verifier and selective abstention.

## Install

Copy into `D:\Prog\JMS_RAG`:

- `33_prepare_stage7_graph_scope_v3.py`
- `34_run_stage7_graph_scope_v3.py`
- `35_validate_stage7_graph_scope_v3.py`
- `RUN_STAGE7_GRAPH_SCOPE_V3.ps1`

Keep the patched `stage7_common.py`.

## Run

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\RUN_STAGE7_GRAPH_SCOPE_V3.ps1
```

Two existing V2 graphs are reused. At most four new graph calls are made.
No answers or verifier calls are made.
