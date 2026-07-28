# Stage 7B — Graph-Routing V2 Technical Gate

The original Stage 7 smoke run completed all 32 answers, but its graph gate
failed. Review found that this was not simply a threshold issue:

- graph evidence changed for only 1 of 4 graph-role questions;
- it changed for all 4 controls;
- two graphs were declared insufficient;
- the risk router still selected graph for graph-insufficient cases;
- the model supplied the final snippet ranking directly;
- no verifier-driven abstention was applied.

Stage 7B therefore tests graph routing alone before spending tokens on another
four-condition answer run.

## What changes

- Uses a 20-snippet hybrid candidate pool.
- The model extracts entities, evidence-backed relations and answer aspects.
- A deterministic coverage selector chooses the five evidence snippets.
- Graph routing requires:
  - a structurally graph-demanding question;
  - graph_sufficient=true;
  - at least one relevant relation;
  - at least one relevant answer aspect;
  - at least one novel snippet outside hybrid top five;
  - an actual evidence-set change.
- Direct controls can never be routed to graph.
- No answer-generation or verification calls are made.

## Install

Copy these files into:

`D:\Prog\JMS_RAG`

- `30_prepare_stage7_graph_routing_v2.py`
- `31_run_stage7_graph_routing_v2.py`
- `32_validate_stage7_graph_routing_v2.py`
- `RUN_STAGE7_GRAPH_ROUTING_V2.ps1`

Keep the existing `stage7_common.py`.

## Run

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\RUN_STAGE7_GRAPH_ROUTING_V2.ps1
```

This makes at most eight resumable Groq graph calls.

## Upload after the run

- `outputs\stage7_graph_routing_v2\stage7_graph_routing_v2_results.jsonl`
- `outputs\stage7_graph_routing_v2\stage7_graph_routing_v2_results.csv`
- `outputs\stage7_graph_routing_v2\stage7_graph_routing_v2_summary.json`
- `outputs\stage7_graph_routing_v2\stage7_graph_routing_v2_validation_report.json`

Do not run the 24-question experiment or another answer smoke yet.
