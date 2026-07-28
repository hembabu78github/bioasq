# Stage 7B JSON Fallback Patch V1

The Stage 7B run stopped because Groq structured JSON mode returned:

`json_validate_failed`

and the `failed_generation` field was empty. This is an output-format failure,
not a data, checkpoint or routing-design failure.

## Changes

1. `stage7_common.py`
   - retains structured JSON mode;
   - retries rate limits as before;
   - after repeated structured-validation failures, switches to a plain-text
     JSON-only fallback;
   - parses and repairs the fallback response;
   - records the response mode in the checkpoint.

2. `31_run_stage7_graph_routing_v2.py`
   - limits graph output to at most 12 entities, 16 relations and 10 answer aspects;
   - omits low-value background graph content;
   - performs a semantic retry when a graph says it is sufficient but contains
     neither evidence-backed relations nor answer aspects.

## Install

Copy these two files into:

`D:\Prog\JMS_RAG`

Replace:

- `stage7_common.py`
- `31_run_stage7_graph_routing_v2.py`

Do not delete:

`outputs\stage7_graph_routing_v2\checkpoints`

The two existing completed graph checkpoints will be reused. The failed S7Q-009
call did not create a valid checkpoint.

## Resume

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
$env:STAGE7_MAX_RATE_WAIT_SECONDS = "3600"
$env:STAGE7_MAX_RATE_WAIT_EVENTS = "30"
.\RUN_STAGE7_GRAPH_ROUTING_V2.ps1
```
