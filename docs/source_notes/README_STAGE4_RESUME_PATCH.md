# Stage 4 resume patch

The failure was caused by one nearly valid Groq JSON response containing:

```text
"graph_edge_ids:[]
```

instead of:

```text
"graph_edge_ids":[]
```

This is a model JSON-formatting error, not a problem with the dataset, API key,
knowledge-graph logic, or your computer.

This patch:

1. recovers `failed_generation` from Groq's 400 response;
2. repairs minor JSON syntax errors using `json-repair`;
3. resumes from already completed questions instead of overwriting them;
4. flushes each completed record immediately for safer restart;
5. runs Stage 4 validation after completion.

## Apply

Copy these four patch files into:

```text
D:\Prog\JMS_RAG
```

Allow replacement of the existing:

```text
17_run_graph_claim_pilot.py
```

Do not delete the existing partial file:

```text
outputs\stage4\graph_claim_pilot_results.jsonl
```

It should contain the first five completed questions and will be used for resume.

## Run

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\RUN_STAGE4_RESUME.ps1
```

Expected beginning:

```text
Resuming Stage 4: 5 completed question(s) found...
[1/12] SKIP completed ...
...
[6/12] list — List 3 PD-L1 inhibitors...
```

The malformed answer from question 6 will be repaired if Groq returns it again.
