# Stage 4 correction and independent-audit preparation

A detailed review of the Stage 4 raw outputs found a parser/schema issue:

- The verification prompt requested a `verifications` list.
- For two questions, the model returned valid verification items under a
  `claims` list instead.
- The original sanitizer did not accept that alias and replaced the returned
  supported labels with `insufficient_evidence`.

The raw model output is still preserved, so no Groq rerun is required.

The correction package:

1. re-parses the stored raw verifier responses;
2. accepts `verifications` or `claims`;
3. recalculates support and abstention summaries;
4. measures whether graph-assisted generation actually differed from text-only
   generation;
5. creates a blinded manual-audit CSV;
6. keeps the model-predicted labels in a separate private key file.

## Copy into the project

Copy this package's files into:

```text
D:\Prog\JMS_RAG
```

## Run

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
python 19_reaudit_stage4_outputs.py
```

No API calls are made.

## Upload

Upload these four review files:

```text
outputs\stage4_correction\graph_claim_pilot_summary_corrected.json
outputs\stage4_correction\graph_claim_pilot_by_type_corrected.csv
outputs\stage4_correction\graph_claim_pilot_reaudit.json
outputs\stage4_correction\stage4_correction_validation_report.json
```

Keep these locally for the independent annotation step:

```text
outputs\stage4_correction\claim_manual_audit_blinded.csv
outputs\stage4_correction\claim_manual_audit_model_key_private.jsonl
```

Do not open the private model-key file before the human labels are completed.

## Manual audit requirement

The blinded CSV should eventually be labelled independently by two authors or
qualified evaluators using exactly:

- `supported`
- `contradicted`
- `insufficient_evidence`

The claim should be judged only against the evidence and graph shown in the
row—not against outside knowledge.

The two annotation columns must be completed independently before adjudication.
