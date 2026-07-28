# Stage 6 — Verifier Hardening and Adversarial Evaluation

Stage 5 established a locked 28-claim human reference set:

- 24 supported claims;
- 4 insufficient-evidence claims;
- no contradicted claims;
- 100% human-human agreement;
- the original verifier produced three false-support errors;
- all three errors involved an unsupported temporal/market-status inferential bridge.

Stage 6 has two phases.

## Phase A — Create and human-label an adversarial development set

The preparation workflow:

1. Reads the locked Stage 5 human-reference claims.
2. Selects only claims labelled `supported`.
3. Uses Groq to create two controlled variants per supported claim:
   - one contradiction candidate;
   - one insufficient-evidence candidate.
4. Preserves the source question, evidence, graph and perturbation provenance.
5. Creates two independently randomized blinded annotation packets.
6. Withholds the generated target label from annotators.

Expected candidate count:

```text
24 supported source claims × 2 variants = 48 adversarial claims
```

The generated target labels are provisional. Human labels are the reference
standard.

## Phase B — Compare three verifier configurations

After both annotators finish and the human labels are locked, the evaluation
workflow compares:

1. `original_prompt_same_model`
   - original permissive verifier prompt;
   - `openai/gpt-oss-20b`.

2. `hardened_prompt_same_model`
   - strict material-qualifier verification prompt;
   - `openai/gpt-oss-20b`.

3. `hardened_prompt_alternate_model`
   - same hardened prompt;
   - a different Groq model discovered at runtime.

The evaluation set combines:

- the original 28 human-audited claims;
- the new 48 human-audited adversarial claims.

Target combined size:

```text
76 development claims
```

The sealed test partition is not accessed.

## Files expected in the existing project

The scripts expect:

```text
outputs\stage4_correction\claim_manual_audit_blinded.csv
outputs\stage5_annotation_review\claim_audit_merged_for_adjudication.csv
outputs\stage4_correction\claim_manual_audit_model_key_private.jsonl
.env
```

The model-key file is used only after Stage 5 human labels were locked.

## Installation

Copy the contents of this ZIP into:

```text
D:\Prog\JMS_RAG
```

Keep all previous stages and the existing `.venv`.

Install/confirm the small dependencies:

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements_stage6.txt
```

## Phase A — Generate adversarial candidates and packets

Run:

```powershell
.\RUN_STAGE6_PREP.ps1
```

This may make up to 24 Groq calls, excluding retries.

Expected outputs:

```text
outputs\stage6_prep\adversarial_claim_candidates_private.jsonl
outputs\stage6_prep\adversarial_generation_summary.json
outputs\stage6_annotation\stage6_annotator_A.csv
outputs\stage6_annotation\stage6_annotator_B.csv
outputs\stage6_annotation\stage6_annotation_packet_manifest.json
```

Give each annotator only:

- their own CSV;
- `Stage6_Adversarial_Claim_Annotation_Protocol_V1.0.docx`.

Do not give either annotator:

- the other annotator's CSV;
- `adversarial_claim_candidates_private.jsonl`;
- the model-prediction key;
- the generation summary containing provisional target labels.

## Human annotation

Both annotators must use exactly:

```text
supported
contradicted
insufficient_evidence
```

They must judge only the displayed evidence and evidence-derived graph.

After both CSVs are complete, place them back at:

```text
outputs\stage6_annotation\stage6_annotator_A.csv
outputs\stage6_annotation\stage6_annotator_B.csv
```

Then run:

```powershell
.\RUN_STAGE6_MERGE.ps1
```

Expected review outputs:

```text
outputs\stage6_annotation_review\stage6_agreement_summary.json
outputs\stage6_annotation_review\stage6_merged_for_adjudication.csv
outputs\stage6_annotation_review\stage6_disagreements.csv
```

### When disagreements exist

Open only:

```text
stage6_disagreements.csv
```

Discuss and adjudicate each disagreement. Enter the final label and note in the
corresponding rows of:

```text
stage6_merged_for_adjudication.csv
```

Use the columns:

```text
adjudicated_label
adjudication_note
```

Do not run verifier comparison until every row has a final human label.

## Phase B — Run verifier comparison

After all rows in `stage6_merged_for_adjudication.csv` have a final
`adjudicated_label`, run:

```powershell
.\RUN_STAGE6_EVALUATE.ps1
```

The script discovers available Groq models. To force a particular alternate
model, add this to `.env`:

```text
STAGE6_ALT_VERIFIER_MODEL=<exact Groq model ID>
```

The alternate model must differ from `openai/gpt-oss-20b`.

Expected maximum:

```text
12 question/evidence groups × 3 verifier configurations = 36 Groq calls
```

The script batches claims sharing the same question and evidence, rather than
making one API call per claim.

## Final Stage 6 outputs

Upload:

```text
outputs\stage6_evaluation\stage6_verifier_comparison_summary.json
outputs\stage6_evaluation\stage6_verifier_by_class.csv
outputs\stage6_evaluation\stage6_verifier_predictions.jsonl
outputs\stage6_evaluation\stage6_verifier_error_cases.csv
outputs\stage6_evaluation\stage6_model_discovery.json
outputs\stage6_evaluation\stage6_validation_report.json
```

Also upload:

```text
outputs\stage6_annotation_review\stage6_agreement_summary.json
```

## Primary model-selection rule

The selected verifier is determined on development data using this ordered
rule:

1. highest `insufficient_evidence` recall;
2. lowest false-support rate;
3. highest `contradicted` recall;
4. highest macro F1 across observed classes;
5. lower mean latency as the final tie-breaker.

Overall accuracy is secondary because supported claims can dominate the label
distribution.

## Scientific limits

Stage 6 is development-only.

It does not establish final test performance, clinical validity or deployment
safety. The selected verifier and thresholds must be frozen before the one-time
sealed test evaluation.
