# Stage 5 — Independent blinded claim audit

The corrected Stage 4 output is structurally valid, but the model-generated verifier labels are not gold labels. The immediate next step is an independent two-annotator evidence-support audit of the 28 unique claims.

## Prepare the two packets

Copy this kit into `D:\Prog\JMS_RAG`, then run:

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
python 20_prepare_blinded_annotation_packets.py
```

This creates:

```text
outputs\stage5_annotation\claim_audit_annotator_A.csv
outputs\stage5_annotation\claim_audit_annotator_B.csv
```

Give each annotator:

- only their own CSV;
- `Independent_Claim_Verification_Annotation_Protocol_V1.0.docx`.

Do not give either annotator the private model-label key.

## Annotator requirements

Two independent annotators are required. At least one should have biomedical, clinical, life-science, or biomedical-informatics familiarity. Both must be able to judge textual evidence carefully and follow the protocol. The manuscript must describe their background truthfully.

Each annotator fills only:

```text
annotator_label
annotator_rationale
```

Allowed labels:

```text
supported
contradicted
insufficient_evidence
```

## Merge after both are complete

Return the two completed files to the project folder using their original names, then run:

```powershell
python 21_merge_annotation_packets.py
```

It creates:

```text
outputs\stage5_annotation_review\claim_audit_agreement_summary.json
outputs\stage5_annotation_review\claim_audit_merged_for_adjudication.csv
outputs\stage5_annotation_review\claim_audit_disagreements.csv
```

Do not open `claim_manual_audit_model_key_private.jsonl` until all disagreements have been adjudicated and the final human labels are locked.

## Important

No Groq calls are made in this stage. The sealed test partition remains untouched.
