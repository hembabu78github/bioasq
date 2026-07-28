# Stage 6 Candidate Quality Patch V1

The original Stage 6 preparation passed structural checks, but a claim-by-claim
scientific audit identified several candidates that were absent rather than
directly contradicted, one insufficiency candidate that was actually supported,
one exact duplicate, and one evidence-conflict pair.

This patch keeps 48 claims and the provisional 24/24 design while replacing nine
candidate records and repairing two provenance lists.

## Replace these local files

Copy the contents of this package into:

D:\Prog\JMS_RAG

Replace:

- outputs\stage6_prep\adversarial_claim_candidates_flat_private.jsonl
- outputs\stage6_prep\adversarial_generation_summary.json
- outputs\stage6_annotation\stage6_annotator_A.csv
- outputs\stage6_annotation\stage6_annotator_B.csv
- outputs\stage6_annotation\stage6_annotation_packet_manifest.json

## Important

Do not use the old Annotator A or B packet.
Do not show the private JSONL or provisional labels to either annotator.
Give each annotator only their corrected CSV and the existing Stage 6 protocol.
