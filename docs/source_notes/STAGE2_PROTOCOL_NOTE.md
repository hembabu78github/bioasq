# Stage 2 protocol note

## Current benchmark definition

The project will initially use a closed retrieval corpus constructed from all
deduplicated BioASQ11 gold snippets. The question-specific relevance mapping is
held separately and used only after ranking.

This design is useful for controlled retrieval experiments but has an important
limitation: it is not equivalent to searching the complete PubMed collection.
The limitation will be disclosed in the manuscript, and later stages will
assess whether a document-level or external validation corpus is feasible.

## Evidence-risk terminology

The Stage 2 rule estimates **evidence-retrieval risk**: the likelihood that a
minimal retrieval route will be insufficient based only on the query surface.

It is not a clinical severity, diagnosis, treatment, or patient-harm score.

## Test-set discipline

The grouped test partition is sealed at Stage 2. No route threshold, model,
prompt, retrieval weight, verification rule, or abstention threshold may be
selected using the test set.
