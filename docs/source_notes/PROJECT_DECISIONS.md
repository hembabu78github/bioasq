# Project decisions — Version 0.1

## Target

- Target journal: *Journal of Medical Systems*
- Required indexing: SCIE
- Publishing route: subscription / no mandatory APC

## Compute and budget constraints

- Operating environment: Windows PowerShell
- Python: 3.11
- RAM: 16 GB
- Free disk: approximately 50 GB
- GPU: none assumed
- Google Colab: not permitted
- Generation: Groq API
- Study must be designed to remain feasible on the local computer

## Scientific-integrity rules

1. No numerical value from the rejected manuscript will be reused unless independently reproduced.
2. No claim of MIMIC-III use, clinician evaluation, CITI training, A100 hardware, hospital deployment, or patient-level adaptation will be carried forward without new evidence.
3. All experimental tables and figures must be generated from saved machine-readable outputs.
4. Every run will record model name, prompt/configuration, dataset version, timestamp, software environment, and random seed where supported.
5. The new manuscript will distinguish:
   - retrieval performance;
   - answer performance;
   - evidence support;
   - abstention/selective-answering behavior;
   - latency and resource use.
6. The old manuscript is a conceptual source only, not a source of validated results.

## Planned scientific direction

Working concept:

**An auditable evidence-grounded retrieval-augmented generation system for biomedical question answering**

The system will be designed around:

- lexical and biomedical dense retrieval;
- optional reranking;
- evidence-linked answer generation through Groq;
- automated evidence-support checking;
- a selective-answering or abstention rule when support is insufficient;
- a structured audit record for each question.

Full ANCE training, FiD fine-tuning, and original REALM pretraining are outside the current hardware constraints and will not be claimed.
