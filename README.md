# Auditable Biomedical Question Answering with Claim-Level Verification and Selective Abstention

This repository contains the source code, frozen protocols, split identifiers, and aggregate reproducibility artifacts for the study:

**Auditable Biomedical Question Answering with Claim-Level Verification and Selective Abstention: A Frozen BioASQ Evaluation**

The final system used BGE dense retrieval, five displayed evidence snippets, `openai/gpt-oss-20b` through the Groq API, atomic-claim generation, a separate evidence-constrained verification call, and an all-claims-supported release rule. Graph routing was evaluated during development and excluded after it failed the prespecified human-validated gate.

## Main frozen results

- BioASQ-QA source questions: 4,719
- Repaired split: 3,304 train, 709 development, 706 sealed test
- Sealed-test retrieval: Hit@5 = 0.9306; MRR@20 = 0.8576
- Released answers: 602/706 (85.27% coverage)
- Deployed macro composite: 0.5008
- Stratified post-test audit: all 92 verifier-supported sampled claims were human-supported; the verifier was conservative on abstention decisions

Aggregate machine-readable results are under `outputs/`. Detailed per-question outputs, evidence text, checkpoints, raw BioASQ data, and embedding arrays are intentionally not included in this public package.

## Repository layout

- `00_...py` to `55_...py`: staged preparation, retrieval, verification, evaluation, and validation scripts
- `stage*_common.py`: shared utilities
- `RUN_*.ps1`: Windows PowerShell runners
- `requirements.txt`: consolidated dependencies
- `requirements_stage*.txt`: stage-specific dependencies retained from the original workflow
- `data/processed/stage3/`: leakage-controlled split identifiers only
- `protocols/`: frozen evaluation and annotation protocols
- `outputs/`: selected aggregate summaries, manifests, decisions, and validation reports
- `docs/source_notes/`: chronological development notes

## Environment

The workflow was developed for Windows PowerShell, Python 3.11, CPU-only execution, and approximately 16 GB RAM. Large local embeddings and raw datasets are generated or downloaded at runtime and are not stored in Git.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Set the key only in the local `.env` file:

```text
GROQ_API_KEY=your_key_here
```

Never commit `.env`.

## Dataset

The source dataset is BioASQ-QA / BioASQ11, available from Zenodo:

- DOI: `10.5281/zenodo.7655130`
- Expected file: `training11b.json`

Use `02_download_bioasq11.py` or download the dataset directly from the authoritative record. The third-party dataset is not redistributed in this repository. Users remain responsible for complying with the dataset terms.

## Reproduction sequence

The project is staged. Read the source notes before rerunning a stage. The final sealed-test entry points are:

```powershell
.\RUN_STAGE7F_PREP.ps1
.\RUN_STAGE7F_ALL_BATCHES.ps1
.\RUN_STAGE7F_FINALIZE.ps1
```

The final route, model, prompt, evidence count, verifier, release rule, random seeds, and test identifiers were frozen before finalization. Reproduction may be affected by hosted-model availability and provider-side changes.

## Public-package boundaries

Included:

- source and validation code;
- PowerShell runners;
- prompt-bearing scripts;
- split identifiers;
- frozen protocol and decision files;
- aggregate evaluation summaries and manifests.

Excluded:

- API keys and `.env`;
- virtual environments and caches;
- raw BioASQ files and evidence text;
- embedding arrays and model caches;
- per-question answers, verifier rationales, checkpoints, and batch logs;
- private annotation keys and internal project-management files.

## Intended use

This is a research benchmark implementation. It is not a clinical decision-support system and must not be used for patient-specific diagnosis, treatment, or safety-critical deployment.

## Licence

Code in this repository is released under the MIT License. Third-party datasets, models, and provider services retain their own terms and licences.

## Contact

- Hemachandran Babu — `hb6034@srmist.edu.in`
- Priyadarsini K (corresponding author) — `priyadak@srmist.edu.in`
