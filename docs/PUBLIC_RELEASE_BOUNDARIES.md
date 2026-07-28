# Public release boundaries

This repository is intentionally conservative about redistribution.

## Included

- Executable source code and stage runners
- Prompt-bearing scripts
- Leakage-controlled split identifiers
- Frozen protocol and architecture-decision records
- Aggregate results, manifests, and validation reports

## Excluded from the public package

- `.env`, API keys, credentials, and local virtual environments
- Raw BioASQ data and normalized evidence snippets
- Dense embedding arrays and downloaded model caches
- Per-question generated answers and verifier rationales
- Job checkpoints, rate-limit logs, and batch-level records
- Private annotation keys and completed identifiable annotation workbooks
- Internal submission trackers and project-management documents

The exclusions reduce credential risk, repository size, and uncertainty about redistributing third-party benchmark text. The authoritative BioASQ record should be used to obtain the dataset.
