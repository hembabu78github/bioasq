# Stage 6 Output Completeness Repair V1

The uploaded Stage 6 files passed the original structural validator, but a
claim-level output audit found silent omissions:

- Original prompt / same model: 11 predictions had no verifier rationale,
  spread across 5 question groups.
- Hardened prompt / same model: 27 predictions had no rationale and qualifier
  audit, spread across 7 question groups.
- Hardened prompt / alternate model: all 76 predictions were complete.

The earlier script converted any omitted claim record into
`insufficient_evidence`. This artificially improved insufficient-evidence
recall and changed the model ranking. Therefore the reported selection of
`hardened_prompt_same_model` is provisional and must not be frozen.

## What this patch does

- Preserves the earlier six evaluation outputs under:
  `outputs\stage6_evaluation\pre_completeness_repair_v1`
- Validates every existing checkpoint group.
- Skips complete groups.
- Reruns only incomplete groups with a larger response allowance.
- Requires exactly one claim record per expected claim ID.
- Requires a non-empty rationale for every prediction.
- Requires material-qualifier checks for every hardened-prompt prediction.
- Recalculates all metrics and the verifier ranking.
- Adds `stage6_output_completeness_report.json`.
- Strengthens final validation to reject silent omissions.

## Install

Copy these files into `D:\Prog\JMS_RAG` and replace the existing versions:

- `25_run_stage6_verifier_comparison.py`
- `26_validate_stage6.py`
- `RUN_STAGE6_COMPLETENESS_REPAIR.ps1`

The existing rate-limit-aware `stage6_common.py` remains unchanged.

## Run

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\RUN_STAGE6_COMPLETENESS_REPAIR.ps1
```

The alternate model is fixed to `llama-3.3-70b-versatile` so its complete
checkpoints remain reusable.

Do not delete the checkpoint directory. Do not run the sealed test set.
