# Stage 7E — BioASQ Development Gold Scoring

This stage makes the final BGE-versus-hybrid text-route decision.

## Why this is the next step

The two-annotator audit rejected selective graph routing. The remaining main
architecture candidates are:

1. BGE text-only
2. Hybrid text-only

The comparison now uses the existing BioASQ gold answers rather than another
annotation round.

## Evaluation

Primary deployed scoring:

- Factoid: lenient reciprocal rank
- List: lenient item-level F1
- Yes/no: accuracy
- Summary: ROUGE-L F1

An abstained answer receives score zero. Raw answer scores are also reported
as a secondary diagnostic.

The overall composite is the unweighted mean of the four question-type means.
The final route is selected using a frozen paired-bootstrap rule:

- select Hybrid only when the entire 95% CI for Hybrid minus BGE is above 0;
- select BGE when the entire CI is below 0;
- when the CI includes 0, select BGE by parsimony.

## Install

Copy all package files to:

`D:\Prog\JMS_RAG`

## Run

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\RUN_STAGE7E_GOLD_SCORE.ps1
```

The script searches common project locations for `training11b.json`.

When it is stored elsewhere:

```powershell
.\RUN_STAGE7E_GOLD_SCORE.ps1 -BioASQJson "D:\path\to\training11b.json"
```

This stage makes no Groq calls and does not access the sealed test.

## Upload after completion

1. `outputs\stage7e_gold_scoring\stage7e_gold_per_question.jsonl`
2. `outputs\stage7e_gold_scoring\stage7e_gold_summary.json`
3. `outputs\stage7e_gold_scoring\stage7e_text_route_decision.json`
4. `outputs\stage7e_gold_scoring\stage7e_gold_validation_report.json`
