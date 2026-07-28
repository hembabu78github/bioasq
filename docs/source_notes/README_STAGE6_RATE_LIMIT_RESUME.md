# Stage 6 Rate-Limit and Resume Patch V1

The earlier evaluation stopped because Groq returned a model-specific
tokens-per-day 429 response. The scientific inputs and human labels are valid;
the failure is operational.

This patch changes only execution reliability:

1. Saves every completed question/evidence group immediately under:

   outputs\stage6_evaluation\checkpoints\

2. Skips completed groups after any interruption.

3. Parses Groq messages such as `try again in 9m25.92s`, waits for the stated
   interval plus five seconds, and resumes automatically.

4. Uses a smaller dynamic completion-token allowance based on the number of
   claims in the group. This lowers requested tokens without changing the
   evaluation set, prompts, models, labels or selection rule.

## Install

Copy these three files into:

D:\Prog\JMS_RAG

Replace the existing versions:

- stage6_common.py
- 25_run_stage6_verifier_comparison.py
- RUN_STAGE6_EVALUATE.ps1

## Run

The previous failed execution created no usable group checkpoints, so the first
patched run starts from group 1. From that point onward, every successful group
is saved.

Run:

```powershell
cd D:\Prog\JMS_RAG
.\.venv\Scripts\Activate.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\RUN_STAGE6_EVALUATE.ps1
```

You may start immediately. If the prior Groq usage is still inside the rolling
limit window, the script will display a wait message and continue by itself.

Do not close PowerShell while it is waiting. If Windows, the internet connection
or the process interrupts later, run the same command again; checkpointed groups
will show `SKIP`.

## Optional controls

The defaults allow a maximum individual Groq wait of 30 minutes and up to 20
rate-limit wait events. They can be changed in `.env`:

```text
STAGE6_MAX_RATE_WAIT_SECONDS=1800
STAGE6_MAX_RATE_WAIT_EVENTS=20
```

Do not delete the `checkpoints` folder until Stage 6 validation passes.
