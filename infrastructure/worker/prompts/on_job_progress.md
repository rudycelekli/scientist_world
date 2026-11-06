# Job Progress Update

A running job has reported intermediate output.

| Field | Value |
|---|---|
| **Job ID** | `{run_id}` |
| **Running for** | {elapsed_min} minutes |
| **stdout** | `{stdout_path}` |
| **stderr** | `{stderr_path}` |

Read the log files to check on the job's progress.

---

## Decision

Based on the logs, should this job **continue** or be **cancelled**?

Cancel if:
- Training loss is diverging or NaN
- The approach is clearly wrong (e.g., wrong data format, missing files)
- Errors indicate the run will not produce useful results
- You realize you need to change the approach before continuing
- The job appears idle / free — no new output for a long time, no visible progress, stuck in a hang or busy-loop with nothing useful happening

Continue if:
- Training is progressing normally
- The output looks reasonable even if not final
- You are not sure — let it run longer

Respond with exactly one word: `continue` or `cancel`
