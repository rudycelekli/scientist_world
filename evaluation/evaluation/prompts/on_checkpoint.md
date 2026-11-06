## Job Progress Check-in

A job you submitted has been running long enough to hit a periodic
check-in. The job is **paused** right now — waiting for your decision.

| Field | Value |
|---|---|
| **Run ID** | `{run_id}` |
| **Running for** | {elapsed_min} minutes |
| **stdout** | `{stdout_path}` |
| **stderr** | `{stderr_path}` |

Read both log files. Skim toward the tail for the most recent state.

---

## Decide: continue or cancel?

**Cancel the job** if any of these are true:

- Training loss is diverging, stuck, or producing NaN
- The approach is clearly wrong (wrong data format, missing file,
  import error, wrong hyperparameters that can't recover)
- Errors indicate the run will not produce useful results
- You realize on review that you need a different approach
- The job appears idle — no new output for a long time, no visible
  progress, stuck in a hang or a busy-loop with nothing useful
  happening

**Continue the job** if:

- Training is progressing normally (loss decreasing, epochs advancing)
- The output looks reasonable even if not yet done
- You are genuinely unsure — let it run and you'll get another
  check-in later

Cancelling saves GPU-hours that would otherwise be wasted. Continuing
costs you nothing at this point — the compute for the time already
spent is sunk.

---

## Respond

Write `action.yaml` with exactly one of:

```yaml
action: continue
notes: "why you're letting it keep running"
```

```yaml
action: cancel
notes: "what went wrong"
```

Then stop. Do not start other work during a check-in — the paused
job is holding its GPU reservation. If you cancel, you get control
back immediately and can plan the next step on your next turn.
