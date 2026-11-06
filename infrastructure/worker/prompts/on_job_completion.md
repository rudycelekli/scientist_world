# Job Completed

A compute job has finished.

| Field | Value |
|---|---|
| **Job ID** | `{run_id}` |
| **Exit code** | `{returncode}` |
| **stdout** | `{stdout_path}` |
| **stderr** | `{stderr_path}` |

Read the log files to review the job's output.

---

## What to Do

**You are fully autonomous. There is no human in the loop. Do not ask for confirmation, do not wait for direction, do not say "let me know" or "please confirm". Decide and act immediately — directly edit code, fix bugs, and create files in this turn.**

1. **Review the logs.** Understand what happened — success, failure, or partial results.
2. **Assess progress.** How does this move you toward your reproduction goals?
3. **Edit code in this turn.** Based on the logs, directly edit scripts, fix bugs, update configs, create new files — whatever is needed. Most turns should make a concrete change that advances the reproduction; spend a turn without editing only when you genuinely need to inspect or plan, not to show motion.
4. **Submit the next job** with your updated code, or signal done if finished.
5. **Update your milestone** if progress was made.

End your response with the structured YAML block as described in your instructions.
