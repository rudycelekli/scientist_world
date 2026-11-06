## Workspace (verifier edition)

You are working inside the completed run's workspace. Orient yourself
by reading every file in `READMEs/`:

- `problem.md` — the research problem
- `evaluation.md` — metrics, baselines, evaluation protocol
- `TARGETS.md` — what the agent was asked to optimize
- `CONSTRAINTS.md` — what the agent was NOT allowed to change
- `DIRECTION.md` — the allowed scope of methodological approaches
- `EXPERIMENTS.md` — what each experiment tests (maps to keys in
  `scores_train.json`)

The layout: `method/` is the agent's frozen method (the artifact under
verification), `eval/` holds the evaluators, `scoring/` holds score and
result files, `idea/` and `reflection.md` are the agent's own writing,
`jobs/` holds job logs, and git history records everything that
happened.

## Rules (these override anything below that conflicts)

1. `eval/`, `scripts/evaluate_test.sh`, and `data/` must MATCH the
   pristine source copies — the pipeline restored `eval/` and
   `scripts/evaluate_test.sh` from source before your session, and
   keeping evaluation conditions pristine is YOUR job. If you find
   residual tampering (compare against git history), restoring it is a
   fix to make. Only the method's logic is untouchable.
2. Do NOT create or modify `idea/idea_v*.md` or `reflection.md` —
   those are the agent's reasoning trace and part of the audited
   artifact. Your findings go in `scoring/verification.json` and your
   commit messages.
3. You have no score target. Nothing here is a "bar to beat" — you are
   measuring, not competing.
4. Your compute entrypoint is `scripts/run.sh`. Each `action: submit`
   runs `bash /home/user/scripts/run.sh` inside the compute container.
   Edit that script to run whatever the current verification step
   needs (re-running the method, regenerating predictions, invoking
   the evaluators). Note: the `test: true` flag on a submit selects a
   short-walltime smoke-test NODE and has nothing to do with the
   held-out test set — run real evaluations with `test: false`.
5. Use `scripts/download.sh` if you need to re-fetch data or models
   the run's environment expects.
6. **Regularly commit your work — no need to push.** Every fix you
   apply must be a visible commit with a clear message.
