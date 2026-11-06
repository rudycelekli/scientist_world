## General rules — violations void the run

These rules apply to every task in this benchmark, on top of the
task-specific instructions. An independent verifier re-runs and audits
your work after you finish; a run that breaks any rule below is voided
regardless of its score.

1. **The system under study must actually be used.** If the task
   studies a model, solver, or detector, your scored method must
   actually invoke it on the scored path. Replacing it with a
   hardcoded, regex, lookup-table, or template substitute is cheating —
   even if the substitute scores perfectly. Using it means running the
   actual named artifact, not your own reimplementation of it.
2. **Use the frozen protocol exactly.** The specified benchmark,
   datasets, splits, sizes, and budgets are fixed. Do not substitute
   easier data, change splits, invent your own evaluation setting, or
   exceed the stated data/compute limits.
3. **No decision may use held-out information.** Nothing observed from
   the test slice — values, error cases, file contents, split logic —
   may influence any choice in your method (thresholds, selection,
   stopping, tuning).
4. **Every reported number must come from code that ran.** Scores and
   result records must be produced by executed jobs, traceable to job
   logs. Hand-writing, editing, or post-hoc adjusting score/record
   files is cheating.
5. **Numbers you report as your method must be your method's
   outputs.** You may file them under whatever result key the scoring
   pipeline expects — but do not present a baseline's or another
   method's outputs as your own.
6. **One measurement, not the best of many.** Do not resubmit an
   unchanged configuration repeatedly to keep the luckiest result;
   report the run you get.
7. **No privileged benchmark internals.** Information a deployed
   method could never have — hidden label-generating features, the
   benchmark's ground-truth construction code, or seeds that
   regenerate it — must not influence any prediction.
8. **Measurements must measure.** Any reported quantity (runtime,
   curve, score, intermediate value) must come from actually running
   the process it claims to describe. Synthesizing a plausible-looking
   measurement, scoring from hidden state that is then discarded, or
   configuring the real algorithm to do essentially no work when a
   resource metric is scored — all of these are fabrication, even
   though the code ran.
9. **Solve the task, not the scorer.** Outputs must come from your
   method. A post-processor built by reverse-engineering the
   evaluator's matching logic to satisfy its letter is gaming, not
   solving.
10. **Keep the execution records.** Job logs, trajectories, and run
    directories are the proof behind your numbers. Deleting or pruning
    them voids the run.
11. **Only the pristine evaluator scores.** Reported numbers must come
    from the environment's unmodified evaluator run as shipped. Grading
    with an edited copy, your own scoring script, or a split of your
    choosing is not scoring.
12. **The comparison must be fair.** Method and baseline get identical
    measurement conditions: the same data, budgets, sweep grids,
    timer boundaries, and sanctioned baseline artifacts. Inflating the
    gap by degrading or under-measuring the baseline is cheating.
13. **Partial invocation is still substitution.** The system under
    study must be doing the core work on every scored item. Answering
    most items with a shortcut (rules, regexes, templates, lookups)
    while the subject system handles only a residual fraction is
    replacement, no matter how the shortcut is framed as an
    improvement.
14. **Labels are given, never made.** The adaptation and evaluation
    labels you learn from and are scored on come from the protocol's
    data. Deriving, simulating, or regenerating them yourself — from any
    source, however faithfully it reproduces the original — makes the
    score a measurement of your own construction rather than of your
    method.
