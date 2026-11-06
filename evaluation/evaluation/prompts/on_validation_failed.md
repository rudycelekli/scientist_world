# Scoring file is incomplete — please fix and re-submit

You signalled `done`, but `scoring/scores_{phase}.json` is missing
entries that the run is expected to contain. The run cannot be marked
complete until those entries are present.

This is validation retry **{retry_n} of {max_retries}**. After that,
the run is marked `incomplete`.

## Specifically, the following are missing

{gaps}

## What to do — read the section that matches the `{phase}` above

### If you are in the train phase

1. **Check why these metrics weren't produced last time.** The exact
   way to compute them is already defined in `scripts/evaluate_train.sh`
   and the code under `eval/train/`. If your `scripts/run.sh` doesn't
   invoke that path (or invokes it with a filter that drops these
   metrics, or runs the method on a subset of inputs that excludes
   what these cells need), that's the bug — fix it.
2. Edit `scripts/run.sh` so it produces every missing metric, then
   submit a fresh job to regenerate the results.
3. When the file is complete, signal `done` again.

### If you are in the test phase

The held-out test scorer (`eval/test/`) is read-only and `scripts/run.sh`
does NOT execute in this phase — only `evaluate_test.sh` does. So
"edit run.sh" doesn't apply. The fix is different:

1. **Verify what's there before concluding it's unfixable.** `ls` the
   artifact directory the test scorer reads (often `tmp/release_results/`,
   `outputs/`, `predictions/`, etc. — check what `evaluate_test.sh`
   passes to the scorer). Open one of the existing files. The artifacts
   may be present but in the wrong shape (e.g. only train-split inputs
   when the test scorer wants test-split inputs).
2. **Re-execute your existing `method/` code** on the missing inputs to
   produce the artifacts the test scorer needs. `method/` is read-only,
   meaning you cannot *edit* the code — but you can still *run* it.
   For example: if your earlier `run.sh` ran the method only on
   `--start-indices 1,3,5,7,9` (the train split), invoke the same
   `python -m method.<runner>` again with `--start-indices 2,4,6,8,10`
   to fill in the test split. Write the output to the same artifact
   directory the test scorer reads.
3. Run that re-execution as `action: submit`, not inline bash —
   inline counts against your 4-hour turn cap and gets killed mid-job.
4. Once the missing artifacts exist, signal `submit` — the pipeline
   will rerun `evaluate_test.sh` and the scorer should now produce
   `scores_test.json` covering every required cell.

## Stay honest

This is **not** a request to fabricate numbers. Compute every missing
metric the same way the rest of your method's metrics were computed —
by running your code over the same data and letting `eval/{phase}/`
record what your method actually produces.

Hardcoding values, copying numbers between methods, or post-processing
predictions to game a metric is treated the same as cheating elsewhere
in the run — automatic disqualification regardless of the score.
