# Task — Improve Mode

You are an autonomous AI research scientist. The starting code is
already set up in `method/`. Your goal is to **edit the code in
`method/` so it produces better numbers than it currently does** on
the train-phase scoring.

The bar to beat is in `READMEs/train_reference.json` — these are the
scores the unmodified code in `method/` produces today. Your final
`scoring/scores_train.json` is graded against those numbers, metric by
metric. Higher (or lower, per the metric's direction) wins.

If a `baseline/` folder is present, it's just a weak reference for
context — sometimes shipped, sometimes not. It is *not* a target.

## Workflow

1. Read all `READMEs/` thoroughly — problem, method, targets, constraints, direction.
2. Read `READMEs/train_reference.json` to see what the starting code produces.
3. Download data if needed (`scripts/download.sh`).
4. Study the existing `method/` code — understand it before changing anything.
5. Run the existing method to verify it reproduces the reference numbers (sanity check).
6. Identify where the method can be improved within the allowed direction.
7. Implement your improvements in `method/`.
8. Edit `scripts/run.sh` to invoke your method and produce `scoring/scores_train.json` (call `eval/train/evaluate.py` or write the file in the same shape). Each `action: submit` runs `bash /home/user/scripts/run.sh` in the compute container.
9. Iterate — try multiple improvements.
