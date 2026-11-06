## Workspace

Read every file in `READMEs/` before doing anything else:

- `problem.md` — the research problem
- `evaluation.md` — metrics, baselines, evaluation protocol
- `TARGETS.md` — what to optimize and how improvement is measured
- `CONSTRAINTS.md` — what you must NOT change
- `DIRECTION.md` — the allowed scope of methodological approaches
- `EXPERIMENTS.md` — what each experiment tests (maps to keys in `scores_train.json`)
- `train_reference.json` — *(improve mode only)* the scores the starting code in `method/` produces. This is the bar to beat. Re-run the code to verify before you trust these numbers; they are informational, not authoritative.

## Rules

1. Do NOT modify anything in `eval/` or `data/`. The evaluation framework and data are fixed.
2. Do NOT change the evaluation protocol described in `evaluation.md`.
3. Respect every constraint in `CONSTRAINTS.md` — violations invalidate your results.
4. Stay within the research direction defined in `DIRECTION.md`.
5. Your compute entrypoint is `scripts/run.sh`. Each `action: submit` runs `bash /home/user/scripts/run.sh` inside the compute container. Put your training / inference / scoring command there, and have it write `scoring/scores_train.json` by calling `eval/train/evaluate.py` — the file must be produced by the evaluator, never written directly.
6. Use `scripts/download.sh` to fetch any required data or models before running experiments.
7. Put your method code under `method/`, your run scripts under `scripts/`.
8. The workspace is the source of truth. If the briefing refers to an artifact that is not in it, producing that artifact is part of the task, not a reason to stop. Building a missing artifact is never a constraint violation, and no one will add files to the workspace on request.

## About `baseline/`

If a `baseline/` folder is present, it's just a weak reference kept
for context — sometimes papers ship one, sometimes they don't. It is
**not** a target to beat. If it's there, you can use it as a sanity
check on the harness or to see the I/O shape end to end. Matching or
exceeding its numbers does not constitute a result. If it's not there,
no problem — proceed without it.

## How to do an experiment

Work like a real researcher: form a hypothesis, implement it, run an
experiment, read the feedback honestly, and use what you learned to
design the next one. Use your own judgment for the details — you know
how to do research.

**Record your thinking in `idea/`.** Each time you meaningfully refine
your approach, add a new file: first `idea/idea_v1.md`, then
`idea/idea_v2.md`, and so on. Never overwrite an earlier version — the
history is the reasoning trace. One clear paragraph per version is
enough: the idea itself, the motivation, and what changed from the
previous version.

**Regularly commit your work — no need to push.**
