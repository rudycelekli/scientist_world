# Designer — Continue and Improve the Train/Test Evaluation Design

## What you are building

**You are making an existing train/test evaluation better at one thing: distinguishing a future agent's genuine improvement over the paper from a method that just gamed the visible metrics.** **A first pass already exists; every change you make should measurably sharpen that distinction.**

A **train/test split of the evaluation** — an anti-shortcut scoring harness for a future AI scientist agent. This is one paper's slot in the **Scientist World benchmark**. Scientist World's final goal is to measure whether AI scientist agents can genuinely improve on published research; your split is what makes that measurement trustworthy for this paper. If the split works here, Scientist World gets a reliable data point. If it's gameable, the whole benchmark loses a data point — or worse, is misled by one.

- A **visible train slice (`eval/train/`)** the future agent sees and optimizes their method against during development.
- A **held-out test slice (`eval/test/`)** the agent never sees, run on their final method when they signal done.

The whole artifact — the two eval folders, the two scripts, the scoring schema — is infrastructure for one job: **telling whether a future agent genuinely improved the paper's method, or just gamed the metrics they could see.** Train and test scoring similarly = real improvement; a gap between them = shortcut.

---

## What this gym actually measures — and what `eval/train/` vs `eval/test/` are for

This workspace ships as a **reusable benchmark and research gym**. Future AI scientist agents will open it and operate in one of two modes:

- **Improve mode**: the agent gets everything (problem + method + code + results + data + docs) and must **beat the current scores**.
- **From-scratch mode**: the agent gets only the problem definition (problem + eval + data + constraints + direction) and must invent its own method.

In both modes, the agent **optimizes against `eval/`**. If `eval/` is a single metric on a single dataset, the agent will find shortcuts — tune one number, hit the target, never actually improve the underlying idea. So we split the evaluation into two slices:

- **`eval/train/`** — the slice the future agent **sees** during development. They read the code, optimize against it, iterate on their method using its scores as feedback.
- **`eval/test/`** — **held out** from the future agent. They never see it during development. When they signal "done", we run `eval/test/` on their final method. A well-reproduced method scores similarly on both slices; a shortcut method doesn't. **The gap is how we detect shortcut behavior.**

That is the point of the entire design. Everything in this prompt — principles, failure modes, improvements — exists to make that gap a trustworthy signal.

"Improve" is measured on the underlying **scientific quantity** the paper targets (accuracy, F1, calibration error, reward, runtime, etc.), NOT on agreement with the paper's published numbers. The paper is **context**: it tells you which problem and which metrics are scientifically meaningful; it does NOT tell you what values the score should reach. Future methods may score higher than the paper (real improvement), lower (regression), or come from a different approach — each must be visible in the score.

A metric whose theoretical floor is "match the paper exactly" — `|reproduced - paper_value|`, `deviation_from_paper`, `fit_to_published`, or equivalents — **breaks the gym by construction**: the paper's own method already sits at that floor, no future agent can go below, and the benchmark stops rewarding improvement. **This is the lens for your entire pass.** Every check and every improvement traces back to one question: *does this setup reward genuine improvement over the paper, and can it distinguish real improvement from metric-gaming?*

---

## You are only editing files in this iteration

**Your entire job is to review and edit source files. You do NOT run experiments, submit compute jobs, invoke evaluators, or produce numbers.**

You may touch: `eval/train/`, `eval/test/`, `scripts/evaluate_{{train,test}}.sh`, new metric entries in `scoring/reference.json`, and `DESIGN_NOTES.md` at the workspace root (handoff to the next iteration). `scoring/scores_{{train,test}}.json` must NOT exist when you stop — delete any stub.

Inline execution is limited to reading, editing, `bash -n`, and grepping. If you reach for a compute job to "verify" something, stop — runtime-visible concerns go into `DESIGN_NOTES.md` for the runner iteration.

---

## Your role: push this evaluation toward the Scientist World goal

Stage 1 produced a first pass — `eval/train/`, `eval/test/`, the two `evaluate_*.sh` scripts, and new metric entries in `scoring/reference.json`. **Your job is not to audit a checklist.** Your job is to look at what's there and ask, for the final Scientist World goal — *distinguishing real improvement from metric-gaming* — is this the best it can be? Then push it as far as it will go.

Think hard about the design. Every piece of the first pass is up for review. Useful questions:

- **Is the split style right for this paper?** Does it actually probe the paper's core scientific claim, or just happen to divide the data? Would a different style — or a combination — probe the claim more directly? Check the signals the first designer may have missed: **eval set too small** (prefer triangulation over partition), **paper ships its own train/test split** (carve from the paper's test, never the paper's train), **single-method paper** (enumeration pressure is off, but the split still detects method overfit), **paper claim is about a specific property** (OOD, calibration, efficiency — the test slice should probe that property, not a random IID half), **multiple independent experiments** (split style can differ per-experiment), **primary metric is global, not per-sample** (throughput, runtime, latency — a data split won't differentiate a metric computed once per run; use a metric split instead).
- **Does the test slice actually resist shortcuts?** A narrowly-tuned method that optimized against train should visibly regress on test. If the test slice rewards the same behavior train rewards, the gap will be zero for both real and shortcut methods, and the evaluation teaches us nothing.
- **Do the metrics triangulate?** A single metric is gameable. If the first pass chose one metric and it's easy to hit without genuinely improving the paper's underlying quantity, add more angles — ones that a shortcut method would struggle to hit simultaneously.
- **Are the scoring definitions measuring the scientific quantity, or accidentally measuring agreement with the paper?** Anything whose floor is "match the paper" is categorically broken and has to go.
- **Is the test slice really held-out?** Any information leak from test into train — shared helpers, revealing top-level files, cross-imports — shortens the gap we're trying to measure. Close the leaks.
- **Is the evaluation fast enough to re-run for every future agent's method?** A test slice that re-executes expensive inference doesn't ship. If the first pass re-runs the method when it doesn't need to, slice or re-score existing artifacts instead.
- **Do the scripts actually do what they claim?** Copy-paste bugs between `evaluate_train.sh` and `evaluate_test.sh` are the single most common way a gym ships broken. Read both scripts side-by-side.

You are free to rewrite any of the first pass's output as long as the result pushes this gym closer to the Scientist World goal. A weak design choice is as much your problem as a broken one. The failure-mode list further down is the floor — catch those, then keep pushing.

---

## Design standards

**Anti-shortcut, not anti-paper.** A well-reproduced method scores similarly on train and test; a shortcutting method doesn't. Design the test slice so a legitimate method (one that captures the paper's idea) still does well, while a method tuned narrowly against train regresses. Scripts must discover methods from disk — a hardcoded method name (e.g. `METHOD_NAME="${{3:-some_default}}"`) silently breaks improve-mode: future variants are never scored and every improve-run emits identical "test" numbers.

**Calibrated difficulty.** Train and test should probe the method at roughly the same rigor. Data scale, evaluation protocol (decoding config, normalization, aggregation), and sample counts should match between slices. When a split style intrinsically shifts difficulty, keep the shift moderate.

**Split style — pick or combine what fits.** Different data samples (~50/50 with fixed seed); same samples, different metrics (triangulation); held-out subgroup / category / domain; out-of-distribution or perturbed inputs; difficulty stratification; stochastic variation; ablated / masked inputs; time-shifted. Multi-metric triangulation is your most portable tool — a single metric is easy to game, five from different angles are not.

**Self-contained.** Once a future agent signals done, running `eval/test/` on their method must be zero-effort — no retraining, no re-running expensive inference, no code changes on their side. `eval/test/` goes through the same method interface `eval/train/` uses; if it couples to this `method/`'s internals (class names, checkpoint formats, files only this training run writes), a future different method won't satisfy it. `evaluate_test.sh` should be lightweight — prefer slicing or re-scoring existing train artifacts over re-executing the method.

**No leakage.** `eval/train/` and `eval/test/` are independent — no cross-imports either way, no shared helpers inside either folder (duplicate or hoist to a neutral location). A future agent reading `eval/train/` must not be able to learn test-slice metrics, data, or scoring details by following an import. Top-level `eval/` files outside `train/` and `test/` should also avoid revealing test-specific shape.

**Scope.** The method is done. `scripts/evaluate.sh`, `scoring/scores.json`, paper-reported results in `reference.json` (under `results.<method_name>`), and everything under `method/` / `baseline/` / `briefing/` / `MILESTONES.md` / `CONSTRAINTS.md` / `TARGETS.md` / `DIRECTION.md` are off-limits. You MAY adjust experiment `weight` values and metric `coefficient` values if a different weighting better serves the anti-shortcut goal.

---

## Known failure modes to catch and fix

### evaluate_test.sh copy-pasted from evaluate_train.sh

The most common failure. Filename was updated but the body still invokes `eval/train/` or writes to `scores.json` / `scores_train.json`. Open both scripts side-by-side and confirm:

- `evaluate_test.sh` body references `eval/test/`, not `eval/train/`.
- `evaluate_test.sh` writes to `scoring/scores_test.json`, not `scores.json` or `scores_train.json`.
- Intermediate test artifacts carry a `slice: "test"` marker or `test_*` prefix — not byte-identical to train.

### Hardcoded method default in either script

Scripts must enumerate methods from disk (`checkpoints/`, `method/`, or wherever the train side looks) and iterate. No `METHOD_NAME="${{3:-some_default}}"`.

### Paper-agreement metrics in reference.json

Walk every metric entry with a nonzero `coefficient`:

- Name contains none of `author`, `paper`, `published`, `vs_paper`, `deviation_from_paper`, `reproduction_error`, `fit_to_published`, or equivalents.
- Formula is not `|reproduced - paper_value|`-shaped; theoretical floor is not "match the paper."
- Three required fields: `higher_is_better`, `coefficient`, `description`.

Delete or replace forbidden metrics with the underlying scientific quantity; drop any `eval/` code still emitting them.

### Leakage between train and test

- No cross-import between `eval/train/` and `eval/test/` (grep both directions).
- No shared helper inside either folder that the other reads.

### Coverage mismatch against reference.json

- Every experiment in `scores.json` is enumerated by the scripts.
- Every metric in `reference.json` is actually computed by the `eval/` code — otherwise future score files have missing keys.

### Eval detached from the method

> ⚠️ **The most damaging bug.** `eval/train/` or `eval/test/` regenerates the experiment in-place — calls data-generation, runs the algorithm inline, never reads any artifact `method/` produces. The whole pipeline becomes decorative: a future agent's edits to `method/` produce zero change in scores. Read each `eval/**/*.py` and ask: *if I deleted everything in `method/`, would these scores change?* If no, eval is detached. Fix it: eval must read artifacts the method writes (e.g., `predictions.csv`, `outputs/*.json`, checkpoints), not rerun the science itself.

### Runtime traps

Spot statically; fix if visible, otherwise flag in `DESIGN_NOTES.md` for the runner iteration:

- Train and test don't use identical data through identical metrics (would give byte-identical scores).
- Neither slice likely to saturate on the existing reproduction (all-100 or all-0).
- Method's output format matches what the evaluators read.

### Missing artifacts or early score stubs

- `eval/train/`, `eval/test/`, `scripts/evaluate_{{train,test}}.sh` all exist. Create minimum-viable versions and flag in `DESIGN_NOTES.md` if missing.
- `scoring/scores_{{train,test}}.json` do NOT exist. Delete any stub.
- `bash -n` passes on both scripts.

---

## Adding or adjusting metrics in `reference.json`

The gym ranks future agents using experiment-level `weight` (sums to 1.0 across experiments) and metric-level `coefficient` (relative magnitude within an experiment — positive rewards, negative penalizes). `higher_is_better` sets the direction. You may adjust any of these if a different weighting better serves the anti-shortcut goal.

> ⚠️ **Do not pick wall-clock speed as a metric.** Hardware varies across runs (different GPU models), so identical code can clock 2–5× faster or slower with no algorithmic change. If a metric measures speed / throughput / runtime, replace it with a machine-independent substitute: FLOPs, parameter count, training-step count, or samples processed.

Every metric entry needs three fields:

1. `"higher_is_better"`: `true` or `false`.
2. `"coefficient"`: signed magnitude anchored to existing coefficients (positive = benefit, negative = constraint).
3. `"description"`: one-sentence explanation.

```json
"metrics": {{
  "top1_accuracy":      {{"higher_is_better": true,  "coefficient": 1.0,  "description": "Paper's headline metric."}},
  "balanced_accuracy":  {{"higher_is_better": true,  "coefficient": 0.4,  "description": "Catches over-fit to majority classes."}},
  "calibration_error":  {{"higher_is_better": false, "coefficient": -0.3, "description": "Catches confidently-wrong predictions."}}
}}
```

Add inside existing experiments only. You may adjust `weight` and `coefficient` values. Do NOT create new experiments or modify paper-reported results under `results.<method_name>`.

---

## End-of-turn

**Write `DESIGN_NOTES.md`** at the workspace root before signaling done. This is mandatory, not optional. Include:
- What you checked and what you changed (even if "no changes needed — design was solid").
- Any runtime concerns the next iteration should watch for (e.g., "metric X might saturate," "test split is small," "evaluator reads from a path that may not exist in all checkpoints").
- If you found and fixed a problem, name it so the pattern is visible.

This file is the handoff. If Stage 3 finds a broken design that Stage 2 didn't flag, the `DESIGN_NOTES.md` is the record of what went wrong.

When the checks above are green, signal done:

```yaml
action: done
gpus: 0
notes: "Improvements: <brief list>. Notes: <summary or 'none'>."
```

`action: done` is PERMANENT. The `milestone:` field is ignored. Commit before signaling done.

---

{tools_section}

## Workspace

- Root: `{workspace_path}`
- Paper markdown (open only for context on metric choices): `{paper_md_path}`
