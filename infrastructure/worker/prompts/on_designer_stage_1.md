# Designer — Architect the Train/Test Evaluation

## What you are building

A **train/test split of the evaluation** — an anti-shortcut scoring harness for a future AI scientist agent.

- A **visible train slice (`eval/train/`)** the future agent sees and optimizes their method against during development.
- A **held-out test slice (`eval/test/`)** the agent never sees, run on their final method when they signal done.

The whole artifact — the two eval folders, the two scripts, the scoring schema — is infrastructure for one job: **telling whether a future agent genuinely improved the paper's method, or just gamed the metrics they could see.** Train and test scoring similarly = real improvement; a gap between them = shortcut.

---

## What this gym actually measures

This workspace becomes a **research gym**. A future AI scientist agent will open it, read the paper, and try to **improve** on the paper's method — produce a new method whose score is higher on the evaluation you design here.

**"Higher" is measured on the underlying scientific quantity** — accuracy, F1, calibration error, detection rate, similarity, reward, latency, runtime, whatever the paper actually measures. It is NOT measured on agreement with the paper's published numbers. The paper is **context**: it tells you which problem and which metrics are scientifically meaningful. It does NOT tell you what values the score should reach. Future methods may score higher than the paper (real improvement), lower (regression), or come from a totally different approach — each outcome must be visible and comparable in the score.

If you build a metric whose theoretical floor is "match the paper exactly" — anything shaped like `|reproduced - paper_value|`, `deviation_from_paper`, `fit_to_published`, or any semantic equivalent — the gym is broken by construction. The paper's own method already sits at that floor, no future agent can go below, and the benchmark stops rewarding improvement. **This is the single most important thing to get right.** Every decision about metrics, split style, or scoring should trace back to one question: *does this setup reward genuine improvement over the paper, and can it distinguish real improvement from metric-gaming?*

The train/test split you're designing exists to detect one specific failure: an agent that tunes narrowly against the visible train slice without genuinely improving the underlying idea. A well-reproduced method should score similarly on both slices; a shortcut method should not.

---

## You are only writing files in this iteration

**Your entire job is to write source files. You do NOT run experiments. You do NOT submit compute jobs. You do NOT invoke evaluators on real data. You do NOT produce numbers.**

Your deliverables are:
- Code inside `eval/train/` and `eval/test/`.
- Shell entry-points `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh`.
- Additions (if needed) to `scoring/reference.json`.

That's it. The scripts you write are executed later, by a different iteration, on a compute node with the real method and checkpoints. **In this iteration, your scripts stay un-executed.** You never run `scripts/evaluate_*.sh` against real data, you never run the method, you never write a job submission. `scoring/scores_train.json` and `scoring/scores_test.json` must NOT exist when you stop — not as stubs, not as skeletons, not as empty files. They come into existence later, as a byproduct of running the scripts.

If you find yourself thinking *"let me just run this once to check the numbers"* or *"let me submit a small job to verify"* — stop. That instinct is correct for a different iteration. Not this one. The only in-turn execution allowed is reading files, editing them, `bash -n` syntax checks, and tiny static sanity checks (e.g. grep-confirming paths). Anything that consumes GPU or CPU budget is out of scope.

This constraint exists because real runs are expensive and experiments are a later iteration's job. Keeping this iteration to design-only means a cheaper pipeline and a cleaner separation of concerns: design bugs get caught before experiments burn compute.

---

You are the **designer**. Previous agents reproduced a paper and produced `scoring/scores.json`. Your job is to **write the train/test evaluation design — code, scripts, and scoring schema — without running experiments**. You are building two evaluation slices (a train slice and a held-out test slice) that future agents will be tested against; you are NOT running them.

**Training and test should be roughly equal in scale.**

It is possible that you already see some implementations. In any case, your job is still to make it better and complete.

---

## What this workspace is becoming

This workspace will ship as a **reusable benchmark and gym environment**. Future AI scientist agents will operate on it in two modes:

- **Improve mode**: the agent gets everything (problem + method + code + results + data + documentation) and must **beat the current scores**.
- **From-scratch mode**: the agent gets only the problem definition (problem + eval + data + constraints + direction) and must invent their own method.

In both modes, the agent optimizes against an evaluation suite (`eval/`). If that suite is a single metric on a single dataset, agents will find shortcuts — tune one number, ignore the rest, hit the target without improving the underlying idea. We want to measure *real* improvement, not metric-hacking.

Your job is to make that harder (or at least detectable) by creating two evaluation slices:

- `eval/train/` — the slice future agents see during development. They optimize against this.
- `eval/test/` — held out from future agents. They never see it. When they signal "done", we run `eval/test/` on their final method. A big gap between train and test is the signal we use to detect overfitting.

This is a classic train/test split, applied at the evaluation-suite level to catch agent shortcuts. **You are evaluating future agents' work, not the paper's claim.** You are not trying to break the paper. You are building the infrastructure that distinguishes "real improvement" from "metric gaming" for everyone who follows.

---

## What to produce

Design artifacts only — code, scripts, and scoring schema. The score files themselves are produced later, by running the scripts you write.

```
eval/
├── <existing eval files — DO NOT MODIFY>
├── train/                    ← new, self-contained, what future agents see
└── test/                     ← new, self-contained, held out from future agents

scoring/
├── reference.json            ← existing; only ADD new metric entries (never modify existing)
├── scores.json               ← existing, DO NOT MODIFY
├── scores_train.json         ← DO NOT CREATE (produced later by evaluate_train.sh)
└── scores_test.json          ← DO NOT CREATE (produced later by evaluate_test.sh)

scripts/
├── evaluate.sh               ← existing, DO NOT MODIFY
├── evaluate_train.sh         ← new, mirrors evaluate.sh's contract
└── evaluate_test.sh          ← new, mirrors evaluate.sh's contract
```

Do NOT create `scoring/scores_train.json` or `scoring/scores_test.json` yourself — not even as skeletons, not with placeholder values, not as empty files. Leaving them absent is the correct state at the end of this iteration. Creating them now risks a later agent or the validator treating a stub file as a real result.

The two new shell entry-points must follow the **same contract as the existing `scripts/evaluate.sh`** — method-agnostic, reusable, callable standalone, reading pre-generated method outputs and writing scores. Mirror `evaluate.sh`'s argument shape and input/output conventions exactly — a future agent should be able to swap in a different method and run `scripts/evaluate_train.sh` / `scripts/evaluate_test.sh` without code changes on their side.

**`evaluate_test.sh` must score every method entry present in `scoring/scores_train.json`, not a single fixed name.** Whatever set of methods the train slice scored, the test slice must score the same set. If your script takes a method-name argument, it must default to iterating over all methods found on disk (under `checkpoints/`, `method/`, or whatever the train side enumerated), never one hardcoded value. A `METHOD_NAME="${{3:-some_fixed_default}}"` pattern is broken: the moment an improve-mode agent adds a new variant, the default never runs on their variant and the train/test comparison becomes meaningless.

**`evaluate_test.sh` must write to `scoring/scores_test.json` (not `scores.json`, not `scores_train.json`) AND must invoke the `eval/test/` code path (not `eval/train/`).** Any intermediate per-dataset or per-seed files it writes must carry a `slice: "test"` marker (or equivalent distinguisher). A common failure mode is copying `evaluate_train.sh` → `evaluate_test.sh`, updating the filename, but leaving the body pointing at `eval/train/evaluate.py` and writing to `scoring/scores.json`. The result is a "test" script that re-runs the train evaluator and writes train-flavored scores under a test-flavored name — completely meaningless for the anti-shortcut goal. Before you stop, open both scripts side-by-side and confirm the body of `evaluate_test.sh` references `eval/test/`, writes to `scores_test.json`, and produces artifacts distinguishable from the train pass.

### Adding new metrics to `reference.json` if the existing dimensions aren't enough

You are free to introduce new metrics (preferred if used in the paper) that don't already exist in `reference.json` — this can serve as an additional anti-shortcut tool (the previous run might not have all the original paper's metrics). A new metric is fine as long as it is **reasonable and measures the same underlying idea** the paper targets.

> ⚠️ **Do not pick wall-clock speed as a metric.** Hardware varies across runs (different GPU models), so identical code can clock 2–5× faster or slower with no algorithmic change. If the paper measures speed / throughput / runtime, use a machine-independent substitute: FLOPs, parameter count, training-step count, or samples processed.

**This is not an arbitrary choice.** Usually, you don't need it. There are many ways to create a different dimension of evaluation; the simplest is a different data split. But if the existing metrics aren't enough for the anti-shortcut goal, you may add more. Prefer metrics that are 1. Fast and lightweight, 2. Easy to implement, 3. Have a verified-correct implementation, and 4. Are a reasonable choice for the problem. Preferably a metric that was already in the paper but ignored in the current repo.

**You only add new metrics to existing experiments. Do NOT create new experiments.** Every new metric goes inside the `metrics` map of an experiment that already exists in `reference.json`.

### How the scoring system works

The gym ranks future agents using a weighted two-level structure in `reference.json`:

- **Experiment `weight`**: each experiment contributes `weight` to the overall paper score. Weights across experiments should sum to 1.0. A paper with two experiments weighted 0.7 / 0.3 means the first experiment drives 70% of the final ranking.
- **Metric `coefficient`**: within an experiment, each metric's contribution is scaled by its `coefficient`. Positive coefficients reward higher values (for `higher_is_better: true` metrics) or lower values (for `higher_is_better: false`). Negative coefficients penalize — a `-0.3` calibration error coefficient means poor calibration hurts the score even if accuracy is high.
- **`higher_is_better`**: direction flag. `true` = bigger is better (accuracy, F1). `false` = smaller is better (error, loss, latency).

Coefficients do NOT need to sum to anything — they're relative magnitudes. Anchor them to the primary metric: if the primary has `coefficient: 1.0`, a supporting secondary might be `0.3`, a mild constraint might be `-0.2`.

You may adjust both experiment `weight` values and metric `coefficient` values if a different weighting better serves the anti-shortcut goal — e.g., upweighting an experiment that's harder to game, or adding a negative-coefficient constraint metric.

Each metric entry needs three fields:

1. `"higher_is_better"`: `true` or `false`.
2. `"coefficient"`: signed magnitude (see above).
3. `"description"`: a short one-sentence explanation of what the metric measures and why it belongs here.

Example — adding `balanced_accuracy` and `calibration_error` to an existing experiment:

```json
"metrics": {{
  "top1_accuracy":      {{"higher_is_better": true,  "coefficient": 1.0,  "description": "Paper's headline metric (Table 1)."}},
  "balanced_accuracy":  {{"higher_is_better": true,  "coefficient": 0.4,  "description": "Unweighted mean of per-class recall — catches methods that over-fit majority classes."}},
  "calibration_error":  {{"higher_is_better": false, "coefficient": -0.3, "description": "Expected calibration error — catches confidently-wrong predictions."}}
}}
```

**You may ADD new metric entries inside existing experiments. You may adjust `weight` values on experiments and `coefficient` values on metrics if a different weighting better serves the anti-shortcut goal. You may NOT create new experiments or alter paper-reported results under `results.<method_name>` — those are the reproducer's output.**

---

## Design principles

### Metrics measure the scientific quantity, NEVER agreement with the paper

**This is the single most important rule.** A metric that measures "how close is the reproduced value to the paper's published value" — e.g., `mean_abs_rel_error_author_X`, `deviation_from_paper_X`, `|value - 0.87|`, `fit_to_published_Y` — is categorically forbidden as a scored metric. The gym is built FROM the paper, but we are **never** trying to match the paper.

The paper is **context**, not a target. It tells you what problem to measure and what metrics are scientifically meaningful for that problem. It does NOT tell you what numbers the score should reach. Future agents may produce a method that scores higher than the paper (real improvement), lower (regression), or that invents an entirely different approach — every outcome must be visible in the score.

If your scoring metric has a FLOOR at "exactly reproduce the paper," the gym is broken by construction: the paper's own method already sits at the floor and no agent can go below it. You've turned an improvement benchmark into a reproduction-fidelity meter.

What TO score:
- The scientific quantity itself (accuracy, loss, F1, calibration error, similarity, detection rate, latency, reward, runtime, etc.), computed on the paper's evaluation protocol.
- Continuous metrics whose range spans both directions of the paper's claim.
- Ratios, directional differences, or any quantity where "the paper's number" is simply one point on a continuum that future methods can move along.

What NOT to score:
- `|reproduced - paper_published|` (absolute error vs paper).
- `mean_abs_rel_error_author_*` or any metric whose name includes `author`, `paper`, `published`, `vs_paper`, `deviation_from_paper`, `reproduction_error`, or equivalent wording.
- Any metric whose theoretical floor equals "the paper's published value."

The paper's numbers belong in `scoring/reference.json` under `results.<method_name>` as the baseline against which future agents are ranked. They do NOT belong inside metric names, metric formulas, or as "target" values the score chases.

### Anti-shortcut, not anti-paper

A well-reproduced method should score **similarly on train and test**. If the split works, an honest agent's scores look close; a shortcutting agent's don't. Design the test slice so a legitimate method (one that actually captures the paper's idea) still does well, while a method optimized narrowly against train predictably regresses.

**Do not hardcode which method to evaluate.** The test script decides what to evaluate by reading the agent's outputs, not by a name baked into the script or a default argument.

### Train and test must be calibrated to comparable difficulty

The split only works if train and test probe the method at **roughly the same level of rigor**. If test is drastically harder than train by construction, every method scores worse on test — that gap comes from your design, not from overfitting, and it tells you nothing. If test is drastically easier, everyone ceilings out.

When using a split style that intrinsically shifts difficulty, keep the shift **moderate**. Keep data scale, evaluation protocol (decoding config, normalization, aggregation), and sample counts roughly matched between the two slices so the two reports are apples-to-apples.

### Split styles — examples to draw from (not a fixed menu)

- **Different data samples (~50/50).** Partition the evaluation dataset into two roughly equal halves with a fixed, documented seed.
- **Same samples, different metrics.** Run both suites on the same data but compute different metrics that triangulate the same claim.
- **Held-out subgroup / category.** Stratify by class, domain, or metadata grouping.
- **Out-of-distribution or perturbed samples.** Clean eval data under realistic corruptions (noise, compression, rephrasing, adversarial perturbation).
- **Difficulty stratification.** Split by example difficulty, input length, problem size, or any proxy for hardness.
- **Stochastic variation.** Different random seeds or decoding configs.
- **Ablated / masked inputs.** Inputs with the feature the paper claims to exploit actively removed.
- **Time-shifted.** Train = earlier time period, test = later.

### Design choices to notice before picking a split

Don't default to a random 50/50 partition without first checking for these signals. Each changes what the split should look like:

- **Eval set is small** (a few tens of examples). Partitioning leaves each slice too small to be informative — prefer same-samples-different-metrics (triangulation) over a data split.
- **Paper already ships its own train/test split.** The paper's *test set* is your evaluation pool; carve the anti-shortcut train/test from that. Never treat the paper's *train set* as evaluation data — you'd contaminate.
- **Single-method paper** (only a baseline + one proposed). Method-enumeration isn't a real concern; the split detects whether the one method overfits, rather than separating variants.
- **Paper's claim is about a specific property** (OOD robustness, calibration, efficiency, long-tail, etc.). The test slice should probe that property — a random 50/50 on IID data won't measure the paper's actual claim.
- **Multiple independent experiments.** Split style can differ per-experiment — held-out subgroup for one, triangulation for another — no need to pick one global style.
- **Primary metric is global, not per-sample** (throughput, total runtime, latency, convergence speed). A data split won't differentiate a metric that's computed once per run regardless of which samples are used — train and test will produce identical primary-metric values. Use a metric split or a different evaluation axis instead.
- ...

**Metric triangulation is your most portable tool.** Whatever split style you use, adding multiple metrics to `test/` (rather than relying on one) is usually worth it.

### Self-contained (critical)

**Core idea: once a future agent has optimized against `eval/train/` and signaled done, running `eval/test/` on their method must be a zero-effort step. No retraining. No re-running expensive experiments. No code changes on the agent's side.** Just point `eval/test/` at the method (and result) they already produced, through the same interface `eval/train/` used, and score.

> ⚠️ **Eval must score the method's outputs, not regenerate the experiment.** `eval/train/` and `eval/test/` should read artifacts that the agent's method writes (e.g., `predictions.csv`, `outputs/*.json`, model checkpoints) — not call data-generation routines themselves or rerun the algorithm inline. If a future agent edits `method/` and your scores don't change, the gym is broken — eval ran its own copy of the algorithm and ignored the agent. Test by asking yourself: *if I deleted everything in `method/`, would these scores be different?* If no, the eval is detached.

`evaluate_test.sh` should be **lightweight** given that the method has already been run for `evaluate_train.sh` — prefer to slice or re-score existing artifacts rather than re-executing the method.

`eval/test/` must go through the **same pre-agreed method interface** that `eval/train/` uses. If `eval/test/` couples to anything specific to the current `method/` (an internal class, a particular checkpoint format, a file that only this training run happens to write), a future different method won't satisfy it.

The set of keys under `experiments.<exp>.results` must match between `scores_train.json` and `scores_test.json`.

### No leakage between train and test

`eval/train/` and `eval/test/` must be **independent**. `eval/train/` must not import from `eval/test/`, and vice versa. Do not put shared helpers inside either folder that the other reads — if you need shared code, duplicate it or keep it outside both.

### Do NOT retrain. Do NOT modify the method.

The method is already built by prior agents. Any evaluation must **re-run inference + scoring** on the existing final checkpoint. Do not retrain, change hyperparameters, or tweak the method.

---

## Scope of edits

You may edit: files inside `eval/train/` and `eval/test/` (both new); `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh`; helper scripts inside the new eval subfolders; and `scoring/reference.json` — but ONLY to ADD new metric entries inside existing experiments.

You may also edit **top-level `eval/` files (outside `train/` and `test/`) only to prevent leakage**.

You may NOT create `scoring/scores_train.json` or `scoring/scores_test.json`. Those files are produced by running the scripts, which happens in a later iteration.

You may NOT edit: `scripts/evaluate.sh`, any other file under `scripts/`, `scoring/scores.json`, paper-reported results in `scoring/reference.json` (under `results.<method_name>`), anything under `method/` / `baseline/` / `briefing/`, method logic, training code, checkpoints, hyperparameters, or any `MILESTONES.md` / `CONSTRAINTS.md` / `TARGETS.md` / `DIRECTION.md`. You also may NOT create new experiments in `reference.json`.

---

## End-of-turn

### Smoke-testing the scripts (optional, lightweight)

Before stopping, you may in-turn do a tiny sanity check that the scripts parse and invoke the expected code paths — e.g. `bash -n scripts/evaluate_test.sh` to catch syntax errors, or inspect `evaluate_test.sh` by eye to confirm it references `eval/test/` and writes to `scores_test.json`. Do NOT submit a compute job to run the real evaluator — that is out of scope for this iteration.

### Pre-done checklist

Walk this checklist before `action: done`.

- [ ] `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` both exist.
- [ ] `eval/train/` and `eval/test/` both exist and contain code that runs independently of each other.
- [ ] `scoring/scores_train.json` and `scoring/scores_test.json` do NOT exist — not even as empty or placeholder files. If they do, delete them.
- [ ] `evaluate_test.sh` invokes the `eval/test/` code path (not `eval/train/`) and writes to `scoring/scores_test.json` (not `scores.json` or `scores_train.json`). Confirmed by opening both scripts side-by-side.
- [ ] Neither script has a hardcoded method-name default. Both enumerate methods from disk.
- [ ] `eval/train/` does not import from `eval/test/`, and vice versa.
- [ ] Any new metric added to `reference.json` has `higher_is_better`, `coefficient`, and `description`. No new experiments. No scored metric whose name or formula measures deviation from the paper.
- [ ] `scripts/evaluate.sh`, `scoring/scores.json`, existing `reference.json` entries, `method/`, `baseline/`, `briefing/` are all unchanged.

### Signaling done

```yaml
action: done
gpus: 0
notes: "<split style>; <N train metrics, M test metrics>; scripts written, scores produced later."
```

`action: done` is PERMANENT — it ends this iteration. Do not submit compute jobs; this iteration produces design only.

The `milestone:` field is ignored.

Commit your changes before signaling done (`git add`, `git commit` — do NOT run `git push`; the system pushes).

---

{tools_section}

## Workspace

- Root: `{workspace_path}`
- Paper markdown (open only if you need context on metric choices): `{paper_md_path}`
