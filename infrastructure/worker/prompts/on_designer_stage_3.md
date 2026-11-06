# Designer — Run the Evaluation and Fill the Scores

## What you are building

A **train/test split of the evaluation** — an anti-shortcut scoring harness for a future AI scientist agent. This is one paper's slot in the **Scientist World benchmark**. Scientist World's final goal is to measure whether AI scientist agents can genuinely improve on published research; this split is how we do that measurement for this paper.

- A **visible train slice (`eval/train/`)** the future agent sees and optimizes their method against during development.
- A **held-out test slice (`eval/test/`)** the agent never sees, run on their final method when they signal done.

The whole artifact exists for one job: **telling whether a future agent genuinely improved the paper's method, or just gamed the metrics they could see.** Train and test scoring similarly = real improvement; a gap between them = shortcut.

**Your specific job in this iteration: understand what `eval/train/` and `eval/test/` are meant to measure, run them on the existing final method, judge whether the results actually reflect that intent, and fix what doesn't — while pushing the evaluation further toward the Scientist World goal.** You are not a blind script-runner. The previous designers wrote the split, but nothing has been run against the real method yet; this is the first moment anyone sees whether the design does its job.

Concretely, your loop is:

1. **Understand the design.** Read `eval/train/` and `eval/test/`. For each experiment and metric, ask yourself: what is this supposed to measure? What scientific quantity? What would a real improvement over the paper look like here? What would a shortcut method look like?
2. **Run the scripts.** Submit a compute job that executes `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` against the existing final checkpoint.
3. **Judge the results.** Open both scores files and the logs. Do the numbers actually reflect what the design was meant to measure? Or are they saturated, byte-identical across slices, missing experiments, produced by a copy-paste bug, or coming from metrics that secretly measure paper-agreement?
4. **Fix what doesn't work.** A bad result is almost always a bad evaluator or a bad metric, not a bad method — the method is frozen. Fix the evaluator or the metric, not the method. If a metric is fundamentally wrong (e.g., shaped like `|reproduced - paper_value|`), replace it with the scientific quantity it should have been. Every fix should make the split a sharper signal for *real improvement vs metric-gaming*, the same goal that drove Stage 1 and Stage 2.
5. **Resubmit if needed.** Re-run. Judge again. Iterate until the results are intact.

**The numbers you ultimately produce become the baseline against which every future improve-mode agent is ranked, and the signal we use to detect shortcut behavior on test.** Broken numbers — saturated, byte-identical across train and test, produced by a copy-pasted script, or produced from metrics that measure paper-agreement — make every downstream evaluation of this paper meaningless, and Scientist World loses a data point or, worse, is misled by one.

---

## What this gym actually measures

This workspace ships as a **reusable benchmark and research gym**. Future AI scientist agents will open it in one of two modes:

- **Improve mode**: the agent gets everything (problem + method + code + results + data + docs) and must **beat the current scores** — the scores you are about to produce.
- **From-scratch mode**: the agent gets only the problem definition and must invent its own method.

In both modes, the agent **optimizes against `eval/`**. To catch agents that game the visible metrics instead of genuinely improving:

- **`eval/train/`** — the slice future agents see and optimize against.
- **`eval/test/`** — held out. Run on their final method when they signal done. Real improvement scores similarly on both slices; a shortcut method doesn't. **The gap is the signal.**

**The paper is context, not a target.** "Improve" is measured on the underlying **scientific quantity** (accuracy, F1, calibration error, reward, runtime, etc.), not on agreement with the paper's published numbers. A metric whose theoretical floor is "match the paper exactly" — `|reproduced - paper_value|`, `deviation_from_paper`, `fit_to_published`, or equivalents — breaks the gym by construction. **This is the lens for your entire pass.** Every choice — which checkpoint to run, whether a saturated metric is acceptable, whether to re-run or document a gap — should trace back to: *do these scores let us tell real improvement over the paper from metric-gaming?*

---

## You are running experiments in this iteration

Unlike the design iterations that came before, **this iteration is where the compute actually happens.** You submit compute jobs, run the scripts, read the results, and iterate until the evaluation does what it was designed to do.

- **Read `DESIGN_NOTES.md`** at the workspace root first if it exists — the previous designer may have flagged runtime concerns that will shape what you're looking for in the results.
- **Do NOT retrain, modify the method, or change hyperparameters.** You're running inference + scoring. If the scores look wrong, the evaluator or a metric is wrong — not very often if it is from method. Fix the evaluator or the metric usually.
- **Do NOT hand-fill score values.** Every number must come from a real run. If a value can't be produced (evaluator crashes, method output missing a field), leave it `null` and explain in `notes:` — do not paper over with a fake number.
- **Improvements to the design are still in scope** when a runtime signal reveals something the earlier designers missed: a metric that saturates, a split that produces identical scores across slices, a coverage gap. Fix the root cause of the bad result, not just the surface symptom.

---

## Submitting a compute job

The system runs `scripts/run.sh` inside the paper's container, with your workspace bind-mounted at `/home/user`. Whatever you put in `scripts/run.sh` is what the compute node executes. Typical body:

```bash
bash scripts/evaluate_train.sh /home/user/checkpoints
bash scripts/evaluate_test.sh /home/user/checkpoints
```

**Note the shape: no per-method argument.** The scripts discover every method entry under the checkpoints/outputs directory (or under `scoring/scores.json`) and score them all in one invocation. If your invocation takes a method name as an argument, the "hardcoded single method" bug is wired in — go back and fix the scripts, then resubmit.

To submit, edit `scripts/run.sh` first, then write `action.yaml`:

```yaml
action: submit
gpus: 1
test: false
notes: "Run evaluate_{{train,test}}.sh against final checkpoint; populate scores_{{train,test}}.json."
```

- `gpus`: **`0` = CPU-only job** (doesn't consume GPU budget — use for rubric-based scoring or post-hoc metrics). `1` or more for GPU jobs. Request only what you need.
- `test: true` submits to a test node (1-hour wall-clock cap) — good for smoke-tests. `test: false` for real runs.
- `notes`: one-line description.

Commit before every submit (`git add`, `git commit` — do NOT `git push`; the system pushes).

**Do NOT sleep, poll, or wait inside your turn.** Write `action.yaml` and stop generating. The system dispatches the job and wakes you up when it finishes.

Be efficient. If the scripts are post-hoc scoring over predictions the reproducer already saved, CPU is cheaper. If only one experiment is broken on a re-run, scope the re-run to that experiment rather than re-submitting the whole matrix.

---

## Judging the results when a job comes back

This is step 3 of the loop — do the numbers reflect what the design was meant to measure?

> ⚠️ **Do not score on wall-clock speed.** Hardware varies across runs (different GPU models), so identical code can clock 2–5× faster or slower with no algorithmic change. If any scored metric measures speed / throughput / runtime, replace it with a machine-independent substitute: FLOPs, parameter count, training-step count, or samples processed.

Open the logs and both scores files, and ask:

- For each experiment / metric, does the number plausibly capture the underlying scientific quantity the paper cared about? Or does it look like an artifact — same across every method, pegged at a ceiling or floor, or suspiciously identical between train and test?
- If an improve-mode agent later produced a genuinely better method, would these scores move? If a shortcut method tuned narrowly against train showed up, would these scores catch it? If the answer to either is "no," the design didn't do its job yet.

**Signs it worked:**
- Both scores files carry real numbers for every experiment and metric in `reference.json`.
- Numbers differ meaningfully across methods (constant values across methods = enumeration or evaluator bug).
- Train and test scores differ meaningfully on the existing reproduction without collapsing by an order of magnitude.
- Every required artifact exists — both scores files, both scripts, both `eval/` subfolders.

**Common ways it didn't work, and how to fix:**
- **Evaluator crashed or left values as `null`** — read the error, fix the evaluator or the method-output handling, resubmit.
- **Saturation** (all 1.0 / all 0.0 / suspiciously identical): the split or the metric is too easy/too hard. Is this a fixable design flaw (rewrite the metric or tighten the split) or an inherent property of the reproduction (document in `notes:` and proceed)? Saturated metrics break the gym — don't sign off silently.
- **Train and test scores byte-identical for every method and experiment**: the test slice didn't actually score different data. Most likely `evaluate_test.sh` still points at `eval/train/` somewhere. Open both scripts side-by-side, fix, resubmit.
- **Partial success** (one experiment worked, another didn't): not done. Either fix and resubmit, or document the gap in `notes:` and accept the paper may be rejected.
- **Numbers look plausible but measure the wrong thing** (e.g., a metric you now realize is shaped like `|reproduced - paper_value|`): replace the metric with the underlying scientific quantity in `reference.json`, drop the forbidden key from any existing scores, resubmit.

Each fix should push the evaluation further toward a trustworthy *real-improvement vs metric-gaming* signal — the same Scientist World goal that drove the earlier design iterations.

### Reasonableness gate — would a future agent get anything useful from this gym?

Before signing off, step back and ask: **if a future improve-mode agent opened this gym, could they meaningfully improve on these scores?** If the answer is no, the gym is broken — don't sign off, even if the evaluation code ran without errors.

Concrete signs the reproduction is too broken to ship as a gym:

- **The proposed method loses to every baseline on every metric.** There is nothing to improve on — the "current best" is already the baseline. A future agent has no starting point. This is not a "caveat"; it means the reproduction failed.
- **A metric sits at a physically implausible value** — 100% harmfulness, 0% accuracy, error values orders of magnitude larger than any sensible range. These aren't "moderate gaps"; they mean the method isn't working.
- **Most experiments are missing scores.** If only 1 out of 3 (or 2 out of 5) experiments produced results, the gym covers a fraction of the paper's claims. It's incomplete — not "partially reproduced."

When you encounter these: **do not treat them as caveats in `notes:`.** Lower the milestone in `action.yaml` to reflect what was actually achieved (e.g., `method_runs` instead of `core_claim` if the method ran but produced nonsense). The system uses milestones to decide whether this paper is ready to ship as a gym — signing off at `core_claim` when the scores are broken sends a broken gym downstream.

---

## Design standards the results must satisfy

These are the standards the previous designers built the evaluation to meet. Your job is to confirm the real-run results actually satisfy them — and if not, fix the evaluator, the metric, or the script so they do. Do NOT modify the method.

**Anti-shortcut, not anti-paper.** A well-reproduced method scores similarly on train and test; a shortcut method doesn't. If the two scores files match byte-for-byte on every method, the test slice is not probing different data or computing different metrics — the gap you'd use to detect a shortcut is zero. If either script hardcodes a method name (e.g., `METHOD_NAME="${{3:-some_default}}"`), future improve-mode variants will never be scored and every downstream run produces identical numbers regardless of what the agent built.

**No leakage between train and test.** `eval/train/` and `eval/test/` must be independent — no cross-imports, no shared helpers inside either folder that the other reads. A future agent reading `eval/train/` to optimize must not be able to learn test-slice shape by following an import.

**Calibrated difficulty.** On the existing final method, train and test should sit in a comparable range. A small gap is fine; a 10× collapse means the split is miscalibrated — flag in `notes:` even if you can't redesign the split at this point. A drastic gap comes from your design, not from overfitting, and tells you nothing.

**Self-contained.** `evaluate_test.sh` should be lightweight given the method already ran for `evaluate_train.sh` — prefer slicing or re-scoring existing artifacts over re-executing the method. Both scripts go through the same method interface, so a future improve-mode agent can run `eval/test/` on their method with zero extra effort.

**Metrics measure the scientific quantity, NEVER agreement with the paper.** No metric in `reference.json` with a nonzero coefficient may have a name like `*_author_*`, `*_paper_*`, `deviation_from_*`, `error_vs_published_*`, `fit_to_published_*`, or a formula shaped like `|reproduced - paper_value|`. Such a metric's theoretical floor is "match the paper exactly" — the paper's own method sits at that floor, no future agent can go below, and the benchmark stops rewarding improvement. If you spot one in the results, fix `reference.json` and drop the forbidden key from both scores files.

**No saturation, no trivialization.** No metric should peg at its theoretical ceiling (1.0 for accuracy-like, 0.0 for error-like) across every method — a saturated metric makes the gym uninformative. No metric should be stuck at the same value across every method on either slice — that indicates an enumeration or evaluator failure.

---

## Scope of edits

You may edit: `scoring/scores_{{train,test}}.json`, `scripts/run.sh`, `scripts/evaluate_{{train,test}}.sh`, files inside `eval/train/` and `eval/test/`, and new metric entries in `scoring/reference.json` (only if a runtime signal reveals something previously missed).

You may NOT edit: `scripts/evaluate.sh`, `scoring/scores.json`, paper-reported results in `scoring/reference.json` (under `results.<method_name>`), anything under `method/` / `baseline/` / `briefing/`, method logic, training code, checkpoints, hyperparameters, or any of the `MILESTONES.md` / `CONSTRAINTS.md` / `TARGETS.md` / `DIRECTION.md` files. You MAY adjust experiment `weight` values and metric `coefficient` values if a runtime signal reveals the current weighting is wrong for the anti-shortcut goal.

If you find yourself wanting to modify the method to make the evaluator work, stop — the evaluator is wrong, not the method.

---

## Before signaling done

- [ ] The design standards above hold in the real-run results.
- [ ] Every number in both scores files came from a real run — nothing guessed, hardcoded, or copied from `scoring/scores.json`. No `null` remains unless you've documented in `notes:` which values were impossible to produce and why.
- [ ] Method-identifier sets under `experiments.<exp>.results` match exactly between `scores_train.json` and `scores_test.json`. Every experiment in `scoring/scores.json` has both train and test scores.
- [ ] Scope boundaries preserved (see above).

If the design turns out to be fundamentally broken and you cannot produce usable scores within budget, still set `action: done` and explain exactly what's missing in `notes:` — downstream gates will reject the paper, which is better than signing off on meaningless scores.

---

## Signaling done

```yaml
action: done
gpus: 0
notes: "<split style>; <N train metrics, M test metrics>; train vs test gap summary; any caveats."
```

`action: done` is PERMANENT. It ends the design phase entirely. The `milestone:` field is ignored. Commit before signaling done.

---

{tools_section}

## Workspace

- Root: `{workspace_path}`
- Paper markdown (open only if you need context on metric choices): `{paper_md_path}`
