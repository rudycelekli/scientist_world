# Final Verification

You are the final step in a multi-agent workflow. The above prompt is given to previous agents (some still apply to you). Previous agents have reproduced a paper and iterated on the results. Your job is a **quality check** — make sure the workspace is clean, correct, and ready to serve as a research gym for future AI scientist agents.

The previous agent reached milestone `{previous_milestone}`. You start at that milestone — do NOT reset it unless you find genuine issues.

Any improvement to the markdown files are encouraged; making them clear, precise, and complete is critical for a successful gym.

---

## What This Workspace Becomes

This workspace will be used as a **benchmark environment** where AI scientist agents try to beat the reproduced scores. They operate in two modes:

- **Improve mode**: they get everything (problem + method + code + results + data + documentations) and must improve on it
- **From-scratch mode**: they get only the problem definition (problem + eval + data + documentations of constraints and directions) and invent their own method

For both modes to work:
- **Problem/solution separation must be clean**: `briefing/problem.md`, `briefing/evaluation.md`, `eval/`, `data/` must be completely method-agnostic. A scientist in from-scratch mode sees only these.
- **Scoring must be correct**: `scoring/reference.json` defines the metrics and their relative importance via `coefficient`. `scoring/scores.json` has the actual results. The scientist's improvement is measured by these — wrong coefficients mean wrong incentives.
- **Evaluation must be reusable**: `scripts/evaluate.sh` must work for ANY method that produces the right output format, not just the paper's code.

The `coefficient` field matters. It defines how much we care about relative improvement in each metric. Positive = benefit, negative = constraint. Wrong coefficients make the gym measure the wrong thing.

---

## **Method-Agnostic Files Are Critical**

Future scientist agents must be able to understand the problem and evaluation without being told the paper's method. 
If we leak the method, the environment is not useful.

Treat the following files as **strictly method-agnostic**:
- `briefing/problem.md`
- `briefing/evaluation.md`
- `scoring/EXPERIMENTS.md`

Treat the following files as **high-level and method-safe**:
- `scoring/TARGETS.md`
- `scoring/CONSTRAINTS.md`
- `scoring/DIRECTION.md`

These files must NOT leak ANY architecture, algorithm, module design, training recipe, loss design, or implementation tricks from the paper. They should describe:
- what problem is being solved
- what each experiment is trying to establish
- what must stay fixed
- what research direction is allowed

If these files leak the paper's method, the gym is compromised for from-scratch scientist agents. Treat this as a real issue, not a cosmetic one. Fix if needed.

---

## Verification Checklist

Go through these in order. Fix issues you find, but **do not rewrite working code or waste compute re-running experiments that already succeeded.**

### 1. Folder Structure and correctness
- **Is the workspace root clean?** The root should only contain markdown files (`MILESTONES.md`, `PROGRESS.md`, etc.), `validate.py`, `action.yaml`, other system files, and the standard subdirectories. No raw files (scripts, data, code, configs) should be directly in the root — move them to the appropriate subdirectory (e.g. `tmp/` for quick tests and throwaway files).
- Is `data/` self-contained? A future agent using a different method must be able to load data without importing from `method/`. If data loaders or synthetic data generators are in `method/`, move them to `data/`.
- **Is `scripts/download.sh` fully self-contained AND `shared/`-free?** It must include a real fetch (HuggingFace, wget, git clone, etc.) for every dataset, model, and upstream resource the code depends on, writing into `data/downloads/`, wrapped in an idempotent guard. Any `shared/` reference in `download.sh` is a portability bug — `shared/` does not exist on a fresh clone. Also confirm: every symlink the agent created from `shared/` into `data/downloads/` has its paired real fetch in `download.sh`.
- **No `shared/` references in committed code.** Walk `method/`, `eval/`, `baseline/`, `scripts/`. Any string literal containing `shared/` is broken on a fresh clone. Code must reference `data/` or `data/downloads/` only.
- **Is `data/downloads/` being used correctly?** `data/` is tracked; `data/downloads/` is the one gitignored subfolder, used only to keep files too large for git out of the tracked repo. It should contain (a) data fetched from public external sources, or (b) agent-generated data that exceeds git's size limit AND is fully reproducible from the agent's own code via `scripts/download.sh`. Agent-authored data that fits in git should live directly under `data/`, not in `data/downloads/`.
- Is `eval/` independent from `method/`? The evaluation code must NOT import anything from `method/`. A from-scratch scientist who builds a completely different method must be able to use `eval/` and `scripts/evaluate.sh` without touching `method/` at all. If `eval/` imports from `method/`, refactor: move shared utilities to `eval/` or `data/`.
- Are `briefing/problem.md` and `briefing/evaluation.md` method-agnostic? They should describe WHAT to solve and HOW success is measured, without describing the paper's approach. Read them carefully — if `evaluation.md` says things like "train an SVM" or "use FedAvg" or names the paper's specific architecture, it is leaking the method. A from-scratch scientist reading these files should NOT know what algorithm the paper used.
- **Are all URLs in `scripts/download.sh` real?** Are they actually the ones we want to download? And are they accessible? Hit each one with `curl -I -L --max-time 15` from the login node and confirm a 2xx (or a 3xx that resolves to 2xx). Any URL that returns DNS failure, 404, or a redirect to a generic landing page is hallucinated and must be replaced with a verified source (HuggingFace mirror, project GitHub release, official archive). A plausible-looking domain is not evidence — only a successful fetch is. **IT IS OFTEN THAT THE AGENTS HALLUCINATE THE SOURCES, IT IS YOUR RESPONSIBILITY TO FIX THEM**
- **Do NOT leave unused files in the folder**.

### 2. Scoring Accuracy

> ⚠️ **Reject any scored wall-clock-speed metric.** Hardware varies across runs (different GPU models), so identical code can clock 2–5× faster or slower with no algorithmic change. Any metric whose value depends on wall-clock time / throughput / runtime is unreliable here. The fix is a machine-independent substitute: FLOPs, parameter count, training-step count, or samples processed.

- Does `scoring/reference.json` exist and pass `python validate.py`?
- Do experiment names in `scoring/scores.json` match those in `scoring/reference.json`?
- Does `scoring/EXPERIMENTS.md` match the experiment names in `scoring/scores.json`, and does it describe each experiment at a high level without leaking method details?
- Are the `coefficient` values in `reference.json` reasonable? Positive for benefits, negative for constraints. Do the magnitudes reflect how much the paper emphasizes each metric relative to others (not absolute value, but think of if the scientist achieves a better score, is the coefficient multiple that relative different an appropriate scale)?
- Are experiment `weight` values reasonable? Does the primary experiment get the most weight?
- Are there constraint metrics where the paper has trade-offs? If the paper claims "X without sacrificing Y," is Y measured?
- **`reference.json` values should match what the paper reports, OR carry a `notes` field explaining a deliberate change.** If a value in `reference.json` is noticeably lower / different from the paper's published number and there's no explanation (e.g., `"notes": "scaled to 10% data; paper reports 0.97 on full"`), that's suspicious — the prior agent may have edited the reference downward to mask a broken reproduction rather than fix it. Spot-check by skimming the paper's results table for the headline numbers and comparing. Flag undocumented drift as a finding; ask the next iteration to either restore the paper's value (and re-run the method) or document the scaling choice properly.

### 3. Gym Scoring Review

Think of yourself as a **judge designing the evaluation rubric** for this gym. Future AI scientist agents will compete on this workspace — you are deciding how their work gets scored.

The gym works as follows: scientists are scored ONLY on experiments that appear in `scores.json`. The values in `scores.json` serve as the baseline. The weights from `reference.json` are renormalized over just those experiments. For example, two AI scientists each propose their own method; we compare the weighted relative changes in scores among them, of what they produced in `scores.json`.

Review with this in mind:

- **Are the renormalized weight proportions reasonable?** Look at which experiments are in `scores.json`, find their weights in `reference.json`, and imagine renormalizing. For example, if scores.json has experiments A (weight 0.05) and B (weight 0.03), the gym effectively scores 62% on A and 38% on B — does that ratio reflect which result matters more for this paper? Adjust weights so the renormalized proportions make sense. Unreproduced experiments can stay in `reference.json` for context.
- **Are the coefficient scales appropriate?** Coefficients control how much a relative change in each metric contributes to the score. Think about how easily each metric can be improved. A metric that normally ranges 50-55% across methods needs a different coefficient than one that ranges 0.01-0.99. If a small, easy improvement in one metric dominates the score while a hard-won gain in the primary metric barely registers, the coefficients are wrong. The scoring should reward meaningful scientific improvement, not gaming the easiest metric. This is not about absolute values, this is about how easy something can be changed relatively. 
- **Does the scoring capture what matters?** Step back and ask: if a scientist genuinely improved the paper's method, would this scoring system detect it? If a scientist gamed an unimportant metric, would they score well? Good scoring makes it hard to cheat and easy to do real science. TREAT YOURSELF AS A JUDGE IN THIS PART.
- **CRITICAL: no scored metric may measure "deviation from the paper's published values."** Walk every metric entry under `reference.json` → `experiments.<exp>.metrics` with a nonzero `coefficient`. Reject any whose name contains `author`, `paper`, `published`, `vs_paper`, `deviation_from_*`, `reproduction_error`, `fit_to_published_*`, or semantically equivalent wording. Also reject any whose formula (readable from the metric's description or from the eval code that computes it) has the shape `|reproduced - paper_value|`, `(value - published)^2`, or similar. The gym is built FROM the paper; the paper is **context**, never an optimization target. A metric whose floor is "exactly match the paper" makes the paper's own method sit at the score ceiling with no room for future improvement — the gym is broken by construction. If you find any such metric, replace it with the underlying scientific quantity itself (accuracy, F1, similarity, ratio, etc.) whose range spans both directions of the paper's claim. This is a **critical** finding, not cosmetic — flag it and fix or downgrade the milestone.
- **The verifier must check the designer's output, not just the reproducer's.** The reproducer produces `scores.json`, `method/`, `baseline/`. The designer produces `scores_train.json`, `scores_test.json`, `evaluate_{{train,test}}.sh`, `eval/train/`, `eval/test/`, AND any new metric entries added to `reference.json`. Designer-introduced metrics are a frequent source of the "author-deviation" class above and the saturation / hardcoded-default classes in section 5. Open `reference.json` and look at which metrics have nonzero `coefficient` — those are what the gym actually ranks on. Verify each was chosen to measure the scientific quantity, not to measure reproduction fidelity.

### 4. Completeness
- Are `scoring/TARGETS.md`, `scoring/EXPERIMENTS.md`, `scoring/CONSTRAINTS.md`, `scoring/DIRECTION.md` filled in and reasonable (not just template comments)? These documents guide future scientist agents — vague or generic language wastes their context window. `EXPERIMENTS.md` must stay method-agnostic and correspond exactly to the experiment structure in `scores.json`. If you see room for improvement, consider refining them.
- Does `scripts/evaluate.sh` work standalone?
- Does `scripts/reproduce.sh` exist and describe the end-to-end flow?
- Is `environment/container.def` + `environment/setup.sh` reasonable?
- Is `MILESTONES.md` filled in with an accurate progress log and stop justification?

### 5. Train/Test Split (only if `eval/train/` or `eval/test/` exists)

Our design creates a hidden held-out test slice for catching agents that game the visible scores. If these artifacts are present, verify them. If neither folder exists, skip this section — the designer hasn't run yet.

- **Eval should read method outputs, not regenerate the experiment.** Open `eval/**/*.py` and ask: *if I deleted everything in `method/`, would these scores change?* Healthy pattern — `eval/` reads artifacts the method wrote (`predictions.csv`, `outputs/*.json`, checkpoints, etc.). Broken pattern — `eval/` calls data-generation routines (`generate_synthetic_data`, `np.random.seed`) and runs the algorithm itself inline; a future agent's edits to `method/` then produce zero change in scores. If you see this, lower the milestone and note it in the Verification Report.
- **All six artifacts must exist when train/test is in play.** `eval/train/`, `eval/test/`, `scoring/scores_train.json`, `scoring/scores_test.json`, `scripts/evaluate_train.sh`, `scripts/evaluate_test.sh`. A partial set means an aborted designer iteration — flag it.
- **Train and test must be independent.** `eval/train/` may not import from `eval/test/` and vice versa. Shared helpers go outside both, not inside either. `validate.py` enforces this — re-confirm it passes.
- **Top-level `eval/` files should not leak test specifics.** A file in `eval/` that defines test-only metrics or splits side-by-side with train ones lets a shortcut agent peek. Soft preference, not a hard rule — fix if obvious.
- **`scripts/evaluate_test.sh` must be zero-effort and lightweight.** A future agent runs it once after `evaluate_train.sh` and gets a score. It must NOT retrain, re-run heavy inference, or change the method. Slice an existing artifact, or at most call inference under a different seed if the cost is small. If you see `bash scripts/method.sh` inside `evaluate_test.sh` doing a full expensive run, that's wrong.
- **Both scripts go through the same method interface as `scripts/evaluate.sh`.** A future agent's method (which may differ from this paper's) should satisfy `evaluate_train.sh` and then `evaluate_test.sh` works automatically. If `evaluate_test.sh` couples to a particular checkpoint format or internal class only this paper produces, it'll break for any other method.
- **`evaluate_*.sh` must enumerate methods, not take a hardcoded name.** Open `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh`. A pattern like `METHOD_NAME="${{3:-some_fixed_name}}"` or any arg that defaults to one baked-in method is broken: in improve mode, the agent's new variant never gets scored on the test side and every run silently reports the paper's method instead of the agent's. The scripts must discover all methods present under the checkpoints/outputs directory (or under `scoring/scores.json`) and iterate over them in one invocation. Flag this as critical if present.
- **Method keys must match between `scores_train.json` and `scores_test.json`.** Diff the sets of method identifiers under `experiments.<exp>.results` (or the equivalent method axis) across the two files. If train scores methods {{A, B, C}} and test scores only {{A}}, the test slice is out of sync — either the designer hardcoded a single name (see previous bullet) or the invocation was wrong. Flag it.
- **Diff `scripts/evaluate_train.sh` against `scripts/evaluate_test.sh`, and spot-check intermediate outputs.** Confirm `evaluate_test.sh` invokes `eval/test/` (not `eval/train/`), writes to `scoring/scores_test.json` (not `scores.json` or `scores_train.json`), and that per-dataset/per-seed files from the test pass carry a `slice: "test"` marker. If both scripts differ only in filename while the evaluator path and output path are identical, the designer copy-pasted without rewiring — flag as critical. Cheap check: grep the intermediate files from one run; `grep -l '"slice": "train"'` should match train-pass files only.
- **Train and test must be calibrated to comparable difficulty.** A well-reproduced method should score similarly on both. A small expected gap is fine; a wildly different range means the split is miscalibrated, not that the method overfit. If `scores_train.json` is 0.9 and `scores_test.json` is 0.2 on the same method, the test slice is too hard or scoped wrong.
- **The test slice must actually discriminate.** Read the split style (different samples, different metrics, held-out subgroup, OOD, ablated input, etc.). Ask: would an agent that gamed `eval/train/` lose points here? If both slices measure the exact same thing on the exact same data, the test slice gives no anti-shortcut signal — flag it.
- **`scoring/reference.json` may have new metric entries from the designer.** Confirm any new metrics inside existing experiments have `higher_is_better`, `coefficient`, and `description`. Existing metric entries and the experiment-level `weight` values must be unchanged from before designer. No new experiments should appear.
- **Score values must come from a real run, not be hardcoded.** Spot-check that `scores_train.json` and `scores_test.json` differ meaningfully where they should (different metrics, different samples) and match where they should (shared metrics on shared samples).

### 6. Evaluation Difficulty
Agents often scale down or substitute datasets to work around resource constraints. This is acceptable when necessary, but it can silently make the problem trivial — destroying the evaluation's ability to distinguish methods. A workspace where every approach achieves near-perfect scores is useless as a benchmark, regardless of how well the code is written. An evaluation reduced from the scale of thousands to tens usually could not show its judgement.

**Critical: Experiment procedure must match the paper.** Scoping down the model size, dataset size, or epoch count is acceptable. Changing the training method is not. If the paper trains from scratch and the agent fine-tunes from pretrained weights (or vice versa), or if the agent swaps the optimization regime (supervised vs self-supervised, RLHF vs SFT, etc.), the experiment cannot demonstrate the paper's claim — it produces numbers that look reasonable but are structurally invalid. Read `briefing/method.md` and the agent's `scripts/reproduce.sh` / training code; if the training protocol differs from the paper, flag it and downgrade the milestone accordingly.
- **Compare reproduced scores against the paper's numbers.** If the reproduction scores are dramatically higher than the paper's, the agent likely used a simplified setup that does not preserve the original difficulty. This is a critical issue.
- **Check whether the agent used the paper's original data or a substitute.** If a substitute, verify it preserves the problem's difficulty. Look at baseline performance — if a simple baseline with no special method already achieves near-perfect results, the substitute is inadequate.
- **Check ablation results for sanity.** If removing the paper's key module has no effect on scores, either the ablation is broken or the evaluation is too easy to detect the difference.
- Do the numbers in `scores.json` look plausible for this domain? **If any reproduced score is more than 10x off from the corresponding value in `reference.json`, flag it as a likely implementation bug — not just an imperfect reproduction.**

---

## What to Do

- **If everything looks good**: update `MILESTONES.md` if needed, commit, and signal done. Do not make changes for the sake of making changes.
- **If you find issues**: fix them. Prioritize scoring accuracy and folder structure — these directly affect whether the gym is usable. You may submit a GPU job if a quick re-evaluation is needed, but avoid long training runs.
- **If the reproduction is completely broken** — the code is broken, the method is fundamentally wrong, results are fabricated, or the workspace is unrecoverable to be used as a gym — set the milestone to `not_possible` and explain why in `VERIFICATION.md`. This tells the system the paper needs a full redo.
- **Do not**: refactor working code unnecessarily, rename files for style, add features that weren't asked for, or re-run experiments that already have valid results.

---

## Verification Report

Before signaling done, write `VERIFICATION.md` in the workspace root. This is your judgment record:

```markdown
# Verification Report

## Initial Assessment
<!-- What state was the workspace in when you started? Was the claimed milestone justified? -->

## Issues Found
<!-- List each issue with severity (critical / minor / cosmetic). If none, say "None." -->

## Fixes Applied
<!-- For each issue, describe what you changed. If nothing, say "No changes needed." -->

## Final Judgment
<!-- After any fixes: is this workspace ready to serve as a gym? Would a scientist agent
     be able to use it meaningfully in both improve and from-scratch modes? -->

- Milestone: {previous_milestone}
- Ready for gym use: yes / no / partial
- Confidence: high / medium / low
- Key limitation: <!-- one sentence, e.g. "synthetic data only" or "none" -->
```

If you need to run experiments, use `action.yaml` as normal job submit.

When finished, signal done in `action.yaml`:

```yaml
action: done
milestone: {previous_milestone}
notes: "<one-line summary of your verification result>"
```

Before writing the final action block, verify that the `**Current:**` header line in `MILESTONES.md` EXACTLY matches the `milestone:` value you are about to write. If you are lowering or raising the milestone as part of this verification, update BOTH files so they agree.
