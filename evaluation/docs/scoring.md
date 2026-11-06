# Scoring: Frontier-Normalized Improvement (FNI)

This document describes how a run is turned into a score that can be
averaged across papers. It follows the code in
`evaluation/normalization.py`; constant names in `code font` are the names
used there.

## What is scored

- A **paper** is one built environment. Its repository contains the paper's
  own reproduced results, `source/scoring/scores_train.json` and
  `source/scoring/scores_test.json` (the *source* scores), and
  `source/scoring/reference.json`.
- A **run** is one agent (a *profile*) working on one paper in one **mode**
  (`improve` or `innovate`). The two modes are accounted separately
  everywhere: they have separate coefficient files and separate
  leaderboards.
- A **phase** is one scoring split: `train` (the evaluator the agent can
  run) or `test` (the held-out evaluator the agent never sees). Each phase
  has its own coefficients.
- A **cell** is one number: (experiment, method, metric). The cells of a
  paper and phase are the numeric values in the source score file for every
  method that `reference.json` does not tag `"type": "baseline"`. A method
  that `reference.json` does not list counts as a scored (proposed) method.

`reference.json` is read for four things only: the per-experiment `weight`,
the per-metric `coefficient` and `higher_is_better`, and the per-method
`type`. It is never a source of compared values. The anchor a run is
compared with is always the source score file.

Score files may be wrapped (`{"experiments": {exp: {"results": {method:
{metric: value}}}}}`) or flat (`{exp: {method: {metric: value}}}`); both
are accepted, for source and run files alike.

## Raw improvement

For one cell and one run:

```
improvement = (run_value - source_value)   if higher_is_better
            = (source_value - run_value)   otherwise
```

`higher_is_better` defaults to true when the metric has no entry. The
improvement is signed, is not divided by the source value, and is not
clamped. A source value of 0 needs no special case.

Dividing by `|source|` would not change anything downstream: the per-cell
spread defined below would be divided by the same constant, and every
quantity uses the ratio `improvement / spread`.

## The cohort

Coefficients are computed from a **cohort** of profiles: `profile_cohort`
in the config when it is set, otherwise every profile under `profiles`; in
both cases minus `profile_cohort_exclude`.

A cohort profile contributes to a (paper, mode, phase) only if its run
directory `<profile>-<mode>-eval` has a clean verdict in
`scoring/verification.json`: `verified` (the grader re-ran the frozen
method and re-graded it) or `audit_clean`. The harness itself never writes
`audit_clean`; the status is accepted so that a run that was inspected by
hand can be admitted. For a `verified` run the scores are the grader's
(`scores_<phase>` inside `verification.json`); for `audit_clean` they are
the run's own `scoring/scores_<phase>.json`. A run with any other verdict,
or with none, contributes nothing.

Profiles that share a `publish_as` value are one cohort member. If the
profile whose name equals that value has a clean run it provides the
scores; otherwise the last such profile in cohort order does.

## Coefficients

For one (paper, mode, phase), with at least `MIN_COHORT_PRESENT` = 3 cohort
members present (otherwise see Exclusions):

1. **Prior weight** of each cell:
   `prior = |metric coefficient| x experiment weight`. Missing values count
   as 1.0. Only the magnitude of the metric coefficient is used; the
   direction comes from `higher_is_better`.
2. **Spread** of each cell: the sample standard deviation of the cohort's
   improvements on that cell (0 when only one cohort member reports it).
   Cells that no cohort member reports are dropped.
3. **`typical_spread`**: the median of the non-zero cell spreads of the
   paper.
4. **Normalizer** of each cell: its own spread if that is positive, else
   `typical_spread`.
5. **Effective weight**:
   `raw_weight = prior x (0.5 + spread / (spread + typical_spread))`, then
   divided by the sum of the raw weights so that the effective weights sum
   to 1. The multiplier is 0.5 for a cell on which the cohort is tied, 1.0
   for a cell with the typical spread, and approaches 1.5 for a cell with a
   very large spread.
6. **Raw paper score** of each cohort member:
   `paper_score_raw = sum over its cells of effective_weight x improvement / normalizer`.
7. **`paper_baseline`**: the mean of `|paper_score_raw|` over the cohort
   members.
8. **Stored coefficient** of each cell:
   `coefficient = effective_weight / (normalizer x paper_baseline)`.

Why each piece is there:

- Dividing by the normalizer puts metrics with different units and ranges
  on one scale, so that a metric measured in hundreds does not swamp one
  measured in hundredths.
- The effective weight lets cells on which the cohort models differ count
  more than cells on which they all agree, without ever removing a cell
  and without letting one cell take an unbounded share.
- `paper_baseline` gives every paper the same magnitude: by construction
  the cohort's mean absolute paper score is 1 on every paper, so a paper
  with naturally large improvements does not dominate an average over
  papers.

## Scoring a run

Any run, in the cohort or not, is scored with the stored coefficients
(`normalize_delta`):

```
paper_score = sum over cells of coefficient x improvement
FNI         = 100 x paper_score
```

- 0 means no better than the paper's own reproduced results; negative
  means worse.
- 1 (FNI 100) is the typical magnitude of a cohort model on that paper.
- Cells the run does not report are skipped, not penalised. The result
  records `n_cells_used`, `n_cells_missing` and `n_cells_pruned`.
- Pruned cells (below) are skipped for every run.
- If the run reports none of the cells, or the paper has no coefficients,
  there is no score; the result carries a `reason` instead (`no_score`,
  `no_coeffs`, `no_cells`, `no_cells_used`, or the exclusion reason).

**Method names.** The run's value for a cell is looked up under the
paper's method name. If the run has no row with that name, and its score
file has exactly one other row in that experiment that carries the metric
and is not the name of another scored method of that experiment, that row
is used. This lets an `innovate` run file its results under the name of its
own method. With zero or several candidate rows the cell counts as
missing.

**Scale.** Coefficient files, the dashboard's JSON API
(`train_norm.delta_norm`, `/api/leaderboard`) and the dataset columns
(`train_score`, `test_score`) hold the unscaled `paper_score`. The factor
100 is applied for display only: the per-paper matrix, the run detail view
and the charts of the dashboard show FNI, and so do the dataset columns
`fni`, `fni_train` and `fni_test`. The leaderboard table of the
dashboard prints the unscaled mean.

## From paper scores to a leaderboard

The dashboard aggregates as follows (`_aggregate_leaderboard`):

- Only runs whose status is `completed` and that have both a train and a
  test score file are counted. A run that is complete but has no held-out
  score yet is listed as `aborted`.
- A run's **final** score is the mean of its train and test paper scores
  (or the one that exists, when the paper has coefficients for one phase
  only).
- A model's leaderboard entry is the mean, over its counted runs in one
  mode, of the final scores. Runs on papers without coefficients are left
  out for every model.

The leaderboard does not filter on the grader's verdict: every completed
run with both score files is scored against the coefficients, whatever its
verification status. The verdict decides which runs *build* the
coefficients, not which runs are *shown*. Restrict a reported table to
`verified` runs (the dashboard lists the status of every run; the dataset
holds verified runs only, scored on what the grader measured).

## Exclusions

When coefficients cannot be computed the coefficient file is still written,
with `"excluded": true` and a `reason`, and every consumer shows no score
for that paper, mode and phase:

| reason | meaning |
|---|---|
| `cohort_too_small` | fewer than 3 cohort members with a clean verdict (`min_required` is recorded). With two members the spread is `|a - b| / sqrt(2)`, and a near-tie would produce a huge coefficient. |
| `no_cells` | no cohort member reports any scored cell |
| `all_cells_tied` | the cohort has the same value on every cell, so there is no spread to normalize by |
| `all_priors_zero` | every cell has a prior weight of 0 |
| `paper_baseline_zero` | every cohort member's raw paper score is exactly 0 |

Coefficients are recomputed whenever a run of the paper ends and whenever
a grading ends with `verified`, so a paper leaves the excluded state as
soon as the condition no longer holds.

## Dominant-cell pruning

A coefficient is `effective_weight / (spread x paper_baseline)`. On a
saturated metric the cohort's spread can be tiny, which makes the
coefficient very large; a model that misses that metric by a small amount
then gets a score orders of magnitude outside the normal range.

After the coefficients of a (paper, mode, phase) are computed, each cell's
contribution `coefficient x improvement` is computed for every run of that
paper and mode that has a clean verdict, cohort member or not. A cell is
pruned if, for any such run, its contribution both

- is larger in magnitude than all the other cells of that run together, and
- exceeds `PRUNE_DOMINANCE_SCORE` = 1.0 in magnitude (more than one whole
  typical paper score).

A run that reports a single cell is not considered. The check is repeated on the remaining cells until no cell qualifies, and
stops after `MAX_PRUNED_CELLS` = 5 cells. Pruned cells are listed under
`pruned_cells` in the coefficient file and are skipped for every run, so
all runs are scored on the same reduced set of cells. The coefficients of
the remaining cells are not recomputed.

## Locking

After each recomputation (`update_for_paper`), a mode of a paper is
**locked** when its test-phase coefficient file is not excluded and has at
least `AUTO_LOCK_MIN_MODELS` = 4 cohort members. Locking sets `locked`,
`locked_at` and `locked_n_models` in both the train and the test file of
that mode. A locked mode is never recomputed: later runs, including later
cohort runs, are scored against the frozen coefficients, so a published
score does not move.

Consequences:

- A mode whose paper has no source test scores never locks.
- Until a mode is locked, the scores of earlier runs change when a further
  cohort run is graded.
- The `renormalize` command recomputes unlocked modes and leaves locked
  ones untouched; it does not lock. To recompute a locked mode, delete its
  two coefficient files first.
- When publishing is enabled, newly locked coefficient files are pushed to
  the `normalization` branch of the paper's repository.

## Files

`<data_dir>/envs/<paper>/normalization/coeffs_<mode>_<phase>.json`:

| key | content |
|---|---|
| `paper_id`, `mode`, `phase` | what the file is for |
| `cohort` | the configured cohort (profile names) at computation time |
| `models_present`, `n_models` | cohort members that contributed |
| `n_cells` | number of scored cells of the paper |
| `excluded`, `reason`, `min_required` | see Exclusions |
| `typical_spread`, `paper_baseline` | per-paper quantities |
| `cohort_raw` | `paper_score_raw` of each cohort member |
| `cells` | one entry per cell: `experiment`, `method`, `metric`, `source`, `higher_is_better`, `coefficient`, and the diagnostics `prior_weight`, `spread`, `normalizer`, `effective_weight`, `cohort_imp` |
| `pruned_cells` | cells skipped for every run |
| `locked`, `locked_at`, `locked_n_models` | present once locked |
| `last_updated` | time of the computation |

To score a run, only the cell entries (`experiment`, `method`, `metric`,
`source`, `higher_is_better`, `coefficient`) and `pruned_cells` are needed.

`<data_dir>/normalization/global_sd_<mode>.json` is a per-mode summary
(number of papers with coefficients, number excluded, median
`paper_baseline`). Nothing reads it for scoring.

## Worked example

One experiment with weight 2.0. The proposed method `ours` has two metrics:
`acc` (higher is better, metric coefficient 1.0) and `loss` (lower is
better, metric coefficient 0.5). Three cohort models were graded.

| cell | source | prior | m1 | m2 | m3 | improvements | spread |
|---|---|---|---|---|---|---|---|
| acc | 0.70 | 2.0 | 0.80 | 0.70 | 0.60 | +0.1, 0.0, -0.1 | 0.1 |
| loss | 2.0 | 1.0 | 1.5 | 2.5 | 2.0 | +0.5, -0.5, 0.0 | 0.5 |

```
typical_spread    = median(0.1, 0.5)            = 0.3
multiplier        = 0.5 + 0.1/(0.1+0.3) = 0.75    0.5 + 0.5/(0.5+0.3) = 1.125
raw weight        = 2.0 x 0.75  = 1.5             1.0 x 1.125 = 1.125
effective weight  = 1.5/2.625   = 4/7             1.125/2.625 = 3/7
paper_score_raw   = m1: 4/7 x 1 + 3/7 x 1 = 1     m2: -3/7     m3: -4/7
paper_baseline    = (1 + 3/7 + 4/7) / 3         = 2/3
coefficient       = (4/7)/(0.1 x 2/3) = 60/7      (3/7)/(0.5 x 2/3) = 9/7
paper_score       = m1: 60/7 x 0.1 + 9/7 x 0.5 = 1.5
                    m2: 9/7 x (-0.5)            = -0.643
                    m3: 60/7 x (-0.1)           = -0.857
mean |paper_score| over the cohort              = 1
FNI               = m1: 150    m2: -64    m3: -86
```

A fourth model that reaches `acc` 0.75 and `loss` 2.0 scores
`60/7 x 0.05 = 0.43` (FNI 43) with these coefficients.

`tests/test_normalization.py` checks this example and the rules above.

## Limitations

- **Small cohorts.** The spread is a sample standard deviation over three
  or four values. One unusual cohort run changes a cell's spread, and
  through the effective weights the weight of every other cell.
- **Near-cancelling baseline.** `paper_baseline` is a mean of absolute
  values of sums. If the cohort's raw paper scores are all close to 0
  because cells cancel, the baseline is small and later runs get large
  scores. Only an exact 0 is excluded.
- **Pruning does not re-normalize.** After a cell is pruned, the remaining
  coefficients keep the `paper_baseline` computed with that cell, so the
  cohort's mean absolute score on that paper is no longer exactly 1.
- **Missing cells.** A run that reports fewer cells is scored on fewer
  cells. Train validation makes a `completed` run report every train cell.
  Nothing in the scoring enforces completeness of the test cells; that is
  part of the grader's task.
- **Method-name fallback and baselines.** The fallback excludes the names
  of the paper's other scored methods, not the names of its baseline
  methods. A score file whose only row for an experiment is a baseline row
  is therefore read as the run's own result for that experiment.
- **Zero weights.** An experiment `weight` of 0 in `reference.json` is
  read as 1.0, while a metric `coefficient` of 0 gives the cell a prior
  weight of 0 (the cell stays in the file with a coefficient of 0).
- **Cohort changes.** Changing `profiles`, `profile_cohort` or
  `profile_cohort_exclude` changes the coefficients of every unlocked mode
  at its next recomputation. Locked modes keep the coefficients of the
  cohort that was present when they locked; the file records it.
- **Two views of a score.** The dashboard also shows a raw percentage next
  to each run (a weighted mean of per-cell relative improvements, each
  clamped to +/-100%, with exact method names only). It is a quick
  comparison with the source scores and is not the FNI.
