# Reproduction Checklist

Check off items as you complete them. Order doesn't matter — work however makes sense for this paper.

## Briefing
- [ ] `briefing/problem.md` (method-agnostic)
- [ ] `briefing/evaluation.md` (method-agnostic)
- [ ] `briefing/method.md`
- [ ] `briefing/overview.md`

## Scoring
- [ ] `scoring/reference.json` — paper's reported numbers
- [ ] Workspace validated via `python validate.py`
- [ ] `scoring/scores.json` — reproduced numbers (must match reference.json experiment/metric structure)
- [ ] `scoring/EXPERIMENTS.md` — high-level purpose of each experiment in `scores.json` (do not edit until wrap up)
- [ ] `scoring/TARGETS.md` — primary, constraint, and ablation targets (do not edit until wrap up)
- [ ] `scoring/CONSTRAINTS.md` — what a scientist must hold fixed (do not edit until wrap up)
- [ ] `scoring/DIRECTION.md` — research direction scope for scientist (do not edit until wrap up)

## Code
- [ ] Evaluation code in `eval/`
- [ ] Method implementation in `method/`
- [ ] Baseline implementation in `baseline/` (if applicable)

## Scripts
- [ ] `scripts/evaluate.sh`
- [ ] `scripts/reproduce.sh`
- [ ] `scripts/download.sh` (idempotent, size comment at top)
- [ ] `scripts/baseline.sh` (if applicable)

## Environment
- [ ] `environment/container.def` + `environment/setup.sh`

## Data
- [ ] Dataset acquired (shared dir or `data/`)
- [ ] Pretrained models downloaded if needed

## Milestones
- [ ] `method_runs` — executes end-to-end without errors
- [ ] `core_claim` — minimum experiment supports central claim
- [ ] `core_claim_plus` — additional settings
- [ ] `secondary_claims`
- [ ] `majority`
- [ ] `near_complete`
- [ ] `full`
