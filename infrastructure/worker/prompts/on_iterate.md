# You are Continuing a Reproduction

You are picking up this workspace from a previous agent. The workspace already contains code, scripts, documentation, and possibly results, but we are not sure about the quality and completeness. **Your milestone starts at `{starting_milestone}` — we reset it one level below the previous agent's claimed `{previous_milestone}`.**

**Your job is to push the milestone as high as possible.** Start by auditing what exists, then either restore the milestone if the work is solid, or fix the issues and continue reproduction work. As before, **Your minimum goal is `core_claim`. Do not stop before reaching it unless reproduction is genuinely impossible.**

⚠️ **If auditing found issue, YOU MUST FIX IT.** If you audit and found issue, but you only document them, and stop, you have failed. You must fix the issues you find and continue the reproduction.

⚠️ **Try the existing workspace first. Do not rewrite working files (especially `environment/container.def` and `environment/setup.sh`) without a concrete, verified reason — the previous agent's setup may already work.**

---

## Audit Checklist

Before writing any new code, review the workspace and answer these questions:

### 1. Is the implementation real and complete?
Read the code in `method/`, `eval/`, and other folders and files. Does it implement the paper's actual algorithm, or is it a surrogate? Specifically:
- Is the implementation complete or, from your judgement, reasonable?
- Could someone reading the code identify which algorithm from the paper was implemented?
- Are the paper's computational steps present, or is it a generic model trained to approximate outputs?
- If the implementation is a surrogate, move on directly to fix it before checking others.

### 2. Is the evaluation honest?
- Does `eval/` evaluate independently from `method/`, or does it train its own model?
- Are the numbers in `scoring/scores.json` real (from running the code), or hardcoded/copied from the paper?
- Are the reproduced scores numerically plausible compared to `scoring/reference.json`? If any result is more than 10x off from the reference, the implementation is likely broken — investigate and fix before proceeding.
- Does `scripts/evaluate.sh` work standalone?

### 3. Are the deliverables complete?
- `briefing/` files filled in? Are `problem.md` and `evaluation.md` method-agnostic?
- `scoring/reference.json` validated?
- `scoring/TARGETS.md`, `scoring/CONSTRAINTS.md`, `scoring/EXPERIMENTS.md`, `scoring/DIRECTION.md` filled in?
- `scripts/` (evaluate.sh, reproduce.sh, download.sh) functional?
- `environment/` (container.def, setup.sh) reasonable?

---

## After the Audit

Based on your findings, either:
1. If the reproduction is intact and trustworthy, **restore the milestone to `{previous_milestone}`** and mark complete. This is a valid outcome — not every workspace needs more work.
2. If there are issues, **fix them and continue working** — do not just document problems and stop. Submit jobs, get results, and push milestones higher.
3. Even if the work is solid at `{previous_milestone}`, if you see clear room to push further (more experiments, additional claims), continue working to reach a higher milestone.
