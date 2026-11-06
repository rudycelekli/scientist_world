# Push Forward

As long as you follow the instructions, there is nothing really limit you.

You signalled done at milestone `{current_milestone}`, but the minimum target is `{target_milestone}`. You must keep working.

`{target_milestone}` does not require reproducing the paper's full results or exact numbers. It requires a minimum experiment that supports the paper's central claim — run the actual method, get real results, show the method works as claimed.

If your current approach hit a wall (model too large, data unavailable, results don't match), adapt:
- **The data pipeline is editable.** `scripts/download.sh`, `data/`, and any split/subset file are yours to change. If your evaluation was run on a subset smaller than the paper's (e.g. a smoke-test split you inherited), rewrite the download to pull the paper's reported scale and re-run. Inconclusive results from an under-powered evaluation is NOT a valid stopping condition.
- **Question any "fundamental" limit before accepting it.** If you're about to call a constraint fundamental, inherent, or impossible to fix — first verify it. Is the limit really in the paper/method, or is it in the workspace's current setup (which you can change)? Workspace state is a starting point, not a fixed input.
- Scale down the experiment if the original is infeasible — smaller model, fewer samples, subset of benchmarks. But again, follow the rules, scaling down CANNOT make the results trivial.
- Download the required data if you haven't tried yet, remember to download by yourself directly, not in compute nodes.
- Try different hyperparameters or configurations if initial results are poor.
- Try a different experiment setting from the paper where the claim might be easier to demonstrate.

Do not signal done again until you have either reached `{target_milestone}` or genuinely exhausted all options. This is attempt {attempt} of {max_attempts}.

**⚠️ Do NOT inflate your milestone claim.** If the method does not actually work and follow the paper's claims, do not claim `core_claim`. Claiming a milestone you haven't reached wastes everyone's time — the system will accept your claim and move on, leaving a broken workspace behind. Be honest about where you are.
