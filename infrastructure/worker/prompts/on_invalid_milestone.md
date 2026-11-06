# Invalid Milestone

You signed off with `milestone: {bad_milestone}` in `action.yaml`, but that is not a valid milestone name (attempt {attempt}/{max_attempts}).

Valid milestones, in order:

- `none`
- `method_runs` — the paper's algorithm runs end-to-end without errors
- `core_claim` — minimum experiment supports the paper's central claim
- `core_claim_plus` — central claim plus one additional supporting result
- `secondary_claims` — secondary claims reproduced
- `majority` — majority of the paper's reported experiments reproduced
- `near_complete` — almost all experiments reproduced
- `full` — every reported experiment reproduced
- `not_possible` — the paper genuinely cannot be reproduced in this environment

Pick the one that honestly matches what you achieved. Update the `milestone:` line in `action.yaml` and sign off with `action: done` again.
