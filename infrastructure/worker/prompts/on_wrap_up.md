# Wrap Up

It is time to finalize your work.

1. Update `MILESTONES.md` and `PROGRESS.md` with your final status. Fill in the Stop Justification section at `MILESTONES.md`, especially if you did not end in a good state. YOU MUST FILL IN THIS FILE. **Both the Progress Log and Stop Justification sections must contain your real content — leaving the template placeholder text counts as incomplete work.**
2. Fill in `scoring/TARGETS.md` — for each experiment you ran, document: what it tests, the primary metric and direction, and how to run evaluation. Include three kinds of targets: **Primary** (what the paper optimizes), **Constraint** (what the paper must maintain — a scientist who ignores this should score worse), and **Ablation** (what breaks when the key idea is removed). Every target listed here must have a corresponding entry in `scores.json`. See the template in the file. YOU MUST FILL IN THIS FILE.
3. Fill in `scoring/EXPERIMENTS.md` — for each experiment in `scoring/scores.json`, describe at a high level what the experiment is trying to establish. This MUST correspond one-to-one with the experiment names in `scores.json`. Do NOT describe the method or implementation details here. YOU MUST FILL IN THIS FILE.
4. Fill in `scoring/CONSTRAINTS.md` — define what a scientist must hold fixed (model, data, etc.) when trying to improve on your results, so that improvement requires genuine methodology, not resource scaling. Be specific — name exact models, datasets, splits. See the template in the file. YOU MUST FILL IN THIS FILE.
5. Fill in `scoring/DIRECTION.md` — define the research direction a scientist must follow. This is about what problem they must work on and what approach family they must stay within — NOT about resources (that's CONSTRAINTS.md). The scientist should not be able to beat the paper by switching paradigms entirely (e.g. replacing a Mamba model with a Transformer). See the template in the file. YOU MUST FILL IN THIS FILE.
6. Run `python validate.py` and fix any errors it reports.
7. Cleanup the code if needed, but make sure they are still able to run. Removed unused scripts.
8. Finalize `environment/` with whatever you have — even partial reproduction is valuable.
9. Check if you still miss any thing necessary and should be filled in at `CHECKLIST.md`.
10. Again, YOU MUST FILL IN STOP JUSTIFICATION SECTION AT `MILESTONES.md`.
11. Before you write the final action block below, verify that the `**Current:**` header line in `MILESTONES.md` EXACTLY matches the `milestone:` value you are about to write in the action block. These two sources must be consistent. If you are raising or lowering the milestone, update BOTH files.
12. Commit and push everything.

End your response with:

```yaml
---
action: done
milestone: {best_milestone}
notes: "<brief summary of what was achieved>"
```
