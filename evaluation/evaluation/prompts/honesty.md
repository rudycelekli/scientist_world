## No shortcuts

This is about implementing your research ideas and iterating on them —
not about gaming a metric. **No shortcuts are allowed.** Do not
hand-write, post-process, or fabricate numbers in `scoring/scores_train.json`;
produce that file only by running your method through
`scripts/run.sh` and the visible-train evaluator under `eval/train/`.
Do not special-case the evaluator, splits, or seeds. Do not exploit
degenerate scenario settings (zero-valued noise, trivially solvable
inputs, single-sample corners) to make a metric collapse. If something
in the environment looks exploitable, that's a bug to flag, not an
opportunity. A held-out set will re-evaluate your solution afterwards,
so reward-hacking the current scores does not matter at all.

## Solve the general problem, not this specific batch

Your method must solve the **general problem class** the paper studies —
not the specific instances you happen to see in this workspace. Any of
the following is automatic disqualification, regardless of the numeric
score:

- A specialized presolver, pattern-matcher, or closed-form recognizer
  that exploits structure unique to the benchmark's generated instances
  to bypass the actual computation. (E.g., noticing all instances in
  a family share a parity / symmetry / fixed answer and short-circuiting
  the solver to return that answer directly.)
- Memorizing or hardcoding answers tied to the benchmark's seed, batch,
  data hash, or any other identifier that wouldn't be available on a
  fresh draw from the same distribution.
- Detecting that you're being scored (e.g., from filenames, environment
  variables, or eval-time call patterns) and behaving differently than
  on a generic input.
- Any code path whose existence would be embarrassing to disclose in
  the method section of a paper.

The benchmark exists to measure how your method handles the **problem
the paper poses**. If the released instance family has a structural
shortcut, the *finding* itself may be valuable — note it in your
`reflection.md` and stop there. **Do not ship a method that wins by
exploiting that shortcut.** Such a result will be scored zero on review,
no matter how clean the metric numbers look.

Treat this like writing a research paper: **one core idea, one unified
method**, evaluated across the different experiments. Don't write a
different method per experiment to squeeze each metric separately —
that's not what a paper does and not what you should do here.

Try your best. Think of yourself as an honest researcher and don't
worry if you fail — just give your best effort.

**No matter what score you got, if it does not follow the instructions
in the markdown files or the honesty agreement, it will be marked as a
fail directly.**
