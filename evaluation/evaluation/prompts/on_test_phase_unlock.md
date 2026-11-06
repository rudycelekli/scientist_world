## Test phase, final tier — `method/` is now writable

The held-out test scorer has not been able to run cleanly through the previous fix turns with `method/` locked. To get past whatever shape mismatch is blocking it, `method/` is now writable for up to 3 more attempts.

What you can do now:

- Rename or move files inside `method/` so the test scorer's imports resolve
- Add small adapter / wrapper modules that expose your existing method under class or function names the scorer expects
- Restructure your `method/` layout so the scorer can find and call your code
- **Rename / move / symlink anything else in the workspace that the scorer expects at specific paths or names** — e.g. directories under `checkpoints/`, output filenames, prediction folders. If your method already produced real outputs and the scorer just can't find them because of a path or name mismatch, fixing that mismatch is the correct thing to do here. It is **not** fabrication — the data is yours, the score will be computed on your real outputs.

Concretely, if the scorer fails with "missing prediction directories X, Y, Z" or "cannot find file F" and you already have the equivalent content under different names, **rename / symlink** them to what the scorer wants and resubmit. That is exactly what this tier exists for.

What is still locked (do not try to bypass):

- `eval/test/` — the held-out scorer code
- `scripts/evaluate_test.sh` — the scorer entrypoint
- `scoring/scores_train.json` — your main-phase result

Everything else in the workspace is writable, including `checkpoints/`, `tmp/`, and `method/` (newly unlocked this tier).

### On the score — please read this carefully

There is no pressure on the number.

This benchmark is not ranking you against the paper. It is not evaluating whether your method is good or bad. We are simply recording what your method actually produces when scored. The numbers themselves do not matter — what matters is that they are honest.

A real low score is useful data. A fabricated high score, or a score produced by stubs, is noise that pollutes the dataset.

So:

- **Do not** create empty stub files just to make a file-existence or hash check pass when you have no real content to put there. (Renaming a directory with real outputs to satisfy a path check is fine — that's not a stub.)
- **Do not** rewrite the algorithm in `method/` to chase a particular result, or to guess what the paper's method was doing. This tier is for making your existing method *callable* by the scorer, not for changing what it computes.
- **Do not** hand-write `scoring/scores_test.json`. The scorer must produce it from your real outputs.
- If, after honestly trying renames / symlinks / adapters, the scorer's expected interface still cannot be satisfied because the required content does not exist in your run, write a brief note in `notes.md` explaining what is missing and stop. The run will be recorded as `incomplete`. But please don't stop just because you're worried renaming feels like "cheating" — if the data is real, renaming the path to where the scorer looks is the legitimate fix.

That is the contract for this tier. Submit a fix and the next test job will run.
