## Resources

You have **{gpu_budget:.1f} GPU-hours** total for this run and can
request up to **{max_gpus} GPUs** per compute job. Every `action:
submit` consumes from this budget. Plan accordingly — once exhausted,
no more jobs are accepted.

## Where Your Code Runs

Your container runs on a shared login node. It is capped at **8 GB
RAM**, **2 BLAS threads** (`OMP_NUM_THREADS=MKL=OPENBLAS=2`), and
`nice 19` scheduler priority. These caps are real cgroup / env-var
enforcement — anything heavier gets OOM-killed or throttled to a
crawl.

The login node is **only** for:

- Reading files and logs
- Editing source code
- Light inspection (`head`, `wc`, small `pandas` peeks at a few rows)
- Downloading data / installing packages (wrapped in job scripts)
- Writing `action.yaml`

**NEVER do any of the following on the login node:**

- Training, fine-tuning, or inference of any model
- Running experiments, evaluations, or simulations
- Large data processing (tokenization, embedding, feature extraction)
- Anything involving `torch.cuda`, large matrix ops, or multi-process
  CPU work
- Benchmark runs, even "quick" ones

**All real compute MUST go through `action: submit`.** That routes
the job to a GPU compute node with full memory and CPU. The login
node is shared with many users; heavy work there slows everyone down
and admins will kill your processes — costing you the turn with no
results.

If a script looks like it might do non-trivial work, submit it. If
you are unsure, submit it. "I'll just try it quickly to see if it
works" is the wrong instinct — the container will kill it before you
learn anything useful.
