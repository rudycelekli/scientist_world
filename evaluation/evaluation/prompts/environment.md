## Environment (two containers, one shared workspace)

Your turn runs in the **agent container** — small, login-node, 8 GB
RAM, 2 BLAS threads. When you write `action: submit`, the pipeline
dispatches the job to the **paper compute container**, which is
separate and runs on a GPU node. Both containers bind-mount the same
workspace at `/home/user`, so files you write survive across the
boundary in either direction.

### Two files you control

```
environment/
├── container.def   # paper's base image + apt packages + baked-in deps
└── setup.sh        # OPTIONAL — extra pip installs that land in a
                    #            persistent overlay on top of container.sif
```

**`container.def`** defines the paper's compute image: base OS, CUDA
toolkit, apt packages, anything that has to be baked into the SIF.
Changing it triggers a full `apptainer build` — expensive, minutes.

**`setup.sh`** is for Python-level deps the agent adds on top of the
paper image. It starts **absent** — create it only when you need
something the paper image doesn't already provide. Changing it only
rebuilds the overlay — cheap, seconds to a minute.

### How to add a pip package

Almost always, do this in `setup.sh`:

```bash
# environment/setup.sh
#!/bin/bash
set -euxo pipefail
uv pip install --system faiss-cpu==1.8.0 lightgbm
```

On the next `action: submit`, the pipeline:

1. Hashes `container.def` and `setup.sh`.
2. `container.def` unchanged → keeps the existing SIF.
3. `setup.sh` changed → `truncate + mkfs.ext3 + apptainer exec
   --fakeroot --overlay <img> bash setup.sh` — new packages land in
   a fresh overlay image.
4. Points the Nexus project at the new overlay.
5. Submits the job. Jobs mount the overlay read-only, so installs
   persist across every subsequent submit.

### When to touch `container.def`

Only when `setup.sh` can't solve it:

- You need a new base image (different CUDA version, different OS).
- You need apt packages not in the image (build tools, system libs).
- You need something in `%files` bind-copied at build time.

Edit the `%post` section for apt packages, or the `Bootstrap:` /
`From:` header for a base image swap. The full SIF rebuilds on the
next submit — slower, but cached by `sha256(container.def)` so
identical edits are free.

### Do not `pip install` inside job scripts

`scripts/run.sh` and friends run on GPU nodes with a **read-only**
overlay and an ephemeral writable tmpfs. Any `pip install` there
disappears before the next submit. Put every dep in `setup.sh` (or
`container.def`) so subsequent jobs see it.

### Build failures are fixable

If `apptainer build` or `setup.sh` exits non-zero, the job is NOT
submitted — you get another turn with the error log pointer. Fix the
root cause (missing apt package, bad pin, wrong base image) in the
appropriate file and submit again. Do not mark `done` because of a
build failure.

### Downloads + large data

Datasets, model weights, and anything >a few MB go in
`scripts/download.sh`, not `setup.sh` or `container.def`.
`download.sh` runs in the full workspace with internet access and
writes into `data/downloads/`. Wrap every fetch with an idempotent
guard like `[ -e data/downloads/X ] || fetch_X` so re-running is safe.

### Summary

- New pip dep → write/append `setup.sh`, submit. Overlay rebuilds.
- New apt / base-image / CUDA → edit `container.def`, submit. SIF rebuilds.
- Data / models → `scripts/download.sh`.
- Nothing rebuilds unless a file's hash actually changed.
