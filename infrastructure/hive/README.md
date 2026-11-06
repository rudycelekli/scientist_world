# Hive: interactive compute pool

Hive keeps a pool of SLURM nodes joined into one [Ray](https://www.ray.io/) cluster
and runs shell commands on it as jobs. An agent can run a short debugging command,
read the result, edit the code and launch a longer job without paying the cost of
acquiring an allocation for every step.

In Scientist World, Hive is the bottom layer of the system: [Nexus](../nexus) accounts
each job against a per-project budget and forwards it to Hive, and
[Swarm](../swarm) and [Evaluation](../../evaluation) reach compute only through Nexus.
Hive itself knows nothing about papers or agents and can be used on its own.

```
 login / head node                         compute nodes (one SLURM job each)
┌──────────────────────────┐              ┌──────────┐ ┌──────────┐ ┌──────────┐
│ hive CLI                 │              │ Ray      │ │ Ray      │ │ Ray      │
│ Ray head                 │◄────────────►│ worker   │ │ worker   │ │ worker   │
│ dashboard + REST API     │   Ray        │ (gpu)    │ │ (gpu)    │ │ (cpu)    │
└──────────────────────────┘              └──────────┘ └──────────┘ └──────────┘
        ▲  HTTP
        │
   hive connect / hive.client / Nexus
```

## How it works

1. **SLURM pool.** `hive start` submits one single-node SLURM job per requested
   worker. Small jobs queue faster than one large allocation, and the pool grows or
   shrinks one node at a time (`hive add`, `hive remove`).
2. **Ray cluster.** The Ray head runs on the node where you ran `hive start`. Each
   worker job joins it with `ray start` and tags itself with its resource name. A
   worker that loses the head reconnects on its own; a worker that reaches its
   walltime drains its jobs first and then resubmits itself.
3. **Job dispatch.** Every job gets a `STRICT_PACK` placement group, so all of its
   GPUs are on one node. A job runner (a detached Ray actor) streams stdout and stderr
   to the job record and writes the final status. Normal-priority jobs may preempt
   `low`-priority jobs; a job that is interrupted because its node died or drained is
   put back in the queue and restarted from the beginning.
4. **Interfaces.** The `hive` CLI, the dashboard's REST API, and the dependency-free
   Python client `hive.client.Client` all drive the same backend.

## Requirements

- Linux and a SLURM cluster (`sbatch`, `scancel`, `squeue`, `sacct`).
- A head node that may run long-lived processes and is reachable from the compute nodes.
- A filesystem shared by the head node and the compute nodes that holds this
  checkout, its virtualenv, Hive's state directory and the job working directories.
- GPUs requested as `--gres=gpu:N`; `sbatch` callable from compute nodes (workers
  resubmit themselves when they end).
- Python >= 3.10.

## Install

```bash
cd infrastructure/hive
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add ".[dev]" for pytest
cp hive/config/config.example.yaml hive/config/config.yaml
```

Hive must be installed in editable mode: its default state directory and config file
live next to the source tree. Set `HIVE_DIR` to keep state elsewhere (it must be on
the shared filesystem).

## Configuration

`hive/config/config.yaml` (or `$HIVE_DIR/config.yaml`, which takes precedence).
[`config.example.yaml`](hive/config/config.example.yaml) documents every key.

| Key | Meaning |
|---|---|
| `slurm.resources[]` | One entry per kind of node. `name` is the label used by `-r`; `gpus_per_job` is the GPU count of one node (0 for CPU-only); `max_jobs` is used by `--full`; `test: true` marks a short validation queue. Optional sbatch settings: `partition`, `account`, `qos`, `constraint`, `mem`, `cpus_per_task`, `walltime`, `exclude_nodes`. |
| `slurm.default_walltime` | Walltime of worker jobs when the resource sets none. |
| `slurm.drain_before_timeout` | How long before walltime a worker starts draining its jobs. |
| `slurm.extra_args` | Extra `sbatch` flags for every worker job. |
| `ray.head_port` | Ray head port. Choose a free one on a shared head node. |
| `ray.dashboard_port`, `ray.dashboard_host` | Port and bind address of the dashboard and REST API (default `127.0.0.1:38581`). |
| `ray.head_ip` | Address workers use to reach the head; auto-detected when unset. |
| `ray.launch_pending_grace_sec` | How long a freshly launched job runner is trusted before the launch is retried. |
| `env.venv_path` | Virtualenv activated on compute nodes; it must contain `hive` and `ray`. |
| `env.working_dir` | Default working directory of jobs. Default: the directory you submit from; for a REST call that names no `workdir`, the dashboard's own directory. |
| `env.ray_tmpdir` | Ray temp directory (short path, on storage that is not cleaned periodically). `HIVE_RAY_TMPDIR` overrides it. |
| `env.setup_commands` | Shell lines run on each worker before Ray starts (module loads, proxies). |

## Usage

```bash
# Start the pool: 3 nodes of resource "gpu", 1 node of "cpu"
hive start -r gpu 3 -r cpu 1            # add --wait to block until a worker joins
hive start --full                       # every non-test resource at its max_jobs
hive status                             # nodes, GPUs, free GPUs

# Grow or shrink a running pool
hive add -r gpu 2
hive remove -r gpu 1

# Run jobs (asynchronous by default; prints a job id)
hive submit "python train.py" --gpus 2 --workdir /path/to/project
hive submit "python prep.py" --gpus 0 --cpus 8 -r cpu
hive submit "python sweep.py" --priority low            # yields to normal jobs, restarts after preemption
hive submit "python eval.py" --timeout-sec 3600 --max-retries 2
hive submit "nvidia-smi" --wait                         # block and print the output
hive submit "python train.py" -r gpu,gpu-large          # first resource with room wins

hive jobs [--status running|queued|completed|failed]
hive queue
hive logs <job_id>
hive cancel <job_id> [<job_id> ...]

# Maintenance
hive serve [--port P] [--host H] [--foreground] [--stop]   # dashboard only
hive restart        # restart the Ray head and dashboard; SLURM workers reconnect
hive relaunch       # rebuild local control-plane state around the tracked SLURM jobs
hive clear [--all]  # delete finished job records and logs
hive stop           # cancel all worker jobs, stop Ray, delete job logs (asks first)
```

Notes on behaviour:

- A job never spans nodes. The largest job is one node's GPU count.
- `--timeout-sec` bounds the job itself; `--timeout` only bounds how long
  `--wait` waits.
- The job environment is not the Hive virtualenv: Hive removes its own virtualenv
  from `PATH` before running your command, so `python` is whatever the worker
  provides (activate your own environment inside the command).
- Hive holds its SLURM allocations until you remove them, so size the pool to
  your workload and use `hive remove` or shorter walltimes when demand drops.
- Restarting the dashboard interrupts running job runners; the affected jobs are
  requeued.

### Job records

Everything about a job is under `$HIVE_DIR/jobs/<job_id>/`: `meta.json` (status,
timing, resources, return code), `stdout.log` and `stderr.log` (streamed while the job
runs). Files the command writes stay in its working directory.

### Remote mode

Any machine that can reach the dashboard can use the same CLI without Ray or SLURM:

```bash
hive connect head-node:38581
hive submit "nvidia-smi"
hive jobs
hive disconnect
```

or from Python, with no dependency beyond the standard library:

```python
from hive.client import Client

client = Client("http://head-node:38581")
job_id = client.submit("python train.py", num_gpus=1, workdir="/path/to/project")
result = client.wait(job_id)            # {"status", "returncode", "stdout", "stderr", ...}
print(client.status()["available_gpu"])
```

### REST API

| Method and path | Purpose |
|---|---|
| `GET /api/status` | `total_gpus`, `free_gpus`, `available_gpu`, `available_cpu`, `num_gpu_nodes`, `num_cpu_nodes`, `ray_head` |
| `GET /api/nodes`, `/api/gpu_stats`, `/api/cpu_stats`, `/api/slurm` | Per-node, per-GPU and SLURM job details shown in the dashboard |
| `GET /api/jobs` | Job list (the clients filter by status themselves) |
| `POST /api/submit` | Submit a job: `command`, `num_gpus`, `num_cpus`, `memory_mb`, `workdir`, `resource`, `timeout_sec`, `max_retries`, `priority`, `on_complete`, `user` |
| `GET /api/jobs/<id>/logs` | Status and output of one job |
| `POST /api/jobs/<id>/cancel` | Cancel a job |

## Security

The dashboard has **no authentication**, and `POST /api/submit` runs arbitrary
commands as the user who started Hive. It therefore binds to `127.0.0.1` by default.
Set `ray.dashboard_host: "0.0.0.0"` (or `hive serve --host`) only on a network where
every host that can reach the port is trusted, or put an authenticating proxy or an
SSH tunnel in front of it.

`hive stop`, `hive restart` and `hive relaunch` run `ray stop --force`, which stops
every Ray process you own on the head node, not only Hive's.

## Tests

```bash
pip install -e ".[dev]"
pytest tests            # unit tests; no SLURM, Ray cluster or network needed
```

`tests/live/` holds scripts for checking a running cluster; see its README.
