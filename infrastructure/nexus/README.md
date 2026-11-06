# Nexus

Nexus is the experiment boundary of Scientist World. It sits between an agent
(or the orchestrator that drives the agent) and the Hive compute pool, and
gives every experiment a bounded allocation called a **project**:

- a **GPU-hour budget**: cumulative GPU time over the project's jobs;
- a **concurrency cap**: the number of GPUs its jobs may hold at once;
- **allowed resource pools**: the Hive resources its jobs may run on;
- **periodic checkpoint review**: at a fixed interval of run time a job is
  flagged for review, so that its logs can be read and the job continued or
  cancelled before it spends more of the budget.

Nexus does not run anything itself. It checks each submission against the
project's limits, wraps the command in an `apptainer exec` call for the
project's container image and workspace, hands it to Hive, and then tracks the
job: status, logs, checkpoints, GPU time.

```
 agent or orchestrator          Nexus daemon                   Hive
+--------------------+  HTTP  +----------------------+  HTTP  +------------------+
| nx submit / wait   | -----> | REST API, port 39000 | -----> | dashboard API    |
| or any REST client |        | quota + budget check |        | runs the job on  |
+--------------------+        | job monitor thread   |        | a compute node   |
                              +----------------------+        +------------------+
                                         |
                                         v
                       <workspace>/jobs/<run_id>/   logs, checkpoints, result
```

In Scientist World the Swarm scheduler (`infrastructure/swarm`) and the
Evaluation harness (`evaluation`) each register one project per paper or run
and submit jobs on the agent's behalf through the REST API. The `nx` command
offers the same operations to an agent that is handed `NEXUS_URL` and
`NEXUS_PROJECT_ID` directly. Nexus imports nothing from the other components;
it only needs the HTTP API of a Hive dashboard.

Two command-line tools are installed:

- `nexus`: admin CLI for the daemon, projects and jobs;
- `nx`: agent CLI for submitting and following jobs (Python standard library
  only, see [Agent CLI](#agent-cli-nx)).

## Requirements

- Python 3.9 or newer, with `flask` and `pyyaml` (the test suite is run on
  Python 3.11).
- A running Hive dashboard that the daemon can reach over HTTP
  (`infrastructure/hive`). Nexus has no local execution path.
- Apptainer on the Hive compute nodes: every job is an `apptainer exec` of the
  project's SIF image.
- A filesystem shared by the daemon host and the compute nodes. The workspace,
  SIF image and overlay paths of a project are passed to Hive as they are, and
  the daemon itself writes job files into the workspace.
- Network reachability from wherever `nx` or a REST client runs to the daemon.

## Install

```bash
pip install -e infrastructure/nexus            # daemon, admin CLI, agent CLI
pip install -e "infrastructure/nexus[dev]"     # plus pytest
```

Use an editable install, or set `NEXUS_DIR` (see below): by default the state
directory is placed next to the `nexus` package, which for a regular install
would be inside `site-packages`.

## Configuration

Nexus reads the first of these files that exists:

1. `$NEXUS_DIR/config.yaml`
2. `infrastructure/nexus/nexus/config/config.yaml` (gitignored)

```bash
cd infrastructure/nexus
cp nexus/config/config.example.yaml nexus/config/config.yaml
```

Missing keys take the defaults below; unknown keys are ignored.

| Key | Default | Meaning |
|---|---|---|
| `hive.url` | `""` | Required. Base URL of the Hive dashboard, for example `http://<hive-host>:38581`. If empty, the `HIVE_URL` environment variable is used. The daemon refuses to start without one of the two. |
| `daemon.port` | `39000` | TCP port of the REST API and the web dashboard. |
| `daemon.host` | `127.0.0.1` | Bind address. Set it to `0.0.0.0` (or one interface address) to accept connections from other hosts. Read [Security model](#security-model) first. |
| `monitor.status_poll_interval` | `60` | Seconds between two Hive status polls of an active job. |
| `monitor.checkpoint_interval` | `1200` | Seconds of run time between checkpoints. A project can override it at registration. |
| `monitor.checkpoint_timeout` | `1800` | Seconds to wait for a continue/cancel decision at a checkpoint before the job is continued automatically. A project can override it at registration. |
| `project_defaults.max_gpus` | `4` | Concurrency cap of a project registered without `max_gpus`. |
| `project_defaults.default_job_timeout` | `3600` | Per-job wall-clock timeout in seconds of a project registered without one. |
| `project_defaults.soft_gpu_budget_hours` | `0` | GPU-hour budget of a project registered without one. `0` means unlimited. |

Environment variables:

| Variable | Read by | Meaning |
|---|---|---|
| `NEXUS_DIR` | daemon, `nexus`, `nx` | State directory. Default: `infrastructure/nexus/.nexus/` (next to the package). Read when the package is imported. Never defaults to the home directory. |
| `HIVE_URL` | daemon, `nexus start` | Fallback for an empty `hive.url`. |
| `NEXUS_URL` | `nx` | Daemon URL, for example `http://nexus-host:39000`. Not read by the admin CLI. |
| `NEXUS_PROJECT_ID` | `nx` | Project used when `--project` is omitted. |
| `NEXUS_LOG_STATE` | `nx` | File that remembers the `--log-dir` of each run (default `./.nx_log_dirs.json`). |

State directory layout:

```
$NEXUS_DIR/
  config.yaml                 optional, takes precedence over the package-level file
  projects/<project_id>.json  one record per project
  jobs/<run_id>.json          one record per job
  daemon.pid                  written by `nexus start`
  daemon.log                  output of the background daemon, truncated on every start
  remote.json                 written by `nexus connect`
```

## Quick start

```bash
# 1. Start the daemon (background, 127.0.0.1:39000 unless configured otherwise).
export HIVE_URL=http://HIVE_HOST:PORT          # or set hive.url in config.yaml
nexus start
nexus status

# 2. Register a project: at most 2 GPUs at once, 10 GPU-hours in total,
#    jobs may run on the Hive resources gpu-a and gpu-b.
nexus register --project-id demo --max-gpus 2 --gpu-budget 10 \
    --resource gpu-a --resource gpu-b \
    --workspace /shared/workspaces/demo --sif /shared/images/demo.sif

# 3. Submit and follow a job (this is what an agent does).
export NEXUS_URL=http://127.0.0.1:39000 NEXUS_PROJECT_ID=demo
nx submit "python train.py --lr 1e-4" --gpus 1    # prints "run_id: RUN_ID"
nx wait --run-id RUN_ID --timeout 600
nx logs RUN_ID                                    # while the job is active
cat /shared/workspaces/demo/jobs/RUN_ID/stdout.log   # after it has ended

# 4. Clean up.
nexus deregister demo
nexus stop                                     # also deletes all project and job records
```

`tests/manual_smoke.sh` automates steps 2 and 3 against a live deployment:
it registers a throwaway project, submits a job, waits for it, checks its
output and removes the project.

## Admin CLI (`nexus`)

| Command | What it does |
|---|---|
| `nexus start [--foreground] [--host H] [--port N]` | Start the daemon. In the background (default) it runs `python -m nexus.daemon` as a detached process, writes `daemon.pid` and `daemon.log`, and waits up to 15 s until the daemon accepts connections; it reports an error if the daemon exits during startup, if the port is already in use, or if no Hive URL is configured. `--foreground` serves from the current process until Ctrl+C. |
| `nexus stop` | Send SIGTERM to the daemon started by `nexus start`, then **delete all project and job records**. Jobs still running on Hive are not cancelled. Refuses while connected to a remote daemon. |
| `nexus status` | Daemon PID and uptime, Hive connectivity and GPU counts, project and job counts. |
| `nexus connect <host:port>` | Target a daemon on another host: stores its URL in `$NEXUS_DIR/remote.json`. |
| `nexus disconnect` | Remove `remote.json`. |
| `nexus register --project-id ID --max-gpus N --workspace DIR --sif PATH --resource NAME [...]` | Register a project. `--resource` is repeatable and required at least once. Optional: `--gpu-budget HOURS` (0 = unlimited), `--job-timeout S`, `--shared-bind SRC:DST[:ro]` (repeatable), `--checkpoint-interval S`, `--checkpoint-timeout S`, `--wake-command CMD`. |
| `nexus projects` | List projects with GPU usage, budget and job counts. |
| `nexus project ID` | Show one project. |
| `nexus extend ID --gpu-hours N` | Add `N` GPU-hours to the budget. This lifts a `budget_exhausted` block if the new budget exceeds the usage. |
| `nexus modify ID --max-gpus N` | Change the concurrency cap. Other fields can be changed with `PATCH /api/projects/<id>`. |
| `nexus deregister ID` | Cancel the project's active jobs and delete the project and its job records. |
| `nexus jobs [--project ID] [--status s1,s2]` | List jobs. |
| `nexus logs RUN_ID` | Job details and, while the job is active, its current stdout and stderr. |
| `nexus cancel RUN_ID` / `nexus cancel --project ID` | Cancel one job, or every active job of a project. |

The admin CLI talks to `http://localhost:<daemon.port>` (port from the config
file), or to the URL stored by `nexus connect`. It does not read `NEXUS_URL`,
and it does not know about a `--port` passed to `nexus start`: to use another
port permanently, set `daemon.port` in the config. A request that cannot reach
a connected remote daemon removes `remote.json`; so does `nexus start`.

The daemon can also be run directly, for example under a process supervisor:

```bash
python -m nexus.daemon [--host H] [--port N]
```

## Agent CLI (`nx`)

`nx` needs `NEXUS_URL` (or a `remote.json` in `NEXUS_DIR`) and a project, given
with `--project` or `NEXUS_PROJECT_ID`. It is a single file,
`nexus/agent_cli.py`, that uses only the Python standard library: inside a
container without the `nexus` package it can be copied and run as
`python3 agent_cli.py <command>`. `nx --help` and `nx <command> --help` print
the full reference.

| Command | What it does |
|---|---|
| `nx submit "<command>" [--gpus N] [--cpus N] [--timeout S] [--resource NAME] [--log-dir DIR] [--project ID]` | Submit a shell command. Defaults: 1 GPU, 1 CPU, the project's job timeout, all allowed resources. `--resource`/`-r` is repeatable. Prints `run_id: <id>`. Exits 1 if the submission is rejected. |
| `nx status [RUN_ID] [--project ID]` | One job, or every job of the project. |
| `nx logs RUN_ID` | Current stdout and stderr of an active job. |
| `nx wait [--run-id ID] [--timeout S] [--project ID]` | Block until the next event in the project and print `EVENT`, `RUN_ID`, `STATUS`; print `TIMEOUT: ...` and exit 0 if nothing happens. |
| `nx continue RUN_ID` | Answer a checkpoint: let the job go on. |
| `nx cancel RUN_ID` | Cancel a job. |
| `nx info [--project ID]` | Project status, allowed resources, GPUs in use, budget, job counts. |

`nx wait` follows the wait endpoint described under [REST API](#rest-api): one
call lasts at most 600 s whatever `--timeout` says; an event of any job of the
project ends it; and a job that ended before the call started is reported only
when it is named with `--run-id`.

With `--log-dir DIR`, `nx status`, `nx logs` and `nx wait` also write the
output they fetch to `DIR/<run_id>/stdout.log` and `stderr.log`. This mirror
is refreshed only while the job is active.

### Files in the workspace

The daemon writes each job's files into the project workspace, where the
agent can read them without calling the daemon:

```
<workspace_dir>/jobs/<run_id>/
  meta.json             job record: written at submission, when the job starts, when it ends
  stdout.log            written at every checkpoint and when the job ends
  stderr.log
  result.json           when the job ends: status, return code, duration, stdout, stderr
  checkpoints/NNN.json  one per checkpoint: timestamp, status, stdout, stderr
```

This is the only place where the complete output of a finished job is kept.
A job cancelled through Nexus gets its final `stdout.log` and `stderr.log`
but no `result.json`, and its `meta.json` is not updated.

## Projects

A project record holds:

| Field | Meaning |
|---|---|
| `project_id` | Unique name. A string without `/`. |
| `max_gpus` | Concurrency cap: GPUs held by the project's active jobs (queued, running or at a checkpoint). A submission that would exceed it is rejected. |
| `soft_gpu_budget_hours` | GPU-hour budget, `0` = unlimited. See below for how it is enforced. |
| `allowed_resources` | Hive resource names the project may use. A submission may name a subset; by default it is sent to Hive with all of them. An empty list means any resource. |
| `default_job_timeout` | Wall-clock timeout in seconds given to Hive for jobs that do not set their own. |
| `max_cpus`, `max_memory_mb` | Per-job limits, `0` = unlimited, compared with the `num_cpus` and `memory_mb` of the submit request. `num_cpus` is passed on to Hive; `memory_mb` is not, so the memory limit is not enforced at run time. |
| `workspace_dir` | Host directory mounted at `/home/user` in every job container. Required in practice. |
| `sif_path` | Apptainer image of the jobs. Required in practice. |
| `overlay_path` | Optional overlay image, mounted read-only. |
| `shared_dirs` | Extra Apptainer bind specifications (`SRC:DST[:ro]`). |
| `hf_cache_path` | Optional host directory mounted at `/home/user/shared/hf_cache`. |
| `checkpoint_interval`, `checkpoint_timeout` | Per-project overrides of the `monitor.*` settings. |
| `wake_command` | Optional shell command run on the daemon host at each event. |
| `status`, `gpus_in_use`, `gpu_seconds_used`, `budget_exhausted`, `created_at` | Maintained by the daemon. |

A project is `registered` until its first job is submitted and `active`
afterwards. There is no retired state: deleting a project removes it.

**Budget.** When a job ends, `num_gpus x (end - start)` seconds are added to
`gpu_seconds_used`. Start and end are the moments at which the monitor
observed the job running and ended, so the resolution is the poll interval
(for a job that ends before the monitor ever saw it running, the start time
reported by Hive is used). CPU-only jobs cost nothing.

The budget is *soft*: a running job is never stopped for exceeding it. After
a job ends, if the usage has reached the budget and the project has no other
active job, the project is marked `budget_exhausted` and every later
submission is rejected with HTTP 403 until the budget is raised
(`nexus extend`, or `PATCH` with `soft_gpu_budget_hours`) or the counter is
reset (`PATCH` with `gpu_seconds_used`).

**Wake command.** The template variables `{project_id}`, `{run_id}`,
`{event}`, `{workspace}` and `{jobs_dir}` are filled in and the command is run
through the shell with a 30 s timeout. Events are `checkpoint`, `completed`,
`failed` and `budget_exhausted` (with `run_id` set to `budget`).

## Jobs

```
queued --> running --> completed   (Hive reports completed with return code 0)
              |   \--> failed      (any other end: non-zero exit, timeout, cancelled on Hive)
              |   \--> cancelled   (cancelled through Nexus)
              v
       awaiting_response --> running     (continue, or no answer within checkpoint_timeout)
                         \-> cancelled   (cancel)
```

The monitor thread of the daemon polls Hive for every active job each
`status_poll_interval` seconds.

- **Checkpoints.** When a running job has run for `checkpoint_interval`
  seconds since it started or since its previous checkpoint, the monitor
  writes `checkpoints/NNN.json` and the current logs into the workspace, sets
  the job to `awaiting_response` and runs the wake command. The job itself
  keeps running on its compute node. `continue` returns it to `running`,
  `cancel` stops it, and without an answer it is continued automatically after
  `checkpoint_timeout` seconds.
- **End of a job.** The monitor releases the job's GPUs from the concurrency
  cap, saves the final status, writes logs and `result.json` into the
  workspace, adds the GPU time to the budget and runs the wake command.
- **Restart.** When the daemon starts it reloads the active jobs from disk,
  marks jobs whose project no longer exists as failed, and recomputes each
  project's `gpus_in_use` from its active jobs.
- **Hive preemption.** A job that Hive reports as `preempted` is left alone
  (Hive re-queues it) and is treated as failed only after 10 minutes in that
  state.

### What a job runs as

For a project and a submitted `<command>`, Hive receives:

```
apptainer exec [--nv] --containall --cleanenv
  --env HF_HOME=/home/user/shared/hf_cache
  --env TRANSFORMERS_CACHE=/home/user/shared/hf_cache
  --env HF_DATASETS_CACHE=/home/user/shared/hf_cache/datasets
  --env TORCH_HOME=/home/user/shared/hf_cache/torch
  --bind <workspace_dir>:/home/user
  [--overlay <overlay_path>:ro]
  [--bind <shared_dirs entry>]...
  [--bind <hf_cache_path>:/home/user/shared/hf_cache]
  <sif_path> bash -c 'cd /home/user && <command>'
```

`--nv` is added when the job asks for at least one GPU. The mount point
`/home/user` and the four cache variables are fixed; without `hf_cache_path`
the cache directory is `shared/hf_cache` inside the workspace. Along with the
command, Nexus sends Hive the GPU and CPU counts, the timeout, the resource
names, a `user` label (`nexus/<OS user>`) and `max_retries: 3`.

Nexus uses four Hive endpoints: `POST /api/submit`,
`GET /api/jobs/<id>/logs`, `POST /api/jobs/<id>/cancel` and `GET /api/status`.
A submit is retried for up to 10 minutes on HTTP 5xx, 429 and connection
errors (waits of 5, 10, 20, 30, then 60 s); any other 4xx fails at once.

## REST API

All bodies are JSON. Errors are `{"error": "<message>"}` with a 4xx or 5xx
status. There is no authentication and no separation between admin and agent
routes.

| Method and path | Request | Response |
|---|---|---|
| `POST /api/projects` | `project_id` (required) and any of `max_gpus`, `allowed_resources`, `max_cpus`, `max_memory_mb`, `default_job_timeout`, `soft_gpu_budget_hours`, `workspace_dir`, `sif_path`, `overlay_path`, `hf_cache_path`, `shared_dirs`, `wake_command`, `checkpoint_interval`, `checkpoint_timeout` | 201 `{project_id, status: "registered"}`. 400 without a valid `project_id`, 409 if it exists. |
| `GET /api/projects` | | List of project summaries with `job_counts`, newest first. |
| `GET /api/projects/<id>` | | The project record plus `job_counts`. 404 if unknown. |
| `PATCH /api/projects/<id>` | Any of `max_gpus`, `allowed_resources` (non-empty list), `max_cpus`, `max_memory_mb`, `soft_gpu_budget_hours`, `gpu_seconds_used`, `default_job_timeout`, `sif_path`, `overlay_path` | The updated project record. |
| `DELETE /api/projects/<id>` | | Cancels the active jobs, deletes the job records and the project. `{project_id, status: "deleted"}`. |
| `POST /api/projects/<id>/jobs` | `command` (required), `num_gpus` (1), `num_cpus` (1), `memory_mb` (0), `timeout_sec` (project default), `resource` (comma-separated names), `client_request_id` | 201 `{run_id, hive_job_id, project_id, command, num_gpus, resource, status: "queued", timeout_sec, project_gpus_in_use, project_max_gpus, gpu_seconds_used, soft_gpu_budget_hours}`. 400 without `command`, 404 for an unknown project, 403 if the concurrency cap, the budget, a per-job limit or the resource list forbids it, 500 if Hive could not be reached or rejected the job. |
| `GET /api/projects/<id>/jobs` | | List of job records, active jobs first, then newest first. |
| `GET /api/jobs/<run_id>` | | The job record. While the job is active also `hive_status`, `stdout`, `stderr` as currently reported by Hive. |
| `POST /api/jobs/<run_id>/continue` | (body ignored) | `{run_id, status: "running"}`, whether or not the job was at a checkpoint. |
| `POST /api/jobs/<run_id>/cancel` | (body ignored) | `{run_id, status: "cancelled"}`, whether or not the job was still active. |
| `GET /api/projects/<id>/wait` | Query: `timeout` (seconds, default 300, capped at 600), `run_id` (optional) | `{run_id, status, event}` or `{"timeout": true}`. See below. |
| `GET /api/status` | | `daemon {uptime, pid}`, `hive {url, connected}` plus, when connected, `total_gpus`, `available_gpu`, `num_gpu_nodes`, `num_cpu_nodes`; `projects {total, active, budget_exhausted}`; `jobs {running, completed, failed}` (`running` counts every active job). |
| `GET /` | | Web dashboard (HTML), refreshing every 5 s. |

A job record has `run_id`, `project_id`, `command`, `num_gpus`, `num_cpus`,
`status`, `submitted_at`, `started_at`, `completed_at`, `returncode`,
`last_checkpoint_at`, `checkpoint_count` and, once known, `timeout_sec`,
`resource`, `hive_job_id` and `actual_resource` (the Hive resource the job
ran on). Timestamps are Unix seconds.

**Waiting.** `GET /api/projects/<id>/wait` returns as soon as a job of the
project reaches `awaiting_response`, `completed`, `failed` or `cancelled`
after the request started; `event` equals that status. If `run_id` names a job
that is already in one of these states, the call returns at once with that
job. A job that got there before the request started and is not named by
`run_id` is not reported. After `timeout` seconds without an event the answer
is `{"timeout": true}`.

**Idempotent submit.** A client that may retry a submit (for example after
its own HTTP timeout) should send the same `client_request_id` with every
attempt. The daemon replays the first response for 10 minutes instead of
creating a second job, and makes a retry wait while the original request is
still being processed. The replay cache is kept in memory (it is lost when
the daemon restarts) and is keyed by the id alone, so use an id that is
unique across projects, such as a UUID. Rejections with HTTP 403 whose message
contains `in use` (concurrency cap) or `budget` clear up when a job ends or
the budget is raised, so a client can retry them.

### Python client

`nexus.client.NexusClient` wraps the project and read-only job routes
(`register_project`, `get_project`, `delete_project`, `list_jobs`, `get_job`,
`status`) with the standard library only. It does not submit, wait for,
continue or cancel jobs; use the REST API for those.

## Security model

The Nexus API has **no authentication and no authorization**. Whoever can
open a TCP connection to the daemon port can do everything the API offers:

- register a project with any `workspace_dir`, `sif_path` and `shared_dirs`,
  and submit commands that Hive then runs on the compute nodes as the OS user
  of the Hive pool, with those host directories mounted;
- set a `wake_command`, which the daemon runs through the shell on its own
  host;
- raise a quota, reset a budget, cancel jobs or delete projects, including
  those of other projects. Admin and agent routes are the same server.

Consequences:

- The default bind address is `127.0.0.1`. To let agents or orchestrators on
  other hosts reach the daemon, set `daemon.host` to `0.0.0.0` (or pass
  `--host`) and restrict access with a firewall or a private network. The
  daemon logs a warning when it listens on a non-loopback address.
- Loopback is not isolation from local processes. Any process on the daemon
  host can connect, including an agent inside a container that shares the
  host network namespace (the Apptainer default).
- An agent that holds `NEXUS_URL` can therefore change its own limits. Nexus
  bounds the compute of a cooperative agent and of runaway jobs; it is not a
  defence against an agent that deliberately attacks the API. If that matters,
  do not give the agent network access to the daemon, and submit jobs on its
  behalf from the orchestrator instead (as Swarm and Evaluation do).
- The daemon is served by Flask's built-in server, over plain HTTP.

## Tests

```bash
cd infrastructure/nexus
python -m pytest                      # whole suite
python -m pytest tests/test_daemon.py -v
```

The suite needs only `pytest` and the package's own dependencies: no Hive, no
cluster, no network. Hive is mocked, the REST API is exercised through
Flask's test client, and two autouse fixtures in `tests/conftest.py` remove
the Nexus environment variables and redirect the state directory to a
temporary one, so the suite never touches a real deployment. Run it in its own
pytest invocation: the Hive tests live in a package that is also called
`tests`.

`tests/manual_smoke.sh` is a manual end-to-end check for a real deployment
(running Hive and Nexus, Apptainer, a SIF image); it is not collected by
pytest.

## Known limitations

- **`nexus stop` deletes all state** and does not cancel jobs on Hive. The
  restart recovery of the monitor therefore only applies when the daemon
  crashes, is killed, or is stopped by other means than `nexus stop`.
- **No authentication** (see above), and request fields are not validated
  beyond what is listed: `workspace_dir` and `sif_path` may be empty, numeric
  fields are not range-checked.
- **Output of a finished job is not served by the API.** `nx logs`,
  `nexus logs` and `GET /api/jobs/<run_id>` return stdout and stderr only
  while the job is active; afterwards read the files in the workspace.
- **Checkpoints suspend Hive polling.** While a job is `awaiting_response`
  the monitor does not poll Hive for it. If the job ends in that window, this
  is noticed only after `continue` or after `checkpoint_timeout`, and the
  extra time is counted towards the budget. The checkpoint interval is
  measured from the previous checkpoint, not from the moment the job was
  continued: when the decision takes longer than `checkpoint_interval`, or
  never comes and `checkpoint_timeout` is the larger of the two (as with the
  defaults), the next checkpoint follows at the next poll.
- **A job that Hive has lost stays active.** Every error of the Hive poll,
  including "job not found", is treated as transient. Such a job keeps its
  share of the concurrency cap until it is cancelled or its project deleted.
- **Cancellation is best effort.** If Hive cannot be reached, the job is still
  marked cancelled in Nexus and its GPUs are released, although it may keep
  running on Hive. The GPU time of a job cancelled through Nexus is not added
  to the budget.
- **`max_memory_mb` is an admission check only**, against a value the client
  chooses to send; nothing is enforced at run time. `nx` has no memory flag.
- **`nx wait` and the wait endpoint** report an event of any job of the
  project, last at most 600 s, and miss jobs that ended before the call unless
  `run_id` is given.
- **The admin CLI assumes a local daemon on the configured port** unless
  `nexus connect` was used; it ignores `NEXUS_URL` and `nexus start --port`.
- **Persistence is plain JSON files.** Writes are not atomic, every status,
  dashboard and wait request re-reads all job files, and job records are
  removed only when their project is deleted (or by `nexus stop`).
- **Fixed container conventions**: the workspace mount point `/home/user` and
  the four cache environment variables cannot be changed through the API, and
  Hive is the only execution backend.
- **Startup reconciliation** of `gpus_in_use` covers only projects that have
  at least one job record.
- The background daemon log is truncated on every `nexus start`.
