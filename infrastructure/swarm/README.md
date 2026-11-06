# Swarm

Swarm builds research environments from papers. It coordinates the *builders*,
agents that take one of four roles: reproducer, designer, packager, verifier.

- **Per paper** it drives one build: a workflow of builder passes
  (*iterations*), and inside each iteration the loop "agent turn, compute job,
  agent turn".
- **Across papers** it runs many builds in parallel from a queue, under
  per-state concurrency caps.

The code lives in `infrastructure/swarm`. Swarm does not talk to models itself:
every agent turn goes through the Worker package (`infrastructure/worker`),
which wraps the Claude Code and Codex CLIs and ships the prompts and the
workspace template. Every compute job goes through Nexus
(`infrastructure/nexus`), which enforces the GPU budget and forwards the job to
Hive (`infrastructure/hive`). Papers come from Scout (`scout`), and
the environments Swarm produces are what the Evaluation harness (`evaluation`)
later runs scientist agents on.

```
Scout data --> swarm submit --> queue --> one thread per paper
                                                 |
                         Worker (agent CLI in an Apptainer container on this host)
                                                 |  action.yaml: submit | done
                         Swarm: build container, run setup.sh, submit scripts/run.sh
                                                 |
                         Nexus (budget) --> Hive (compute nodes)
                                                 |
                         git commits in the workspace --> GitHub repository per paper
```

## Contents

1. [Requirements](#requirements)
2. [Install](#install)
3. [Quick start](#quick-start)
4. [How one paper is built](#how-one-paper-is-built)
5. [Workflow](#workflow)
6. [Wake events](#wake-events)
7. [Milestones and gating](#milestones-and-gating)
8. [Validation](#validation)
9. [Configuration reference](#configuration-reference)
10. [Command line](#command-line)
11. [Dashboard and REST API](#dashboard-and-rest-api)
12. [Files Swarm reads and writes](#files-swarm-reads-and-writes)
13. [Tests](#tests)
14. [Operational notes](#operational-notes)
15. [Security notes](#security-notes)
16. [Limitations](#limitations)

## Requirements

- Linux and Python 3.10 or later. The host that runs the daemon also runs the
  agents, builds the containers and installs packages, so it needs internet
  access. Compute nodes do not.
- A running **Hive** pool and a **Nexus** daemon connected to it. Swarm only
  knows Hive by the resource names listed in `nexus.resources`. Swarm does not
  set a job timeout: set `project_defaults.default_job_timeout` in the Nexus
  config (one hour by default) to the value of `nexus.job_time_limit`, which is
  the limit the agents are told.
- **Apptainer** on the host, with:
  - user namespaces for `--fakeroot` (used to run `setup.sh`);
  - cgroups v2 with memory delegation for your user (`apptainer exec --memory`
    is passed for every agent turn; without it every agent launch fails);
  - ext3 overlay support.
- Host tools: `git`, `uv`, `nice`, `truncate`, `mkfs.ext3`, GNU `cp`, `lsof`,
  `pgrep`. `git-filter-repo` is optional (see [Operational notes](#operational-notes)).
- At least one agent CLI on the host, logged in:
  - `claude` (Claude Code). The binary found on `PATH` is bind-mounted into the
    agent container, which has no Node.js, so it must be a self-contained
    executable. Credentials are read from `~/.claude/.credentials.json`.
  - `codex`. Its Node.js prefix is bind-mounted into the agent container.
    Credentials are read from `~/.codex/auth.json`.
  - A model behind an Anthropic-compatible gateway can be used through a
    `claude-code` profile with `env_vars` (see the example config).
- **A GitHub organisation and a token. These are currently required.** Swarm
  creates one private repository per paper in `github_org` and pushes to it at
  the start and end of every iteration. The token is read from
  `~/.git-credentials` (`https://<user>:<token>@github.com`); no other source is
  consulted, and user accounts are not supported as owners. Without an
  organisation or token the daemon still starts (with a warning) and builds
  run, but:
  - every iteration ends with status `stalled` and the error "git push failed",
    including iterations that were cancelled or that ended with an agent error;
  - `stalled` does not stop a workflow, so the workflow keeps advancing;
  - the end-of-iteration commit is rewritten without `trajectory/` (the
    fallback Swarm tries after a failed push), so agent trajectories are not
    committed.
  Treat a setup without GitHub as unsupported.
- Scout output for every paper you submit: `arxiv_mds/<id>/<id>.md` is
  required; `arxiv_mds/<id>/images/`, `arxiv_pdfs/<id>.pdf` and
  `paper_index.json` (for the title) are used when present.

## Install

Swarm imports the Worker package and its workspace template from the repository
checkout, so it must be installed in editable mode from a clone:

```bash
pip install -e infrastructure/hive -e infrastructure/nexus -e infrastructure/swarm

# agent container (see infrastructure/worker/README.md)
apptainer build reproducer.sif infrastructure/worker/container/reproducer.def

cp infrastructure/swarm/swarm/config/config.example.yaml \
   infrastructure/swarm/swarm/config/config.yaml
```

`nexus` has to be installed in the same environment only if you want Swarm to
start the Nexus daemon itself. Then edit `config.yaml`; the lines marked
REQUIRED are `scout_data_dir`, `workspace_base_dir`, `github_org`,
`worker.container_sif`, `nexus.url` and `nexus.resources`.

## Quick start

```bash
swarm start                                   # daemon + dashboard on 127.0.0.1:39100
swarm submit 2512.01152 --date 2025-12-01 --topic nlp
swarm status                                  # queue and capacity
swarm papers                                  # one line per paper
swarm logs 2512.01152                         # state and turns of the last iteration
swarm pause-all                               # mark everything resumable, stop the daemon
swarm start && swarm continue-all             # resume after a restart
```

## How one paper is built

1. **Queue.** `swarm submit` resolves the paper under `scout_data_dir`,
   picks the workflow (`--workflow` or `default_workflow`), the first step's
   profile and GPU budget, and appends the paper to the queue.
2. **Dispatch.** When a slot is free the paper gets its own thread. Its state
   moves between `idle`, `working`, `building`, `installing`, `waiting` and
   `wrapping_up`; each state has a cap (see [Concurrency](#concurrency)).
3. **Workspace.** `<workspace_base_dir>/<arxiv_id>/` is created from
   `infrastructure/worker/workspace_template`, the paper's markdown, images and
   PDF are copied into `paper/`, a git repository is initialised, and a private
   GitHub repository `<github_org>/<arxiv_id>-<title-slug>` is created and
   pushed.
4. **Nexus project.** A project `swarm-<arxiv_id>` is created with the
   iteration's remaining GPU budget, `nexus.max_gpus`, the Hive resources, the
   shared directories and, once they exist, the paper's container and overlay.
   It is deleted when the iteration ends.
5. **First prompt** of the iteration, depending on the role (see
   [Workflow](#workflow)).
6. **Action loop.** After every agent turn Swarm reads `action.yaml` from the
   workspace:
   - `action: submit`: if `environment/container.def` changed, build
     `environment/container.sif`; if `environment/setup.sh` changed, run it into
     `environment/overlay.img`; then submit `bash /home/user/scripts/run.sh` to
     Nexus with the requested number of GPUs, and wait. The agent is woken by
     the events listed under [Wake events](#wake-events).
   - `action: done`: push the agent forward if it is below `target_milestone`,
     otherwise send the wrap-up prompt and end the iteration.
7. **End of iteration.** Swarm fixes the milestone, writes `manifest.json`,
   runs the [validation](#validation) gate, compiles `README.md` from the
   briefing files and the manifest, commits ("End of Iteration N: ...") and
   pushes, and updates the repository's description and topics.
8. **Gate, revert or advance.** See [Milestones and gating](#milestones-and-gating).
   After the last step, `environment/container.final.def` is written: the
   paper's `container.def` plus a pinned install of every package found in the
   overlay and in the `setup.sh` history.

Container builds are cached by the hash of `container.def`, base images are
pulled once per image, and `setup.sh` results are cached by (container hash,
setup hash). The container definition must use `Bootstrap: docker` with a
Docker Hub image; anything else is rejected before the build so that the
definition builds on other hosts.

## Workflow

A workflow is a list of steps; step *N* is iteration *N*. The workflow used to
build the environments in the paper is `standard` in
`swarm/config/config.example.yaml`:

| Iteration | Role | Profile in the example | GPU-hours | Gate (`min_milestone`) | On gate failure, re-run iteration |
|---|---|---|---|---|---|
| 1 | reproducer | `codex-medium` | 30 | `core_claim` | 1 |
| 2 | reproducer | `codex-low` | 15 | `core_claim` | 1 |
| 3 | reproducer | `codex-high` | 10 | `core_claim` | 2 |
| 4 | designer_stage_1 | `codex-high` | 1 | none | |
| 5 | designer_stage_2 | `codex-medium` | 1 | none | |
| 6 | designer_stage_3 | `codex-high` | 5 | `core_claim` | 1 |
| 7 | packager | `codex-high` | 1 | none | |
| 8 | verifier | `codex-medium` | 3 | `core_claim` | 1 |
| 9 | verifier | `codex-high` | 3 | `core_claim` | 2 |

`max_revert_attempts` is 2. The profile names are examples; the three profiles
differ only in reasoning effort.

What each role receives and may do:

| Role | First prompt(s) (in `infrastructure/worker/prompts`) | Milestone | Validation mode |
|---|---|---|---|
| reproducer, iteration 1 | `reproduce_start.md` | set by the agent | `--post-reproducer` |
| reproducer, later iterations | `reproduce_start.md`, then `on_iterate.md` | starts one level below the previous claim; `MILESTONES.md` and `action.yaml` are reset | `--post-reproducer` |
| designer_stage_1 | `on_designer_stage_1.md` | kept from the previous iteration | `--during-design` |
| designer_stage_2 | `on_designer_stage_2.md` | kept | `--during-design` |
| designer_stage_3 | `on_designer_stage_3.md` | may be changed by the agent | `--post-design` |
| packager | `on_packager.md` | kept | `--post-design` |
| verifier | `reproduce_start.md`, then `on_verify.md` | may be changed by the agent | `--post-design` |
| designer | `on_designer.md` (single-stage designer, not used by `standard`) | kept | default mode |

Every iteration starts a new agent session. Designer stages 1 and 2 only write
files: their `action.yaml` is reset to `gpus: 0` and their prompts forbid
submitting jobs.

## Wake events

An agent runs only when Swarm sends it a prompt. Besides the first prompt of an
iteration, these events wake it:

| Event | Prompt | Notes |
|---|---|---|
| `action.yaml` missing or invalid after a turn | built-in fix request | up to 2 times per turn |
| Container build or `setup.sh` failed | `on_container_build_failure.md` | includes the failure class, a signature, a hint and the last 40 log lines |
| Job checkpoint (Nexus `awaiting_response`) | `on_job_progress.md` | cadence is Nexus's `monitor.checkpoint_interval`; a reply containing "cancel" cancels the job, anything else continues it |
| Job cancelled at the agent's request | `on_job_cancelled.md` | |
| Job completed, failed or was cancelled | `on_job_completion.md` | with the return code and the log paths |
| GPU budget of the iteration used up | `on_budget_exhausted.md` + `on_wrap_up.md` | repeated once if the agent does not set `action: done` |
| `milestone_cap` of the step reached | `on_wrap_up.md` | ends the iteration |
| `action: done` below `target_milestone` | `on_push_forward.md` | at most `push_forward_attempts` times; not sent for `not_possible` |
| `action: done` otherwise | `on_wrap_up.md` | not sent for `not_possible` |
| `validate.py` failed | `on_validation_failure.md` | up to 2 times (3 validation attempts) |
| Error at or above `target_milestone` | `on_wrap_up.md` | so the work is documented and committed |
| Resuming a saved session (`swarm continue`) | `on_continue.md` | |
| 10 consecutive build/setup failures, or 10 consecutive failed jobs | built-in stop message | the iteration ends with status `error` |

`on_invalid_milestone.md` exists for milestone names outside the ladder, but is
never sent (see [Limitations](#limitations)).

## Milestones and gating

The ladder, lowest to highest:

`not_possible` < `none` < `method_runs` < `core_claim` < `core_claim_plus` <
`secondary_claims` < `majority` < `near_complete` < `full`

- Milestones are self-reported by the agent in `action.yaml` (authoritative)
  and in `MILESTONES.md` (`**Current: <name>**`, used only when `action.yaml`
  carries none). Common misspellings are mapped to the canonical name; any
  other unknown name is recorded as `none`.
- A later reproducer iteration starts one level below the milestone the
  previous iteration claimed. Packager and designer stages 1 and 2 cannot
  change the milestone.
- If an iteration ends with status `error` below the milestone it started
  from, the previous milestone is restored.
- **Gate.** When a step has `min_milestone` and the iteration ends below it:
  - if the step has `revert_to_iteration` and the paper has used fewer than
    `max_revert_attempts` reverts, that earlier iteration is queued again, at
    the front of the queue, with a new session. Nothing in the workspace or its
    git history is rolled back;
  - otherwise the paper ends with status `unsuccessful`.
- **Advance.** If the gate passes (or the step has none) the next step is
  queued at the front of the queue. Only the statuses `error` and `cancelled`
  stop a workflow.

Paper statuses: `queued`, `active`, `done`, `error`, `cancelled`, `stalled`
(validation or push failed), `unsuccessful` (gate failed, no reverts left),
`paused` (interrupted by a daemon stop or restart; resume with
`swarm continue-all`).

## Validation

At the end of every iteration Swarm runs the workspace's `validate.py` (copied
from `infrastructure/worker/workspace_template` and re-synced before every
commit) with the mode listed in the role table. The validator checks:

- `scoring/reference.json`: structure, weights summing to 1, no metric that
  measures agreement with the paper;
- `scoring/scores*.json`: consistency with the reference;
- with `--post-reproducer` and `--post-design`: the primary metric must not be
  far from the paper's value (sign flip, a factor of 2 or more, or an absolute
  gap of 0.3 or more for values in [-1.5, 1.5]);
- with `--during-design`: `eval/train/`, `eval/test/`,
  `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` exist and the two
  split score files do not; with `--post-design`: all six split artifacts
  exist;
- `eval/` and `data/` do not import from `method/`; imports resolve to
  git-tracked files; `eval/train` and `eval/test` do not import each other;
  committed code does not reference `shared/`;
- the two evaluate scripts point at their own slice and have no hard-coded
  method name; train and test scores cover the same methods, differ from each
  other, and the primary metric distinguishes methods.

On failure the agent is woken with the validator output and the check is
repeated, up to three attempts. If it still fails, the push is skipped and the
iteration ends with status `stalled`. Two properties of the current code are
worth knowing:

- a validator run that exceeds 300 seconds counts as a pass;
- `stalled` does not stop the workflow: the next step starts on a workspace
  that failed validation and was not pushed.

## Configuration reference

Configuration is YAML, read from `$SWARM_DIR/config.yaml` if it exists,
otherwise from `infrastructure/swarm/swarm/config/config.yaml`. Unknown keys
are ignored; unknown roles, milestones, scaffolds and profile references raise
an error when the config is loaded. `config.example.yaml` documents every key
with the values used for the paper; the tables below are a summary.

Environment variables: `SWARM_DIR` (state directory, default
`infrastructure/swarm/.swarm/`), `NEXUS_URL` (used when `nexus.url` is empty),
`SCOUT_DATA_DIR` (used when `scout_data_dir` is empty).

### Concurrency

| Key | Default without `pool_size` | Meaning |
|---|---|---|
| `pool_size` | unset | Papers being built at once. When set it overrides the sum of the caps below. |
| `max_working_workers` | 4 | Agent turns running at once. |
| `max_working_return_overflow` | 0 | Extra working slots only for papers returning from a job, build or install. |
| `max_admitted_workers` | 10 | Papers taken from the queue that have not started their first turn. |
| `max_building_workers` | 2 | Concurrent container builds. |
| `max_installing_workers` | 4 | Concurrent `setup.sh` runs. |
| `max_runtime_workers` | 4 | One shared cap for idle, waiting, building and installing papers. Used only when none of the three caps above is set. |

When `pool_size` is set, an omitted `max_working_workers` or
`max_runtime_workers` defaults to it, and the other omitted caps to the smaller
of it and the default in the table. When any of `max_admitted_workers`,
`max_building_workers` or `max_installing_workers` is set, papers waiting on a
compute job are limited only by the total. Without `pool_size` the total is the
sum of the caps in use.

### Paths

| Key | Default | Meaning |
|---|---|---|
| `scout_data_dir` | required | Scout data directory. |
| `workspace_base_dir` | required | One workspace per paper is created here. Must be visible on compute nodes. |
| `shared_datasets_path`, `shared_models_path` | `/shared/datasets`, `/shared/models` | Mounted read-only in agent, setup and job containers at `/home/user/shared/datasets` and `/home/user/shared/models`. |
| `hf_cache_path` | empty | Shared HuggingFace cache, passed to Nexus for job containers. |
| `container_cache_dir` | `<workspace_base_dir>/../shared/containers` | Cache of built containers and base images. `setup.sh` results are cached in the sibling directory `setup_overlays/`. |
| `uv_cache_dir` | empty | uv cache shared by agent and setup containers. |
| `apptainer_cache_dir` | empty | Exported as `APPTAINER_CACHEDIR`. |
| `local_tmp_dir` | `/tmp` | Node-local disk for agent container copies, builds and overlays under construction. Must exist. |

### Agent containers, dashboard, GitHub, milestones

| Key | Default | Meaning |
|---|---|---|
| `container_memory_limit` | `8G` | Memory limit per agent container. |
| `container_nice` | 19 | `nice` value of agent containers. |
| `container_cpu_threads` | 2 | Thread cap (OMP, MKL, OpenBLAS, NumExpr, VecLib) in agent containers. |
| `dashboard_host` | `127.0.0.1` | Interface of the dashboard and API. Set `0.0.0.0` to expose it; there is no authentication. |
| `dashboard_port` | 39100 | Port of the dashboard and API. |
| `github_org` | empty | GitHub organisation for the per-paper repositories. Required in practice. |
| `target_milestone` | `core_claim` | Threshold for push-forward and for wrap-up after an error. |
| `push_forward_attempts` | 1 | Times an agent that stops below `target_milestone` is asked to continue. |
| `default_workflow` | `standard` | Workflow of `swarm submit` without `--workflow`. Must be defined, or set to `""`. |

### `profiles`

A profile names an agent. Workflow steps, `worker.profile` and `--profile`
refer to profiles by name. Profiles are re-read from the config file each time
a paper starts, so they can be edited while the daemon runs.

| Key | Meaning |
|---|---|
| `scaffold` | `codex` or `claude-code`. |
| `model` | Value for the CLI's `--model`. |
| `effort` | Codex: `model_reasoning_effort`. Claude Code: `CLAUDE_CODE_EFFORT_LEVEL`. |
| `actual_model` | Model name for manifests and git identity when `model` is a gateway alias. |
| `env_vars` | Extra environment variables for the agent process (gateway endpoint and token). |
| `api_keys` | Keys to rotate through on a hard usage-limit error; the next key replaces `env_vars.ANTHROPIC_AUTH_TOKEN`. |
| `compact_token_limit` | Context size at which the CLI compacts the conversation. Codex: `-c model_auto_compact_token_limit` (default 120000). Claude Code: `CLAUDE_CODE_AUTO_COMPACT_WINDOW` (default: the CLI's own). |
| `max_turns` | Agentic turns per invocation (Claude Code only). |
| `container_sif` | Agent container for this profile. |

### `worker`

| Key | Default | Meaning |
|---|---|---|
| `profile` | unset | Default agent, for papers with neither a workflow nor `--profile`. |
| `container_sif` | unset | Agent container. Without it agents run directly on the host. |
| `max_turns` | 200 | Agentic turns per invocation (Claude Code only). |
| `setup_timeout` | 3600 | Seconds allowed for one `setup.sh` run (1200 in the example). |

Any profile key may also be set under `worker` to override the default profile.

### `workflows`

```yaml
workflows:
  <name>:
    max_revert_attempts: 2      # reverts a paper may use in total (default 0)
    steps:
      - profile: codex-medium   # required
        role: reproducer        # default; see the role table
        gpu_budget_hours: 30    # default: nexus.gpu_budget_hours
        milestone_cap: full     # end the iteration when this milestone is reached
        min_milestone: core_claim
        revert_to_iteration: 1  # needs min_milestone; not later than this step
        fallback_profiles: []   # tried in order on a hard API usage limit
```

A workflow may also be a bare list of steps, which means no reverts.

### `nexus`

| Key | Default | Meaning |
|---|---|---|
| `url` | empty | Nexus daemon URL. Required (or `$NEXUS_URL`). |
| `autostart_host` | `127.0.0.1` | Interface of a Nexus daemon that Swarm starts itself. |
| `gpu_budget_hours` | 50.0 | Budget of an iteration whose step sets none. |
| `max_gpus` | 4 | Most GPUs one job may request (2 in the example). |
| `job_time_limit` | `24:00:00` | Per-job limit shown to the agent. Not enforced by Swarm. |
| `resources.gpu`, `.gpu_test`, `.cpu`, `.cpu_test` | empty | Hive resource names per job type. Configure all four: the agent chooses the type, and a job of a type with no resource fails to submit. |

### Excluded papers

Copy `swarm/config/excluded_papers.example.yaml` to
`swarm/config/excluded_papers.yaml` to make Swarm refuse `submit`, `continue`
and `iterate` for the listed arXiv ids. Without that file nothing is excluded.

## Command line

`swarm [-v] <command>`. All commands except `start`, `stop`, `reset`,
`reset-all` and `logs` talk to the running daemon at
`http://localhost:<dashboard_port>`.

| Command | What it does |
|---|---|
| `start [--port N] [--foreground]` | Start the daemon (background by default; output in `$SWARM_DIR/daemon.log`). `--port` changes the port for this start only; the other commands always use `dashboard_port`. |
| `stop` | Send SIGTERM to the daemon. Agents and builds are killed; papers show as `paused` after the next start. |
| `pause-all` | Record how to resume every queued and active paper, then stop the daemon. |
| `continue-all [--profile P] [--no-workflow] [--fresh]` | Resume every paper interrupted by a daemon stop or restart. |
| `submit <arxiv_id> [--workflow W] [--profile P] [--gpu-budget H] [--date YYYY-MM-DD] [--topic T ...]` | Queue a paper. `--date` and `--topic` become GitHub topics. |
| `continue <arxiv_id> [--profile P] [--no-workflow] [--fresh]` | Resume a paper in its existing workspace (cloned from GitHub if it is missing) at its current workflow step. The saved session is reused unless `--fresh` is given, the last run ended below `target_milestone`, or it was interrupted. |
| `iterate <arxiv_id> [--profile P] [--gpu-budget H] [--workflow-step N]` | Start a new agent session on the paper, at its current step or at step `N` (0-based). |
| `verify <arxiv_id> [--profile P] [--gpu-budget H] [--workflow-step N]` | Like `iterate`, with the verifier role. Default profile: the paper's current workflow step, else `worker.profile`. |
| `cancel <arxiv_id>` | Remove a queued paper, or flag an active one. An active paper stops when its current agent turn or compute job returns; its work is then committed and pushed. |
| `status` | Queue size, active papers and the caps. |
| `papers` | All papers with phase, milestone and status. |
| `cap [bucket [value]] [--clear]` | Show, set or clear a runtime cap. Buckets: `total`, `working`, `overflow`, `admitted`, `building`, `installing`. Lowering a cap only stops new admissions. Overrides are lost on restart. |
| `logs <arxiv_id>` | Print the paper's recorded state and the agent turns of its last iteration. |
| `reset <arxiv_id>` | Delete the paper's workspace and Swarm state. Does not touch the GitHub repository. Do not use on an active paper. |
| `reset-all [-y]` | Delete every entry in `workspace_base_dir`, all paper state and `daemon.log`. Asks for confirmation unless `-y`. |

## Dashboard and REST API

`http://<dashboard_host>:<dashboard_port>/` shows the queue, the active papers
(profile, iteration, milestone, state, budget, elapsed time) and the finished
ones, refreshing every 10 seconds.

| Method and path | Purpose |
|---|---|
| `GET /api/status`, `GET /api/workers` | Counters and caps. |
| `GET /api/dashboard` | Status and paper list in one call. |
| `GET /api/papers`, `GET /api/papers/<id>` | Paper list, one paper. |
| `POST /api/papers` | Submit: `arxiv_id`, optional `gpu_budget`, `date`, `topics`, `profile`, `workflow`. |
| `POST /api/papers/<id>/continue` | `profile`, `no_workflow`, `fresh`, `queue_front`. |
| `POST /api/papers/<id>/iterate` | `profile`, `role`, `gpu_budget`, `workflow_step`, `workflow`, `queue_front`. |
| `POST /api/papers/<id>/cancel` | Cancel. |
| `POST /api/papers/<id>/recover`, `POST /api/recover-all` | Resume papers interrupted by a restart. |
| `POST /api/pause-all` | Record resume information for all papers. |
| `GET`, `POST`, `DELETE /api/caps` | Runtime caps. |

The API has no authentication.

## Files Swarm reads and writes

- `$SWARM_DIR` (default `infrastructure/swarm/.swarm/`, gitignored):
  `daemon.pid`, `daemon.log`, `nexus.log`, and per paper
  `papers/<arxiv_id>/meta.json` (state), `manifest.json` (all iterations),
  `trajectory.json` (last iteration), `setup_snapshots/setup_iterNNN.sh`.
- `<workspace_base_dir>/<arxiv_id>/`: the environment being built. Besides the
  template files it holds `paper/`, `jobs/<run_id>/` (job logs written by
  Nexus), `logs/container_build.log`, `logs/setup.log`, `trajectory/` (raw agent
  output), `manifest.json`, the compiled `README.md`,
  `environment/container.sif`, `environment/overlay.img` and, when the workflow
  ends, `environment/container.final.def`.
- Caches: `container_cache_dir` (`<sha256>.sif`, `base_sifs/`), the sibling
  `setup_overlays/`, `uv_cache_dir`, `apptainer_cache_dir`, and copies of the
  agent container in `local_tmp_dir`.
- GitHub: one private repository per paper. Its description is
  "`<milestone>: <title>`", its homepage the arXiv page, and its topics are
  `milestone-<name>`, the paper's domain (from `briefing/overview.md`),
  `date-<date>`, the `--topic` values, `iteration-<N>-complete`, and `done`
  after the last step. `scripts/retag_github_topics.py` rebuilds this metadata
  from the local state if an update was missed.

Commits: Swarm commits as `swarm <swarm@reproducer>`; agents commit as
`<scaffold>-<model>-<effort> worker`. Agents cannot push (a `git` wrapper in
their container turns `git push` into a no-op). Swarm commits locally when a
job is submitted, checked, finished or cancelled, and pushes at the start and
end of each iteration. Large job logs (above 5 to 10 MB) and trajectory files
(above 95 MB) are committed as their tail; other files above 50 MB are left
out of the commit.

## Tests

```bash
pip install -e "infrastructure/swarm[test]"
cd <repository root>
python -m pytest infrastructure/swarm/tests
```

Run pytest from the repository root. The tests need `git`, but no cluster, no
Apptainer, no network and no local config file. State paths are redirected to
temporary directories. A fixture that applies to every test refuses HTTP
requests and every program other than `git` and `cp`, and fails the test that
reaches one, so a test cannot pull an image, signal a process or talk to a
running Nexus.

The tests cover configuration loading and the example config, paper
resolution, workspace creation, capacity accounting, recovery, container
definition parsing, the CLI, the REST API and
`scripts/retag_github_topics.py`. `tests/test_orchestration.py` takes one
paper through the real loop (workspace, agent turns, job submission,
validation gate, commits, pushes to a local bare repository, workflow gate and
revert) with a scripted agent, a fake Nexus and stand-ins for the container
build and the overlay setup. Real agent CLIs, real container builds and a real
Nexus are not covered.

## Operational notes

Starting the daemon has side effects beyond Swarm's own state. On every start
it:

- **kills every `apptainer build` and `apptainer pull` process of the current
  user** (SIGKILL to the process group), to clear builds orphaned by a previous
  daemon. Builds you started yourself for other purposes are killed too;
- deletes `<local_tmp_dir>/swarm_overlays`;
- copies the agent container to `local_tmp_dir`;
- probes `nexus.url` once with a 3 second timeout. **If there is no answer it
  sends SIGTERM to every process listening on that port and starts a Nexus
  daemon** as a child process, bound to `nexus.autostart_host` (loopback by
  default) on the port from `nexus.url` (39000 if the URL names none), with
  output in `$SWARM_DIR/nexus.log`. A slow but healthy Nexus daemon is
  therefore restarted, which also affects other clients of that daemon, such as
  the Evaluation harness. The child is not stopped by `swarm stop`;
- marks papers that were queued or active as `paused`;
- deletes Nexus projects named `swarm-*` that have no running or queued job;
- sets `APPTAINER_CACHEDIR` and, if Apptainer's blob cache index is corrupt,
  truncates it to its valid part (a backup is kept next to it).

Importing the scheduler also removes `APPTAINER_DOCKER_USERNAME`,
`APPTAINER_DOCKER_PASSWORD` and their `SINGULARITY_` variants from the daemon's
environment, because Apptainer would send them to every registry.

Other notes:

- `daemon.log` is written at debug level, including one line per API request,
  and is never rotated.
- Nothing is evicted from the container, base-image and `setup.sh` caches.
  Each cached `setup.sh` result is a sparse 30 GB ext3 image.
- If a push is rejected because the history contains a file above GitHub's
  100 MB limit, Swarm removes that path from the whole history with
  `git-filter-repo` and pushes again. Without `git-filter-repo` on `PATH` the
  iteration ends as `stalled`.
- For base images that many papers share you may place pre-built images with
  common packages installed in `<container_cache_dir>/base_sifs/_derived_sifs/`
  (`cuda12.1-runtime-common.sif`, `python3.11-slim-common.sif`,
  `ubuntu22.04-common.sif`); builds on those bases then start from them. No
  recipes are shipped, and a container built this way can contain packages its
  `container.def` does not install.
- One daemon per `SWARM_DIR`. Use a different `SWARM_DIR` and `dashboard_port`
  to run a second instance.

## Security notes

- The dashboard and the Nexus daemon Swarm may start are unauthenticated and
  bind to loopback by default. Anyone who can reach the dashboard can start
  agents that act with your model and GitHub credentials.
- Agents are autonomous and receive credentials:
  - the whole `~/.git-credentials` file is copied into every workspace;
  - `~/.claude/.credentials.json` or `~/.codex/auth.json` is copied into the
    workspace before every turn;
  - the agent process inherits the daemon's environment, except
    `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and Claude Code's own session
    variables. A key exported in your shell is therefore not used for the
    agent; put it in a profile's `env_vars`.
  - Compute jobs see the same workspace files.
- Agent output is committed to the paper's repository. Before it is written,
  GitHub tokens, HuggingFace tokens and credentials embedded in URLs are
  redacted; other key formats are not. A pre-commit hook, installed together
  with the git credentials when a workspace is set up, blocks commits
  containing those token shapes, files above 95 MB, installed-package
  directories, and symlinks that point outside the repository. Repositories
  are created private; review them before changing that.
- `swarm reset-all` deletes everything in `workspace_base_dir`.

## Limitations

- GitHub is required in practice, with an organisation as owner and a token in
  `~/.git-credentials` (see [Requirements](#requirements)).
- A failed validation or push marks the iteration `stalled` but does not stop
  the workflow, and a validator timeout counts as a pass
  (see [Validation](#validation)).
- Milestone names outside the ladder are silently recorded as `none`; the
  prompt meant to ask the agent for a valid name is never sent.
- `nexus.job_time_limit` is only told to the agent. The limits in force are the
  Nexus project's job timeout (`project_defaults.default_job_timeout` in the
  Nexus config, one hour by default; see `infrastructure/nexus`) and the
  walltime of the Hive resource. Set them to match.
- The agent prompts describe one deployment shape: agents on a host with
  internet access, compute nodes without it, GPUs assigned by the cluster,
  test resources limited to one hour, and shared dataset and model
  directories. They are not generated from the configuration.
- `environment/container.final.def` is written to the local workspace after
  the last push and is hidden from git, so it is not part of the pushed
  repository.
- Only the `claude-code` and `codex` scaffolds exist.
- The title lookup parses the whole `paper_index.json` on first use.
- `swarm logs` shows only the last iteration; earlier turns are in the
  workspace's `trajectory/` directory and the per-iteration statistics in
  `manifest.json`.
