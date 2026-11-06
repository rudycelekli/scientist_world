# Worker

Worker is the agent scaffold interface of Scientist World's build pipeline. It
wraps an agent command-line tool (Claude Code or Codex) behind one Python
class, so that Swarm (`infrastructure/swarm`) can drive a builder agent turn by
turn without knowing which tool runs underneath. The directory
`infrastructure/worker` also ships everything the builder agents are given:

- the **prompts** for each builder role and each event (`prompts/`);
- the **workspace template** copied into every paper's workspace
  (`workspace_template/`), including the validator `validate.py`;
- the definition of the **agent container** (`container/reproducer.def`).

Worker has no daemon and no command of its own. Swarm creates one Worker per
paper iteration and calls it from the paper's thread. The Evaluation harness
(`evaluation`) does not use this package, but it reads
`workspace_template/.gitignore`.

## Layout

| Path | Purpose |
|---|---|
| `worker.py` | `Worker` base class: lifecycle, event handlers, reading `action.yaml`, trajectory files with secret redaction, the Apptainer command, the process environment. |
| `claude_code.py` | `ClaudeCodeWorker`: runs the `claude` CLI. |
| `codex.py` | `CodexWorker`: runs the `codex` CLI. |
| `parsing.py` | Parsers for `action.yaml`, Claude Code's stream-json output and `MILESTONES.md`. |
| `prompts.py` | Builders that load a template from `prompts/` and fill its placeholders. Also holds the per-scaffold "Tools" section inserted into the role prompts. |
| `prompts/` | Prompt templates (see [Prompts](#prompts)). |
| `workspace_template/` | Starting content of every workspace (see [Workspace template](#workspace-template)). |
| `container/reproducer.def` | Apptainer definition of the agent container. |
| `tests/` | Unit tests. |

## Requirements and install

- Python 3.10 or later and `pyyaml`.
- Worker is not a separate distribution. It is imported as
  `infrastructure.worker`, which works when the repository root is on
  `sys.path`. Swarm adds the root itself; in your own scripts run from the
  repository root or set `PYTHONPATH` to it.
- At least one agent CLI on the host, logged in: `claude` and/or `codex`.
- For container mode (the mode Swarm is meant to run in): Apptainer with
  cgroups v2 memory delegation (`--memory`), and the agent container:

  ```bash
  apptainer build reproducer.sif infrastructure/worker/container/reproducer.def
  ```

  The image is Ubuntu 24.04 with git, Python 3, pip, build tools, `uv`, and a
  `pip` wrapper that forwards `install`, `uninstall`, `freeze`, `list`, `show`
  and `check` to `uv pip`. It contains neither Node.js nor an agent CLI: those
  are bind-mounted from the host at run time, so the host's `claude` must be a
  self-contained executable, and `codex` must live under a Node.js prefix that
  contains `bin/node`.

  To check that the host accepts the options Worker always passes:

  ```bash
  mkdir -p /tmp/ws && apptainer exec --contain --writable-tmpfs --no-mount bind-paths \
      --home /tmp/ws:/home/user --pwd /home/user --memory 8G reproducer.sif true
  ```

## Interface

```python
from infrastructure.worker import ClaudeCodeWorker   # or CodexWorker

worker = ClaudeCodeWorker(
    scaffold="claude-code",
    model="claude-opus-4-6",
    config={"container_sif": "/path/to/reproducer.sif", "effort": "high"},
)
worker.set_paper_and_workspace(
    {
        "paper_md_path": "/path/to/workspaces/2512.01152/paper/2512.01152.md",
        "paper_images_dir": "/path/to/workspaces/2512.01152/paper/images",
        "paper_pdf_path": "/path/to/workspaces/2512.01152/paper/2512.01152.pdf",
    },
    "/path/to/workspaces/2512.01152",
)
worker.set_resources(
    gpu_budget=30.0,
    max_gpus=2,
    job_time_limit="24:00:00",
    shared_datasets_path="/path/to/shared/datasets",
    shared_models_path="/path/to/shared/models",
)

worker.reproduce()                  # first turn: reproduce_start.md
action = worker.get_last_action()   # TurnAction(action="submit", gpus=1, ...) or None
```

| Method | Purpose |
|---|---|
| `set_paper_and_workspace(paper, workspace_path)` | Paper file paths (inside the workspace) and the workspace directory. |
| `set_resources(gpu_budget, max_gpus, job_time_limit, shared_datasets_path, shared_models_path, uv_cache_dir)` | Values shown in the first prompt and directories to mount. |
| `reproduce()` | Prepare the workspace and send `reproduce_start.md`. Used for reproducer and verifier iterations. |
| `start(prompt)` | Prepare the workspace and send a given first prompt. Used for designer and packager iterations. |
| `chat(message)` | Send a follow-up prompt in the same session. |
| `on_job_progress(run_id, stdout_path, stderr_path, elapsed_min)` | Ask the agent whether a running job should continue. Returns `"cancel"` if the reply contains "cancel", else `"continue"`. |
| `on_job_completion(...)`, `on_job_cancelled(...)`, `on_budget_exhausted(...)`, `on_container_build_failure(...)` | Send the matching event prompt. |
| `get_last_action()` | The `TurnAction` parsed from `action.yaml` after the last turn, or `None`. |
| `get_status()`, `get_error()` | `idle`, `working`, `done` or `error`, and the error text. |
| `get_context()` | Status, session id, last action, turn count, token and tool statistics, milestone from `MILESTONES.md`. |
| `save_trajectory(dir)` | Write `trajectory.json`: one record per turn plus cumulative statistics. |
| `request_cancel()`, `kill_subprocess()`, `terminate()` | Abort retry waits, kill the agent process tree, mark the worker done. |

Every method that sends a prompt blocks until the agent's turn ends.

### End-of-turn protocol

The agent does not answer in a fixed text format. Before its turn ends it edits
`action.yaml` in the workspace root, and Worker reads that file after the turn:

```yaml
action: submit        # submit | done  (the template starts with "pending")
gpus: 1               # 0 = CPU job
test: false           # true = short test job
milestone: method_runs
notes: "First training run."
```

- `submit` asks Swarm to run `scripts/run.sh` as a compute job; `done` ends the
  iteration.
- If the file is missing, is not valid YAML, or `action` is neither `submit`
  nor `done`, Worker asks the agent to fix it, up to `action_retries` times
  (default 2).
- `milestone` is normalised: case, spaces and hyphens are ignored, common
  variants are mapped (for example `core_claims` to `core_claim`), and any
  other unknown name becomes `none`.

## Scaffolds

### Claude Code (`scaffold: claude-code`)

```
claude --print --verbose --output-format stream-json --model <model>
       --allowedTools Bash,Read,Write,Edit,WebSearch,WebFetch,Glob,Grep
       --strict-mcp-config [--max-turns <max_turns>] [--resume <session_id>]
```

- The prompt is written to stdin. Output is streamed line by line to
  `trajectory/turn_NNNN_raw.jsonl`.
- `--strict-mcp-config` keeps the CLI from loading the connectors of the
  account it is logged in with.
- Environment: `CLAUDECODE`, every `CLAUDE_CODE_*` variable, `ANTHROPIC_API_KEY`
  and `OPENAI_API_KEY` are removed, so the CLI uses its stored login. Then
  `TMPDIR` and `CLAUDE_CODE_TMPDIR` point to a per-workspace directory,
  `CLAUDE_CODE_EFFORT_LEVEL` is set from `effort`,
  `CLAUDE_CODE_AUTO_COMPACT_WINDOW` from `compact_token_limit`, and `env_vars`
  are applied last.
- Workspace setup: `.claude/settings.json` with permission rules (so that the
  CLI never waits for approval), and `CLAUDE.md` as a symlink to `protocol.md`.
- A turn is killed after 20 minutes without output or after
  `turn_wall_timeout` seconds (default 3600) in total.
- Transient API errors (rate limits, overload, connection errors) are retried
  up to `api_retries` times (default 8) with growing delays, for at most
  `max_total_retry_sec` (default 900). Hard usage limits are not retried: if
  `api_keys` holds more than one key the next one is written to
  `env_vars.ANTHROPIC_AUTH_TOKEN` and the turn is repeated, otherwise the
  worker ends in `error` and Swarm may fall back to another profile.

### Codex (`scaffold: codex`)

```
codex exec --experimental-json --model <model>
      --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check
      -C <workspace> -c model_auto_compact_token_limit=<compact_token_limit or 120000>
      [-c model_reasoning_effort="<effort>"] [resume <session_id>] -
```

- The prompt is written to stdin; output is saved to
  `trajectory/turn_NNNN_raw.jsonl` when the turn ends.
- Workspace setup: `AGENTS.md` as a copy of `protocol.md`.
- A turn is killed after `idle_timeout` seconds without output (default 1200)
  and repeated up to `idle_retries` times (default 2). Transient API errors
  are retried as for Claude Code.
- A "context window exceeded" error ends the session with status `done`.
- Codex's own sandbox is disabled; isolation comes from the container.

### Worker configuration

The `config` dictionary given to a worker. Swarm fills it from the selected
profile and its own settings.

| Key | Default | Meaning |
|---|---|---|
| `container_sif` | `None` | Agent container. `None` runs the CLI directly on the host. |
| `container_memory_limit` | `8G` | `apptainer exec --memory`. |
| `container_nice` | 19 | `nice -n` value. |
| `container_cpu_threads` | 2 | Thread cap for OMP, MKL, OpenBLAS, NumExpr, VecLib. |
| `effort` | `None` | Reasoning effort. |
| `compact_token_limit` | `None` | Auto-compaction threshold in tokens. |
| `env_vars` | `{}` | Extra environment variables for the agent process. |
| `api_keys` | `[]` | Keys to rotate through on hard usage limits. |
| `actual_model` | unset | Model name used in the git identity when `model` is an alias. |
| `action_retries` | 2 | Fix requests when `action.yaml` is invalid. |
| `api_retries`, `max_total_retry_sec` | 8, 900 | Retry policy for transient API errors. |
| `max_turns` (Claude Code) | 200 | `--max-turns`. |
| `allowed_tools`, `turn_wall_timeout`, `timeout` (Claude Code) | see above, 3600, `None` | Tool list, wall-clock limit of a turn, limit for the final process wait. |
| `idle_timeout`, `idle_retries` (Codex) | 1200, 2 | Output inactivity limit and retries. |

## Container mode

With `container_sif` set, the agent CLI runs inside Apptainer:

```
nice -n <nice> apptainer exec --contain --writable-tmpfs --no-mount bind-paths
    --home <workspace>:/home/user --pwd /home/user --memory <limit>
    --env OMP_NUM_THREADS=<n> ...                        # thread caps
    <scaffold binds>                                     # agent CLI, CA certificates
    --bind <shared_datasets>:/home/user/shared/datasets:ro
    --bind <shared_models>:/home/user/shared/models:ro
    --bind <uv_cache>:<uv_cache> --env UV_CACHE_DIR=...  # if configured
    --bind <workspace>/.claude-tmp:/tmp/claude-code-tmp
    --bind /dev/null:/home/user/<hidden file>:ro ...
    --bind <workspace>/.empty_<dir>:/home/user/<hidden dir>:ro ...
    --bind <workspace>/validate.py:/home/user/validate.py:ro
    --bind <workspace>/.git-shims:/opt/git-shim:ro
    <container_sif> <agent command>
```

- The agent sees the workspace as `/home/user` (also its home directory), the
  two shared directories read-only, and a private `/tmp`. Admin-configured
  bind paths of the host are not mounted. The network is not isolated.
- Hidden from the agent: `manifest.json`, `README.md`, `protocol.md`,
  `environment/container.final.def` (bound to `/dev/null`) and the directories
  `trajectory/` and `.claude-tmp/` (shown empty). Swarm also lists these paths
  in `.git/info/exclude` and removes the files from the git index, so the
  agent's `git add -A` is not affected; Swarm commits them itself.
- `validate.py` is read-only for the agent.
- `git push` is disabled: `/opt/git-shim/git` is first on `PATH`, prints a
  notice for `push` and forwards every other command to `/usr/bin/git`.
- Git identity of agent commits: `<scaffold>-<model>-<effort> worker`,
  `<scaffold>@reproducer`.
- Credentials are copied into the workspace before every turn:
  `~/.claude/.credentials.json` to `.claude/.credentials.json`, or
  `~/.codex/auth.json` to `.codex/auth.json`. The template `.gitignore`
  ignores both directories.

Without `container_sif` the CLI runs on the host as your user. Only
`.claude/settings.json` then restricts the file tools, and shell commands are
not restricted at all. Use this mode for development only.

## Prompts

All templates are `str.format` templates in `prompts/`; `prompts.py` has one
builder per template. They are the method of the paper and are shipped
unchanged.

| Template | Sent when |
|---|---|
| `reproduce_start.md` | First turn of a reproducer or verifier iteration. |
| `on_iterate.md` | Second turn of every reproducer iteration except iteration 1. |
| `on_verify.md` | Second turn of a verifier iteration. |
| `on_designer_stage_1.md`, `on_designer_stage_2.md`, `on_designer_stage_3.md` | First turn of the three designer stages (write the train/test evaluation split, review it, run it). |
| `on_designer.md` | First turn of the single-stage designer role. |
| `on_packager.md` | First turn of the packager. |
| `on_continue.md` | A saved session is resumed. |
| `on_job_progress.md` | A running job reaches a checkpoint. |
| `on_job_completion.md` | A job completed, failed or was cancelled. |
| `on_job_cancelled.md` | The agent cancelled a job. |
| `on_container_build_failure.md` | The container build or `setup.sh` failed. |
| `on_budget_exhausted.md` | The iteration's GPU budget is used up (followed by `on_wrap_up.md`). |
| `on_wrap_up.md` | The iteration is about to end. |
| `on_push_forward.md` | The agent signalled `done` below the target milestone. |
| `on_validation_failure.md` | `validate.py` failed at the end of the iteration. |
| `on_invalid_milestone.md` | A milestone name outside the ladder (not reachable at present, see [Limitations](#limitations)). |

Two prompt texts are Python strings instead of files: the "Tools" sections in
`prompts.py` (one for Claude Code, one for Codex, inserted into
`reproduce_start.md` and the designer and packager prompts) and the request to
fix an invalid `action.yaml` in `worker.py`.

## Workspace template

`workspace_template/` is copied to `<workspace_base_dir>/<arxiv_id>/` when a
paper is first built. The agent turns it into the environment.

| Path | Purpose |
|---|---|
| `protocol.md` | Short rule sheet for the agent. Becomes `CLAUDE.md` (symlink) or `AGENTS.md` (copy). |
| `action.yaml` | End-of-turn action file. |
| `MILESTONES.md`, `PROGRESS.md`, `CHECKLIST.md`, `VERIFICATION.md` | Documents the agent maintains. |
| `README.md` | Empty. Swarm overwrites it with a page compiled from `briefing/` and the build history. |
| `briefing/` | `overview.md`, `problem.md`, `method.md`, `evaluation.md`: the description of the environment. |
| `scoring/` | `TARGETS.md`, `EXPERIMENTS.md`, `CONSTRAINTS.md`, `DIRECTION.md`, and examples of `reference.json` and `scores.json`. |
| `scripts/` | `run.sh` (the compute job Swarm submits), `evaluate.sh`, `method.sh`, `baseline.sh`, `reproduce.sh`, `download.sh`. |
| `environment/` | `container.def` (OS-level packages) and `setup.sh` (Python packages). |
| `method/`, `baseline/`, `eval/`, `data/`, `data/downloads/`, `notes/`, `paper/`, `tmp/` | Empty directories the agent fills. |
| `examples/` | Example `action.yaml` content and an example `run.sh`. |
| `validate.py` | Validator, standard library only. Run by the agent and, at the end of every iteration, by Swarm. Modes: default, `--reference-only`, `--compare`, `--during-design`, `--post-design`, `--post-reproducer`. |
| `.gitignore` | Keeps weights, containers, caches, credentials and scaffolding out of the paper's repository. |

The template's own `.gitignore` also applies inside this repository. It
ignores `data/downloads/`, so `data/downloads/.gitkeep` stays tracked only if
it is added with `git add -f`.

## Trajectories and redaction

- `trajectory/turn_NNNN_raw.jsonl`: raw CLI output of each turn.
- `trajectory/session.json`: the session id, for resuming.
- `trajectory.json` (written by `save_trajectory`): scaffold, model, turn
  count, cumulative statistics, milestone, and per turn the final text,
  statistics, errors, return code and duration.

Before any of this is written, text matching these patterns is replaced:
GitHub tokens (`ghp_...`, `github_pat_...`), HuggingFace tokens (`hf_...`) and
`user:password@` in URLs. No other secret format is recognised.

## Adding a scaffold

1. Subclass `Worker` and implement the abstract hooks: `_config_defaults`,
   `_initial_cumulative_stats`, `_invoke`, `_build_command`,
   `_setup_workspace`, `_copy_credentials`, `_add_scaffold_container_binds`,
   `_configure_env`, `_accumulate_stats`.
2. Add a "Tools" section for it to `TOOLS_BY_SCAFFOLD` in `prompts.py`.
3. Register the scaffold name in `SUPPORTED_SCAFFOLDS`
   (`infrastructure/swarm/swarm/config.py`) and in `_create_worker`
   (`infrastructure/swarm/swarm/scheduler.py`).
4. Export the class from `__init__.py` and add tests.

## Tests

```bash
cd <repository root>
python -m pytest infrastructure/worker/tests
```

The tests need `pytest` and `pyyaml`, no agent CLI, no Apptainer and no
network. They cover `action.yaml`, stream-json and milestone parsing, command
and environment construction for both scaffolds, workspace setup, and
rendering of every prompt template. `HOME` and the git identity variables are
redirected, so the tests never read your credentials, and a fixture that
applies to every test refuses to start any program other than `git`, so a test
cannot run a real agent. Running a real agent in a real container is not
covered.

## Security notes

- An agent is an autonomous program with a shell. In container mode it can
  read and write the workspace, read the shared directories, and use the
  network. It can read the credentials that are copied into the workspace: the
  agent CLI login (see above) and, when Swarm set the workspace up, a copy of
  your `~/.git-credentials`.
- The agent process inherits the environment of the process that starts it,
  apart from the variables listed under [Scaffolds](#scaffolds). Do not start
  Swarm from a shell that holds unrelated secrets.
- `.claude/settings.json` is a convenience so that the CLI does not prompt. It
  is not a security boundary; the container is.
- Trajectories are committed to the paper's repository by Swarm. Redaction is
  limited to the patterns listed above.

## Limitations

- In container mode `protocol.md` is hidden from the agent, and for Claude
  Code `CLAUDE.md` is a symlink to it, so the file the CLI loads automatically
  is empty there. Codex receives a real copy as `AGENTS.md`.
- The prompts tell agents to download shared datasets and models into the
  shared directories, which are mounted read-only.
- Job event prompts contain the host paths of the job logs. In container mode
  the same files are under `/home/user/jobs/<run_id>/`.
- `on_job_progress` treats any reply that contains the word "cancel" as a
  request to cancel.
- Unknown milestone names are turned into `none` by the parser, so Swarm's
  check for invalid names, and `on_invalid_milestone.md`, never trigger.
- Key rotation always writes `ANTHROPIC_AUTH_TOKEN`; it has no effect for
  Codex profiles.
- The prompts and `protocol.md` describe one deployment shape (agent host with
  internet access, compute nodes without it, one-hour test resources). They
  are fixed text, not generated from configuration.
