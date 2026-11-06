# Evaluation

The evaluation harness of Scientist World. It runs an AI agent (a
"scientist") on a built paper environment, records everything the agent
does, has an independent grader session re-score the result on held-out
data, and normalizes the scores so that they can be averaged across papers
(Frontier-Normalized Improvement, FNI).

Two packages live in this directory:

| Package | Purpose |
|---|---|
| `evaluation/` | the harness: runner, agent loop, grader, normalization, dashboard, publishing |
| `dataset_builder/` | turns finished runs into the traces dataset and runs LLM judges over finished runs |

`idea_impl_eval.py` is a separate driver for the idea-implementation
ablation (see below). `docs/scoring.md` describes the scoring in full.

## How it fits with the other components

| Component | Role for the harness |
|---|---|
| `sciworld` (the Scientist World CLI, `cli/`) | Clones a paper's environment repository and lays out a run workspace (`sciworld.research(setup_only=True)`); builds the agent container image; provides the container command that isolates the agent (`sciworld.launcher`). |
| Nexus (`infrastructure/nexus`) | Every compute job of a run goes through a Nexus project created for that run. Nexus enforces the GPU-hour budget and the per-job GPU cap and pauses long jobs for check-ins. |
| Hive (`infrastructure/hive`) | The compute pool behind Nexus. The harness never talks to Hive; it only names Hive resources in its config. |
| Swarm (`infrastructure/swarm`) | Builds the environments. The harness uses the same profile schema and the same container cache keys as Swarm, and `sw-eval seed` can copy the `profiles` and `nexus` blocks of a Swarm config. |

The harness process runs on one host. Agents run there too, each turn in a
fresh Apptainer container; only the jobs an agent submits run on the pool.

## Requirements

- Linux, Python 3.10 or newer.
- The `sciworld` package, installed from `cli/`, and an agent container
  image built with it (`sw init`, `sw build`).
- Apptainer. The agent container uses `--memory` (a memory cgroup). Paper
  containers are built with `apptainer build`; dependency overlays with
  `apptainer exec --fakeroot`, or through a rootless variant when user
  namespaces are disabled. `truncate` and `mkfs.ext3` are needed for
  overlays.
- `git`. Every run workspace is a local repository. Publishing needs git
  2.31 or newer.
- A reachable Nexus daemon in front of a Hive pool. The data directory must
  be visible, at the same path, on the nodes that run jobs.
- For each scaffold you use, a working login on the host: the harness
  copies the agent CLI's credential files from the host user's home
  (`~/.claude`, `~/.codex`, ...) into the run workspace before every turn.
  Models served through a gateway take their endpoint and key from the
  profile instead. `claude-tmux` also needs `tmux`.
- Optional: `matplotlib` for the dashboard's charts; `datasets` and
  `huggingface_hub` for the dataset builder; the `codex` CLI for the
  dataset builder's consolidation and judge scripts.

## Install

From the repository root:

```bash
pip install -e cli                       # sciworld
pip install -e evaluation                # this package
pip install -e "evaluation[charts]"      # optional: dashboard charts
pip install -e "evaluation[dataset]"     # optional: dataset builder
```

This installs the `sw-eval` command. Without installing this package, the
same commands run as `PYTHONPATH=evaluation python -m evaluation.cli ...`
from the repository root (`flask`, `pyyaml` and `sciworld` must be
importable).

## Configuration

The configuration file is `evaluation/evaluation/config.yaml`. It is not
tracked (it may hold API keys); `evaluation/evaluation/config.example.yaml`
is the documented template:

```bash
cp evaluation/evaluation/config.example.yaml evaluation/evaluation/config.yaml
```

Set `EVALUATION_DIR=/abs/dir` to read `<dir>/config.yaml` instead. If you
already run Swarm, `sw-eval seed` writes a config from the defaults plus
the `profiles` and `nexus` blocks of Swarm's config; review it afterwards.

A run refuses to start, with a message listing what is missing, until
`data_dir`, `agent_sif`, `nexus.resources.gpu` and
`nexus.resources.gpu_test` are set. The example file ships placeholder
values for them, which you must replace. At least one profile is needed.

### Reference

| Key | Default | Meaning |
|---|---|---|
| `data_dir` | none, required | Root of all state: `envs/`, `cache/`, `normalization/`. |
| `agent_sif` | none, required | Agent container image built by the sciworld CLI. |
| `profiles` | `{}` | Model profiles, keyed by name (next table). |
| `max_concurrent` | 10 | Runs in flight at once (queue of `bench` and the dashboard). |
| `profile_concurrency` | `{}` | Per-profile cap on runs in flight, `{profile: n}`. |
| `max_turns` | 200 | Agent turns in the main loop of a run. |
| `max_retries` | 1 | Further attempts after a transient agent failure. |
| `profile_cohort` | `[]` | Profiles that define the normalization coefficients. Empty: every profile. |
| `profile_cohort_exclude` | `[]` | Profiles removed from the cohort. |
| `container_cache_dir` | `""` | Shared cache of built paper containers; overlays are cached in `../setup_overlays` next to it. Empty: no cache. |
| `uv_cache_dir` | `""` | uv cache bound into the container while `setup.sh` runs. |
| `local_tmp_dir` | `/tmp` | Local disk used while building overlays. |
| `workspace_gitignore` | `""` | `.gitignore` template for run workspaces. Empty: `infrastructure/worker/workspace_template/.gitignore` of this repository. |
| `agent_memory` | `8G` | Memory hard cap of the agent container. |
| `agent_cpu_threads` | 2 | BLAS/OpenMP thread cap exported into the agent container. |
| `agent_nice` | 19 | `nice` value of the agent container. |
| `host` | `127.0.0.1` | Dashboard bind address. |
| `port` | 38582 | Dashboard port. |
| `cohort_file` | `""` | JSON list of paper ids, relative to `data_dir`. When set, the dashboard offers and accepts only these papers. |
| `dashboard.paper_groups` | `{}` | `{label: [paper ids]}`. Each group adds a coverage strip and a pair of charts restricted to its papers. |
| `dashboard.model_order` | `[]` | Profile names shown first, in this order, as matrix columns. |
| `claude_code_max_retries` | 50 | Exported to the claude CLI as `CLAUDE_CODE_MAX_RETRIES`. |
| `api_timeout_ms` | 3000000 | Exported to the claude CLI as `API_TIMEOUT_MS`. |
| `claude_code_max_output_tokens` | 64000 | Exported to the claude CLI as `CLAUDE_CODE_MAX_OUTPUT_TOKENS`. |
| `claude_oauth_refresh_model` | `claude-haiku-4-5-20251001` | Model of the one-line host-side `claude --print` call that refreshes a Claude login token close to expiry. |
| `claude_tmux_binary` | `""` | `claude-tmux` only: host path of the `claude` binary bound into the agent container. |
| `checkpoint_interval` | 1200 | Seconds between check-ins on a running job. |
| `checkpoint_timeout` | 1800 | Seconds Nexus waits for the agent's decision at a check-in. |
| `warm_download_timeout` | 7200 | Time limit of `scripts/download.sh` during `warm`. |
| `nexus.url` | `http://localhost:39000` | Nexus daemon. |
| `nexus.gpu_budget_hours` | 30.0 | GPU-hour budget of one run. |
| `nexus.max_gpus` | 2 | GPUs one job may request. |
| `nexus.resources.gpu`, `.gpu_test`, `.cpu`, `.cpu_test` | `[]` | Hive resource names per job category. They must match the resource names of the Hive configuration. `gpu` and `gpu_test` are required; `gpu_test` is used when the agent marks a job `test: true` and may repeat the `gpu` list. |
| `verifier.enabled` | `false` | Grade every completed run (see Independent grading). |
| `verifier.grader_profile` | `""` | Profile used as grader. Required when grading; there is no default. |
| `verifier.max_turns` | 12 | Turns of one grader session. |
| `verifier.gpu_budget_hours` | 8.0 | GPU-hour budget of one grader session. |
| `verifier.max_relaunches` | 2 | Total attempts at a grader session that crashes. |
| `verifier.publish` | `true` | Also push the graded state (only if publishing is enabled). |
| `publish.enabled` | `false` | Push completed runs to GitHub (see Publishing). |
| `publish.github_org` | `""` | Organisation that holds the per-paper repositories. |

### Profiles

A profile names one (scaffold, model, effort) combination. Runs,
workspaces and branches are identified by the profile name. The schema is
the same as Swarm's.

| Field | Meaning |
|---|---|
| `scaffold` | `claude-code`, `codex`, `kimi`, `gemini` or `claude-tmux` |
| `model` | model id passed to the agent CLI |
| `effort` | reasoning effort passed to the CLI (optional) |
| `compact_token_limit` | context size at which the CLI compacts (optional) |
| `env_vars` | extra environment variables for the agent container, for example a gateway endpoint and its key (optional) |
| `api_keys` | several keys for one gateway, assigned round-robin as `ANTHROPIC_AUTH_TOKEN` and rotated when one reports a usage limit (optional) |
| `actual_model` | the real model id when `model` is an alias; used for the dataset's `model` column (optional) |
| `publish_as` | external identity when several profiles serve the same model: used for branch names and counted once in the normalization cohort (optional) |
| `idle_timeout_sec` | seconds without agent output after which a turn is killed and retried; default 1800 (optional) |

### Scaffolds

| Scaffold | Agent CLI | Status |
|---|---|---|
| `claude-code` | `claude --print` (stream-json) | supported |
| `codex` | `codex exec` | supported |
| `kimi` | `kimi --print` | experimental; no token-usage extraction |
| `gemini` | `gemini` CLI | experimental; no reasoning trace in the dataset |
| `claude-tmux` | interactive `claude` driven through tmux (`tmux_driver.py`) | experimental; depends on the screens of one CLI version (`claude_tmux_binary`); no token-usage extraction |

"Experimental" means the scaffold works with the CLI versions it was
written against and has less coverage in the harness (see the notes in
the table); `claude-code` and `codex` are the reference scaffolds.

## Commands

```
sw-eval run <paper> <profile> [--mode innovate|improve]
sw-eval bench --papers ID,ID --models PROFILE,PROFILE [--modes innovate,improve]
sw-eval serve [--host HOST] [--port PORT]
sw-eval warm <paper> [--force]
sw-eval renormalize [--paper ID]
sw-eval publish [--paper ID] [--profile NAME] [--mode MODE] [--dry-run] [--limit N]
sw-eval seed [--overwrite] [--swarm-config PATH]
python -m evaluation.verifier <paper> <profile> <mode> [--grader PROFILE]
        [--max-turns N] [--gpu-budget-hours H] [--no-publish] [--force]
python -m evaluation.usage <run directory> --scaffold claude-code|codex|gemini
```

| Command | What it does |
|---|---|
| `run` | One trial, blocking. Exit code 0 if the run ends `completed`. The default mode is `innovate`. |
| `bench` | Queues every (paper, profile, mode) combination, runs them under the concurrency caps and waits. |
| `serve` | Dashboard plus run queue (blocks). Runs are queued from the page or through `POST /api/queue`. |
| `warm` | Builds a paper's container and overlay once, runs its `scripts/download.sh`, and stores the result in `<data_dir>/cache/<paper>` for all later runs of that paper. Needs network access on the host. |
| `renormalize` | Recomputes the normalization coefficients of one paper or of all, and the per-mode summary. Locked coefficients are left untouched. |
| `publish` | Pushes completed runs found under `data_dir` (requires `publish.enabled`). `--dry-run` reports what would be pushed without contacting GitHub. |
| `seed` | Writes a config from the defaults plus the `profiles` and `nexus` blocks of a Swarm config, and `data_dir` / `agent_sif` from the sciworld settings if they exist. Existing keys win unless `--overwrite`. |
| `evaluation.verifier` | Grades one completed run. Exit codes: 0 `verified`, 2 `unfixable`, 3 `exhausted`, 1 otherwise. |
| `evaluation.usage` | Prints the token usage summed over a run's trajectory. |

The same operations are available from Python:

```python
from evaluation import run, bench, serve
run("<paper id>", "<profile>", mode="improve")
```

## Trial lifecycle

`runner.run_one` does the following for one (paper, profile, mode).

1. **Pre-flight.** The config is checked. If the workspace already holds a
   completed run with both score files it is returned as is. An
   interrupted run with at least one finished turn and no held-out score
   or grader verdict is resumed in place (same workspace, same agent
   session, same GPU accounting). A run whose agent had finished but that
   was never finalized is finalized without relaunching the agent.
   Otherwise the workspace is created afresh. An existing workspace that
   holds scoring files, trajectory turns or at least one GPU-hour of work
   is never wiped automatically: the run fails with a message, and the
   directory has to be moved aside to start over.
2. **Scaffold.** `sciworld.research(setup_only=True)` clones the paper's
   environment to `envs/<paper>/source` (once per paper) and lays out the
   workspace `envs/<paper>/<profile>-<mode>-eval`. The harness then removes
   everything an evaluated agent must not see (the held-out evaluator
   `eval/test`, `scripts/evaluate_test.sh`, reference scores, the paper
   text; in `innovate` mode also the paper's method), replaces `idea.md`
   by an `idea/` folder, writes an example `scripts/run.sh`, and in
   `improve` mode adds `READMEs/train_reference.json` with the scores of
   the starting code. The workspace becomes a local git repository.
3. **Container.** The paper's `environment/container.def` is built to
   `container.sif` and `environment/setup.sh` is run into `overlay.img`,
   or both are copied from the caches. Cached downloads are bound
   read-only.
4. **Nexus project.** A project is created for the run with the GPU-hour
   budget, the per-job GPU cap, the allowed resources and the check-in
   interval.
5. **Main loop** (`loop.run`). Each turn starts the agent CLI in a fresh
   agent container with the workspace as its home directory. The agent
   ends a turn by writing `action.yaml`:

   | `action` | Effect |
   |---|---|
   | `submit` (with `gpus`, optional `test: true`) | If the agent changed `container.def` or `setup.sh`, the container or overlay is rebuilt. The harness submits `bash scripts/run.sh` as a Nexus job in the paper container and waits. The agent is called again when the job ends. |
   | `continue` / `cancel` | Answer to a check-in: every `checkpoint_interval` seconds Nexus pauses the job and the agent is shown the logs. Anything other than `continue` cancels the job. |
   | `done` | Ends the main loop. |

   A missing or malformed `action.yaml` is answered with a corrective
   prompt (the turn still counts). Once the GPU budget is used up, `submit`
   is refused with a prompt that says so. The loop ends at `max_turns`.
   Job logs go to `jobs/<id>/`, each turn's raw output to `trajectory/`,
   and the workspace is committed after every job.
6. **Train validation.** `validate.py` checks, on the host, that
   `scoring/scores_train.json` has a value for every metric of every
   non-baseline method in the paper's own train scores. Gaps are sent back
   to the agent, for up to 5 rounds of at most 30 turns each.
7. **Finalization.** Status is decided on the train scores alone: `failed`
   (no usable `scores_train.json`), `incomplete` (cells still missing) or
   `completed`. Token usage is summed, `.evaluation.json` is written and
   the workspace is committed.
8. **Normalization.** The paper's coefficients are recomputed
   (`docs/scoring.md`). Only graded runs change them.
9. **Publishing** (optional, off by default) of a completed run.
10. **Independent grading** (optional, `verifier.enabled`) of a completed
    run; see the next section.
11. **Cleanup.** The run's `container.sif` and `overlay.img` are deleted
    (they are rebuilt from the cache on resume or grading) and the Nexus
    project is removed.

The held-out (test) score is not produced inside the agent's run. The
code for an in-run test phase (`runner._run_test_phase` and the
`on_test_phase*` prompts) is kept but disabled; the grader produces the
held-out score.

A turn with no agent output for `idle_timeout_sec` is killed and
relaunched, up to two times. A transient failure of the agent CLI
(non-zero exit, kill by the memory cgroup) ends the attempt; the run is
then retried up to `max_retries` times, continuing in place when the
failed attempt left finished turns. A run is not retried once it has
agent-written files under `scoring/`, nor when the provider's safety
filter rejected the conversation.

## Independent grading

A run's own `scores_train.json` is a claim. The verification phase
(`verifier.py`) starts a new session of a different profile (the grader)
in the run's workspace, with its own Nexus project and budget:

1. A mechanical scan (`scoring/verifier_scan.json`) looks for constant
   outputs, byte-identical result files, hand-typed looking values and
   test scores whose job failed. Its findings are leads for the grader.
2. The pristine evaluators, including the held-out one, are restored from
   `source/`.
3. The grader follows `prompts/task_verify.md`: it sets aside the agent's
   result files, re-runs the frozen method on train and test, grades the
   fresh predictions, checks the task's constraints, and may repair
   infrastructure but not method logic.
4. It writes `scoring/verification.json` with one of:

   | Status | Meaning |
   |---|---|
   | `verified` | The checks held. The grader's scores replace `scores_train.json` / `scores_test.json`, and normalization is recomputed. |
   | `unfixable` | The run is not a legitimate attempt at the task. |
   | `exhausted` | The grader worked but reached no verdict, or claimed `verified` without scores or executed jobs. |
   | `incomplete` | The grader session was lost (quota, login, container build, turn budget). Not a verdict; the run can be graded again. |

Only runs with a clean verdict contribute to normalization coefficients
(`verified`, or `audit_clean` set by hand; see `docs/scoring.md`), and the
held-out score of a run exists only once it has been graded. Runs with a
terminal verdict (`verified`, `unfixable`) are not graded again without
`--force`.

## Scoring

`docs/scoring.md` defines the score: per-paper coefficients computed from a
cohort of graded runs, the paper score of any run as a weighted sum of its
raw improvements over the paper's own results, FNI as 100 times that
score, the minimum of three cohort models, dominant-cell pruning and
locking at four models.

## Data directory

```
<data_dir>/
  envs/<paper>/
    source/                       clone of the paper's environment (never modified)
    <profile>-<mode>-eval/        one run workspace (a local git repository)
      .evaluation.json            status, state, timings, usage, publish and verification summary
      action.yaml                 the agent's last signal
      READMEs/ idea/ method/ scripts/ eval/ data/ environment/ reflection.md
      scoring/scores_train.json   written by the agent (replaced by the grader's when verified)
      scoring/scores_test.json    written by the grader
      scoring/verification.json   the grader's verdict
      trajectory/turn_NNNN_raw.jsonl   raw agent output per turn
      jobs/<job id>/              meta.json, stdout.log, stderr.log
    normalization/coeffs_<mode>_<phase>.json
  cache/<paper>/                  container, overlay and downloads from `warm`
  normalization/global_sd_<mode>.json   per-mode summary (informational)
```

Run one harness process per `data_dir`. The dashboard, at startup, deletes
the container image and overlay of every run still marked `running`, on
the assumption that such runs belong to a previous process.

## Dashboard and HTTP API

`sw-eval serve` shows running, queued and finished runs, the per-paper
matrix, coverage, verification status and leaderboards, and lets you queue
runs and change the concurrency caps.

| Endpoint | Purpose |
|---|---|
| `GET /` | the page |
| `GET /api/runs`, `GET /api/run/<run_id>` | run list (summaries) and one full run record |
| `GET /api/queue` | pending and running entries, current caps |
| `POST /api/queue` | queue runs: `{"papers": [...], "models": [...], "modes": [...]}` |
| `DELETE /api/queue` | remove pending entries matching `models` / `papers` / `modes` |
| `POST /api/concurrency` | set `max_concurrent` and `profile_concurrency` until restart. A given `profile_concurrency` replaces the whole map. |
| `GET /api/options` | papers, profiles and modes offered for queueing |
| `GET /api/verification` | verification status of every graded run |
| `GET /api/leaderboard?mode=` | per-model mean, spread and count of train, test and final scores |
| `GET /api/leaderboard.png?mode=&group=` | bar chart (needs `matplotlib`); `group` is a label from `dashboard.paper_groups` |

The queue is kept in memory: after a restart, queue the runs again;
interrupted runs resume in place.

## Publishing

Off by default. With `publish.enabled: true` and `publish.github_org` set,
a completed run is pushed to branch `<mode>-<profile>` of the paper's
repository `<github_org>/<paper id>-<title slug>` (the repository Swarm
created; it is looked up, never created), the graded state to
`<mode>-<profile>-grader`, and locked normalization coefficients to branch
`normalization`. When publishing is disabled a run finishes normally and
records `publish.status = "disabled"`.

What is pushed is a replay of the workspace's commit history restricted to
`method/`, `idea/`, `scoring/`, `trajectory/`, `reflection.md` (stored as
`idea/reflection.md`), `environment/container.def` and `.evaluation.json`,
with a generated `README.md` and `manifest.json` on top. Large trajectory
files are split, and text files go through a pattern-based redaction of
API keys and tokens.

Safety properties (tested in `tests/test_git_publish_safety.py` and
`tests/test_workspace_helpers.py`):

- The target branch is checked against `main` / `master` / `HEAD`; the
  push uses an explicit `<branch>:<branch>` refspec, never `--all`,
  `--mirror` or a bare `--force`, and `--force-with-lease` pinned to the
  commit just fetched (or to "no such branch" for a new branch).
- The push happens from a temporary staging repository. The run workspace
  never gets a remote, and `evaluation/git.py` refuses `git push` and
  `git remote add`.
- The access token is read from the first `github.com` entry of
  `~/.git-credentials` and handed to git as an HTTP header through
  `GIT_CONFIG_*` environment variables. It never appears on a command
  line, in a remote URL or in an error message.

## Dataset builder

Run from this directory (`evaluation/`) or with the package installed. The
scripts read the runs under `<data_dir>/envs` of the evaluation config (or
the parquet built from them) and write to `dataset_builder/out/`.

| Script | Purpose |
|---|---|
| `python -m dataset_builder.build_dataset [--limit N] [--out FILE]` | One row per completed run: task inputs, reasoning trace, idea versions, reflection, normalized scores, grader verdict, usage. Writes `out/scientist_world_traces.parquet`. |
| `python -m dataset_builder.consolidate_batch [--workers N] [--limit N]` | Rewrites each multi-version idea chain into one self-contained statement (`out/consolidated.jsonl`). Run `build_dataset` again afterwards to fill `final_idea`. |
| `python -m dataset_builder.grade_axes_batch [--papers IDS] [--models NAMES] [--workers N] [--limit N]` | LLM judge, one call per property (creativity, engineering, rigor, significance, depth): `out/grades_axes.jsonl`. |
| `python -m dataset_builder.grade_grounded_batch [...]` | The same judge, shown the run's measured score: `out/grades_grounded.jsonl`. |
| `python -m dataset_builder.push_to_hf --repo ORG/NAME [--no-private]` | Uploads the parquet and `DATASET_CARD.md` to the Hugging Face Hub. The repository comes from `--repo` or `$SW_TRACES_REPO`, the token from `$HF_TOKEN`. |

The consolidation and judge scripts call the `codex` CLI on the host with a
read-only sandbox. Their model and effort default to the settings used for
the paper and can be changed with `CONSOLIDATE_MODEL` /
`CONSOLIDATE_EFFORT` and `GRADE_MODEL` / `GRADE_EFFORT`. The dataset's
`model` column is the profile's `actual_model` (or `model`) plus the effort
suffix of the profile name.

## Idea-implementation ablation

`idea_impl_eval.py` measures how the training stage of an idea's author
changes the result when one fixed implementer profile implements the idea.
It reads one idea per held-out paper and stage (`base`, `sft`, `grpo`) from
`../training/out/` (written by `training/baseline_untrained.py`) and runs
each as a variant of an `innovate` run in which the agent is told to
implement the given idea:

```bash
cd evaluation
python idea_impl_eval.py --profile <implementer profile> --smoke
python idea_impl_eval.py --profile <implementer profile> [--stages base,sft,grpo]
        [--papers ID,ID] [--ideas-dir DIR] [--concurrency 3] [--out-dir DIR]
```

Variant runs use their own workspaces
(`<profile>-innovate-impl-<stage>-eval`) and are left out of
normalization, publishing and grading, so each cell has train scores only.
Results are written to `<data_dir>/idea_impl_results/`.

## Tests

```bash
cd evaluation
python -m pytest -m "not slow"
EVAL_TEST_PAPER_ID=<paper id> python -m pytest -m slow
```

The first command needs no cluster, no configuration file and no network:
each test gets an empty configuration in a temporary directory, agent and
judge CLIs are stubbed, and nothing is pushed or signalled. Two tests
drive a mock terminal program through `tmux` and are skipped when `tmux`
is not installed; one chart test is skipped without `matplotlib`. The
`slow` test scaffolds a workspace with the real `sciworld` package, which
clones the environment of the given paper from GitHub.

## Security notes

- **The dashboard has no authentication** and its API can start runs,
  which spend compute and model credits. It binds `127.0.0.1` by default.
  Set `host` (or `--host`) to another address only on a trusted network,
  or put an authenticating proxy in front of it.
- **Credentials in workspaces.** Before every turn the agent CLI's login
  files are copied from the host user's home into the run workspace, where
  the agent process can read them. The workspace `.gitignore` keeps them
  out of the run's repository, and publishing copies an explicit list of
  paths. Keys given in a profile (`env_vars`, `api_keys`) are exported
  into the agent container. Use credentials whose exposure to an evaluated
  model you accept.
- **Agents run arbitrary code.** Isolation is the sciworld launcher's
  container contract (no host home or environment, the workspace as the
  only writable host path, read-only binds over the evaluators and
  briefing) plus the memory cap. Jobs run in the paper container under
  Nexus. The agent container shares the host's network.
- **Secrets in recorded text.** Agent output is redacted for GitHub and
  Hugging Face tokens and URL credentials before it is written to disk; a
  longer pattern list is applied when a run is published and when the
  dataset is built. The redaction is pattern-based and cannot catch every
  secret format.
- **Host paths.** `.evaluation.json` (error messages, exit details) and
  the grader's notes in `scoring/verification.json` can contain absolute
  host paths, and both files are part of what publishing pushes.
- **Process cleanup.** On `SIGTERM` / `SIGINT`, and at startup, the
  dashboard kills agent container process groups it started or that a
  previous dashboard left behind in `<data_dir>/envs`. This relies on the
  process names of a setuid Apptainer installation; with another
  installation type nothing is found and leftover agents must be stopped
  by hand.

## Limitations

- **Scope of the supported setup.** Apptainer on Linux with a Nexus and
  Hive deployment. There is no local-only mode: a run cannot execute its
  jobs without Nexus.
- **Completed means "train scores are complete".** A run is not comparable
  until it has been graded; the dashboard lists a completed run without a
  held-out score as `aborted`, and its leaderboard does not filter on the
  verdict (see `docs/scoring.md`).
- **GPU-hours are an estimate.** A job is charged the time the harness
  waited for it (queueing and check-in turns included) times the GPUs
  requested. The harness enforces the budget by refusing further
  submissions, not by stopping a running job; Nexus applies the project's
  budget on its side as well.
- **Every job requests at least one GPU.** `gpus: 0` in `action.yaml` is
  raised to 1, so the `cpu` and `cpu_test` resource groups are never
  selected for a job.
- **Validation rounds and check-ins.** A validation round starts a new
  agent session, numbers its turns from 1 and restarts the harness's
  GPU-hour count; a check-in turn reuses the number of the turn that
  submitted the job. In both cases the raw trajectory file of an earlier
  turn with the same number is overwritten.
- **Connection errors while waiting for a job** are treated as the end of
  the job: the agent is told the job finished while it may still be
  running.
- **Token usage and cost.** Usage is extracted for `claude-code`, `codex`
  and `gemini` only. `cost_usd` is the claude CLI's own estimate at list
  prices and is 0 for other scaffolds.
- **Prompts state fixed limits.** The agent-facing prompts mention 8 GB of
  memory and a 4-hour limit per turn. `agent_memory` is configurable, and
  the harness limits a turn only by inactivity (`idle_timeout_sec`), not by
  wall-clock time.
- **Variant runs** (idea-implementation ablation) cannot get a held-out
  score, because grading addresses a run by profile and mode.
- **The grader's work is model-driven.** A `verified` verdict means the
  grader re-ran and re-graded the frozen method and reported that the
  constraints held; the harness checks only that scores and executed jobs
  exist.
