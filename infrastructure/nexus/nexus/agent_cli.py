#!/usr/bin/env python3
"""Nexus agent CLI: job submission for agents inside containers.

Standalone script using only the Python standard library. It can be
copied into a container and run directly without installing the nexus
package.

Reads configuration from environment variables:
    NEXUS_URL         Nexus daemon URL, e.g. http://nexus-host:39000
    NEXUS_PROJECT_ID  Fallback project ID (used when --project is omitted)

Usage:
    nx submit "python train.py --lr 1e-4" --project my_proj --gpus 2
    nx status --project my_proj
    nx logs <run_id>
    nx continue <run_id>
    nx cancel <run_id>
    nx wait --project my_proj
    nx info --project my_proj
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# The daemon ends one wait request after at most this many seconds.
_SERVER_WAIT_CAP_SEC = 600


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _get_url():
    """Resolve the Nexus daemon URL.

    Priority chain:
        1. NEXUS_URL env var (explicit override, highest priority)
        2. The daemon_url entry of <NEXUS_DIR>/remote.json
        3. Error exit with helpful message

    Stays stdlib-only: no nexus package imports.
    """
    url = os.environ.get("NEXUS_URL")
    if url:
        return url.rstrip("/")

    # Try remote.json from NEXUS_DIR
    nexus_dir = os.environ.get("NEXUS_DIR")
    if not nexus_dir:
        # Default: <script_dir>/../.nexus/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        nexus_dir = os.path.join(os.path.dirname(script_dir), ".nexus")

    remote_path = os.path.join(nexus_dir, "remote.json")
    if os.path.isfile(remote_path):
        try:
            with open(remote_path) as f:
                data = json.loads(f.read())
            daemon_url = data.get("daemon_url")
            if daemon_url:
                return daemon_url.rstrip("/")
        except (json.JSONDecodeError, OSError):
            pass

    print(
        "Error: cannot determine Nexus daemon URL.\n"
        "  Set NEXUS_URL env var, or use 'nexus connect <host:port>' to create remote.json.",
        file=sys.stderr,
    )
    sys.exit(1)


def _resolve_project(args):
    """Resolve project ID from --project flag or NEXUS_PROJECT_ID env var.

    The --project flag takes precedence over the environment variable.
    Exits with a clear error if neither is set.
    """
    project_id = getattr(args, "project", None) or os.environ.get("NEXUS_PROJECT_ID")
    if not project_id:
        print(
            "Error: project ID required: use --project or set NEXUS_PROJECT_ID",
            file=sys.stderr,
        )
        sys.exit(1)
    return project_id


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------

def _get(url, path, timeout=30):
    """GET JSON from the Nexus daemon API."""
    target = "%s%s" % (url, path)
    req = urllib.request.Request(target)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
            msg = body.get("error", str(e))
        except Exception:
            msg = str(e)
        print("Error: %s" % msg, file=sys.stderr)
        sys.exit(1)
    except (urllib.error.URLError, OSError) as e:
        print(
            "Error: cannot reach Nexus daemon at %s (%s)" % (url, e),
            file=sys.stderr,
        )
        sys.exit(1)


def _post(url, path, data=None):
    """POST JSON to the Nexus daemon API."""
    target = "%s%s" % (url, path)
    body = json.dumps(data or {}).encode()
    req = urllib.request.Request(
        target,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode())
            msg = err_body.get("error", str(e))
        except Exception:
            msg = str(e)
        print("Error: %s" % msg, file=sys.stderr)
        sys.exit(1)
    except (urllib.error.URLError, OSError) as e:
        print(
            "Error: cannot reach Nexus daemon at %s (%s)" % (url, e),
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Log directory management (persisted per-run)
# ---------------------------------------------------------------------------

def _log_state_file():
    """Path to the JSON file that maps run_id to log_dir for this workspace.

    Defaults to <CWD>/.nx_log_dirs.json (agents run in their workspace, so
    this stays with the workspace and survives across turns). Override via
    NEXUS_LOG_STATE env var.
    """
    override = os.environ.get("NEXUS_LOG_STATE")
    if override:
        return override
    return os.path.join(os.getcwd(), ".nx_log_dirs.json")


def _save_log_dir(run_id, log_dir):
    """Record that `run_id`'s logs should be written to `log_dir`."""
    path = _log_state_file()
    try:
        with open(path) as f:
            state = json.loads(f.read())
    except (OSError, json.JSONDecodeError):
        state = {}
    state[run_id] = os.path.abspath(log_dir)
    try:
        with open(path, "w") as f:
            f.write(json.dumps(state, indent=2))
    except OSError as e:
        print("Warning: could not save log-dir state: %s" % e, file=sys.stderr)


def _load_log_dir(run_id):
    """Return the stored log_dir for `run_id`, or None."""
    path = _log_state_file()
    try:
        with open(path) as f:
            state = json.loads(f.read())
        return state.get(run_id)
    except (OSError, json.JSONDecodeError):
        return None


def _write_job_logs(run_id, log_dir, stdout, stderr):
    """Write stdout/stderr content to <log_dir>/<run_id>/{stdout,stderr}.log.

    These are real files (not symlinks) so they work inside containers
    regardless of host paths. Called after fetching job info from the daemon.
    """
    if not log_dir:
        return
    try:
        job_dir = os.path.join(log_dir, run_id)
        os.makedirs(job_dir, exist_ok=True)
        if stdout is not None:
            with open(os.path.join(job_dir, "stdout.log"), "w") as f:
                f.write(stdout or "")
        if stderr is not None:
            with open(os.path.join(job_dir, "stderr.log"), "w") as f:
                f.write(stderr or "")
    except OSError as e:
        print("Warning: could not write logs to %s: %s" % (log_dir, e),
              file=sys.stderr)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_duration(seconds):
    """Format seconds as human-readable duration string.

    Returns 'Xh Ym' for longer durations, 'Xm Ys' for shorter ones.
    """
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return "%dh%02dm" % (hours, minutes)
    if minutes > 0:
        return "%dm%02ds" % (minutes, secs)
    return "%ds" % secs


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_submit(args):
    """Submit a job to the GPU pool."""
    url = _get_url()
    project_id = _resolve_project(args)

    payload = {
        "command": args.command,
        "num_gpus": args.gpus,
        "num_cpus": args.cpus,
    }
    if args.resource:
        payload["resource"] = ",".join(args.resource)
    if args.timeout is not None:
        payload["timeout_sec"] = args.timeout

    result = _post(url, "/api/projects/%s/jobs" % project_id, payload)

    # Plain "key: value" output that is easy to parse
    run_id = result.get("run_id", "?")
    gpus_in_use = result.get("project_gpus_in_use", "?")
    max_gpus = result.get("project_max_gpus", "?")
    # An empty resource means the job may run on any Hive resource.
    resource = (result.get("resource")
                or (",".join(args.resource) if args.resource else "any"))
    timeout = result.get("timeout_sec")
    gpu_used = result.get("gpu_seconds_used", 0)
    gpu_budget = result.get("soft_gpu_budget_hours", 0)

    # Persist log-dir mapping so status/logs/wait can auto-write logs
    log_dir = getattr(args, "log_dir", None)
    if log_dir and run_id != "?":
        _save_log_dir(run_id, log_dir)
        os.makedirs(os.path.join(log_dir, run_id), exist_ok=True)

    print("JOB SUBMITTED")
    print("run_id: %s" % run_id)
    print("command: %s" % args.command)
    print("resource: %s" % resource)
    print("gpus: %d" % args.gpus)
    if timeout:
        print("timeout: %ds" % timeout)
    print("project_gpus_in_use: %s/%s" % (gpus_in_use, max_gpus))
    if gpu_budget > 0:
        print("gpu_budget: %.1f / %.1f gpu-hours" % (gpu_used / 3600, gpu_budget))
    else:
        print("gpu_budget: unlimited")
    if log_dir:
        print("log_dir: %s/%s/" % (os.path.abspath(log_dir), run_id))

    return 0


def cmd_status(args):
    """Show job status for this project."""
    url = _get_url()

    if args.run_id:
        # Single job status: no project needed
        data = _get(url, "/api/jobs/%s" % args.run_id)
        status = data.get("status", "?")
        num_gpus = data.get("num_gpus", 0)
        cmd = data.get("command", "?")

        started_at = data.get("started_at")
        completed_at = data.get("completed_at")
        if started_at:
            end = completed_at or time.time()
            elapsed = _format_duration(end - started_at)
        else:
            elapsed = "-"

        returncode = data.get("returncode")
        rc_str = ""
        if returncode is not None:
            rc_str = " (exit %d)" % returncode

        # Refresh log files if --log-dir was recorded for this run
        log_dir = getattr(args, "log_dir", None) or _load_log_dir(args.run_id)
        if log_dir:
            _write_job_logs(args.run_id, log_dir,
                            data.get("stdout"), data.get("stderr"))

        print("JOB %s" % data.get("run_id", "?"))
        print("  status: %s%s" % (status, rc_str))
        print("  gpus: %d" % num_gpus)
        print("  elapsed: %s" % elapsed)
        print("  command: %s" % cmd)
        if log_dir:
            print("  log_dir: %s/%s/" % (os.path.abspath(log_dir), args.run_id))
    else:
        # All jobs for this project: project required
        project_id = _resolve_project(args)
        data = _get(url, "/api/projects/%s/jobs" % project_id)
        if isinstance(data, list):
            jobs = data
        else:
            jobs = data.get("jobs", [])

        if not jobs:
            print("JOBS FOR PROJECT %s" % project_id)
            print("  (no jobs)")
            return 0

        # Sort by submitted_at descending
        jobs.sort(key=lambda j: j.get("submitted_at", 0), reverse=True)

        print("JOBS FOR PROJECT %s" % project_id)
        for j in jobs:
            run_id = j.get("run_id", "?")
            status = j.get("status", "?")
            num_gpus = j.get("num_gpus", 0)
            cmd = j.get("command", "")
            cmd_display = cmd[:50] + "..." if len(cmd) > 53 else cmd

            started_at = j.get("started_at")
            completed_at = j.get("completed_at")
            if started_at:
                end = completed_at or time.time()
                elapsed = _format_duration(end - started_at)
            else:
                elapsed = "-"

            returncode = j.get("returncode")
            rc_str = ""
            if status in ("completed", "failed") and returncode is not None:
                rc_str = " (exit %d)" % returncode

            print("  %s  %-18s %d GPU  %s  %s%s" % (
                run_id, status, num_gpus, elapsed, cmd_display, rc_str))

    return 0


def cmd_logs(args):
    """Show logs for a specific job."""
    url = _get_url()

    data = _get(url, "/api/jobs/%s" % args.run_id)

    status = data.get("status", "?")
    returncode = data.get("returncode")
    rc_str = ""
    if returncode is not None:
        rc_str = ", exit %d" % returncode

    print("JOB %s (%s%s)" % (data.get("run_id", "?"), status, rc_str))
    print("  command: %s" % data.get("command", "?"))
    print("  gpus: %d" % data.get("num_gpus", 0))

    started_at = data.get("started_at")
    completed_at = data.get("completed_at")
    if started_at:
        end = completed_at or time.time()
        print("  elapsed: %s" % _format_duration(end - started_at))
    print()

    stdout = data.get("stdout", "")
    stderr = data.get("stderr", "")

    # Refresh log files if --log-dir was recorded for this run. The daemon
    # sends no output for a job that has ended; passing None then leaves
    # the files as they are instead of emptying them.
    log_dir = getattr(args, "log_dir", None) or _load_log_dir(args.run_id)
    if log_dir:
        _write_job_logs(args.run_id, log_dir,
                        data.get("stdout"), data.get("stderr"))
        print("  log_dir: %s/%s/" % (os.path.abspath(log_dir), args.run_id))
        print()

    if stdout:
        print("-- stdout --")
        print(stdout, end="")
        if not stdout.endswith("\n"):
            print()
    if stderr:
        print("-- stderr --")
        print(stderr, end="")
        if not stderr.endswith("\n"):
            print()

    if status == "queued":
        print("(job is queued, waiting for resources)")
    elif status == "running":
        print("(job is still running)")
    elif status == "awaiting_response":
        print("(checkpoint reached, respond with: nx continue %s  OR  nx cancel %s)"
              % (args.run_id, args.run_id))
    else:
        # The daemon serves output only while a job is active.
        print("(job has ended; its final output is in jobs/%s/ of the workspace)"
              % args.run_id)

    return 0


def cmd_continue(args):
    """Continue a job after a checkpoint."""
    url = _get_url()

    _post(url, "/api/jobs/%s/continue" % args.run_id)
    print("Job '%s' continued." % args.run_id)
    return 0


def cmd_cancel(args):
    """Cancel a running job."""
    url = _get_url()

    _post(url, "/api/jobs/%s/cancel" % args.run_id)
    print("Job '%s' cancelled." % args.run_id)
    return 0


def cmd_wait(args):
    """Block until the next event (checkpoint or completion) for this project."""
    url = _get_url()
    project_id = _resolve_project(args)

    timeout = args.timeout

    # Long-poll the wait endpoint. The daemon holds the connection open,
    # so the HTTP timeout is longer than the requested wait.
    path = "/api/projects/%s/wait?timeout=%d" % (project_id, timeout)
    if args.run_id:
        path += "&run_id=%s" % args.run_id
    data = _get(url, path, timeout=timeout + 30)

    # No event: the daemon answers {"timeout": true}. It ends one wait
    # after at most _SERVER_WAIT_CAP_SEC, whatever --timeout asked for.
    if data.get("timeout"):
        print("TIMEOUT: no event within %ds" % min(timeout, _SERVER_WAIT_CAP_SEC))
        return 0

    # The event name is the status the job has just reached.
    event = data.get("event", "?")
    run_id = data.get("run_id", "?")
    status = data.get("status", "?")

    # Refresh log files if --log-dir was recorded for this run
    log_dir = _load_log_dir(run_id) if run_id != "?" else None
    if log_dir:
        try:
            job_data = _get(url, "/api/jobs/%s" % run_id)
            _write_job_logs(run_id, log_dir,
                            job_data.get("stdout"), job_data.get("stderr"))
        except SystemExit:
            pass  # a failed log refresh must not fail the wait

    print("EVENT: %s" % event)
    print("RUN_ID: %s" % run_id)
    print("STATUS: %s" % status)
    if log_dir:
        print("LOG_DIR: %s/%s/" % (os.path.abspath(log_dir), run_id))

    # At a checkpoint, say how to answer it
    if status == "awaiting_response":
        print()
        print("Respond with: nx continue %s  OR  nx cancel %s" % (run_id, run_id))

    return 0


def cmd_info(args):
    """Show project info (quota, budget, job counts)."""
    url = _get_url()
    project_id = _resolve_project(args)

    data = _get(url, "/api/projects/%s" % project_id)

    status = data.get("status", "?")
    gpus_in_use = data.get("gpus_in_use", 0)
    max_gpus = data.get("max_gpus", 0)

    # GPU budget
    budget = data.get("soft_gpu_budget_hours", 0)
    used_hours = data.get("gpu_seconds_used", 0) / 3600
    if budget > 0:
        budget_str = "%.1f / %.1f gpu-hours used (soft limit)" % (used_hours, budget)
    elif used_hours > 0:
        budget_str = "%.1f gpu-hours used (unlimited)" % used_hours
    else:
        budget_str = "unlimited"

    # Job counts
    job_counts = data.get("job_counts", {})
    running = job_counts.get("running", 0)
    completed = job_counts.get("completed", 0)
    failed = job_counts.get("failed", 0)

    # An empty list means the project may use any Hive resource.
    resources = data.get("allowed_resources", [])

    print("PROJECT %s" % project_id)
    print("  status: %s" % status)
    print("  resources: %s" % (", ".join(resources) if resources else "any"))
    print("  gpus: %d/%d in use" % (gpus_in_use, max_gpus))
    print("  gpu_budget: %s" % budget_str)
    if data.get("budget_exhausted"):
        print("  budget_exhausted: YES")
    print("  job_timeout: %ds" % data.get("default_job_timeout", 3600))
    print("  jobs: %d running, %d completed, %d failed" % (running, completed, failed))

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _add_project_arg(parser):
    """Add --project flag to a subparser."""
    parser.add_argument(
        "--project", default=None,
        help="Project ID (falls back to NEXUS_PROJECT_ID env var)",
    )


_TOP_DESCRIPTION = """\
nx: Nexus agent CLI for submitting GPU jobs from inside a container.

You are running in an isolated container with no GPUs attached. To do
GPU work, submit a job via nx. The Nexus daemon forwards your job to
the GPU pool (Hive), enforces your project's quota, and returns a
run_id. You can then check status, read logs, wait for completion,
and respond to checkpoints, all without leaving your session.

Your project is pre-registered by the orchestrator that started this
container. NEXUS_PROJECT_ID is already set for you. If nx submit returns
"Project ... not found", your container was launched without a
registered project: contact whoever started it.
"""

_TOP_EPILOG = """\
TYPICAL WORKFLOW
  1. Submit a job
       nx submit "python train.py" --gpus 2
  2. Wait for the next event (completed / failed / cancelled / checkpoint)
       nx wait --run-id <run_id> --timeout 600
  3. Inspect output
       nx logs <run_id>
  4. At a checkpoint, continue or cancel
       nx continue <run_id>
       nx cancel   <run_id>

ENVIRONMENT
  NEXUS_URL         Nexus daemon URL (e.g. http://localhost:39000).
                    If unset, nx tries <NEXUS_DIR>/remote.json.
  NEXUS_PROJECT_ID  Default project ID used when --project is omitted.
                    Usually pre-set by the container orchestrator.
  NEXUS_DIR         Override the nexus state directory (advanced).

EXIT CODES
  0   success (including a wait that timed out)
  1   daemon unreachable, or the request was rejected
      (quota, budget, unknown project or job)
  2   invalid command line

PER-COMMAND DETAILS
  Run  nx <command> --help  for the full options and behavior of any subcommand.

EXAMPLES
  nx submit "bash scripts/run.sh" --gpus 2
  nx status                     # all jobs in my project
  nx status 7f8c2a31-4b1        # one job
  nx logs   7f8c2a31-4b1
  nx wait   --timeout 600
  nx cancel 7f8c2a31-4b1
"""

_SUBMIT_DESCRIPTION = """\
Queue a shell command for execution on the GPU pool. The Nexus daemon
validates your project's quota synchronously, forwards the request to
Hive, and returns a run_id.

The <command> runs inside the same workspace you see now (bind-mounted
on the compute node). Compute nodes may have no internet access:
download models and data before submitting. CUDA_VISIBLE_DEVICES is set
for you; do not override it.

If you pass --log-dir, the logs for this job will be mirrored to
<log-dir>/<run_id>/stdout.log and stderr.log whenever you call
nx status / nx logs / nx wait while the job is active. These are real
files inside your workspace: tail them like any other file. When the
job ends, its complete output is in jobs/<run_id>/ of the workspace.
"""

_SUBMIT_EPILOG = """\
OUTPUT (stdout)
  JOB SUBMITTED
  run_id: <id>                   # use this for status/logs/cancel/continue
  command: <cmd>
  resource: <names> | any
  gpus: <N>
  timeout: <S>s                  # only shown if a timeout was set
  project_gpus_in_use: <u>/<max>
  gpu_budget: <used> / <total> gpu-hours  |  unlimited
  log_dir: <DIR>/<run_id>/       # only shown if --log-dir was passed

EXAMPLES
  nx submit "python train.py --epochs 5"
  nx submit "bash scripts/method.sh" --gpus 4 --timeout 10800
  nx submit "python eval.py" --resource gpu-a --cpus 4
  nx submit "bash run.sh" --log-dir jobs    # mirror logs to jobs/<run_id>/
"""

_STATUS_DESCRIPTION = """\
Inspect one job or every job in your project.

With a run_id:   shows that job's current state, resources, and command.
Without a run_id: lists every job in the project (--project or
                  NEXUS_PROJECT_ID).

STATUS VALUES
  queued              Waiting for resources.
  running             Executing on a compute node.
  awaiting_response   At a checkpoint: answer with continue or cancel.
  completed           Finished successfully.
  failed              Exited non-zero or timed out.
  cancelled           Cancelled through Nexus (by you or an operator).
"""

_STATUS_EPILOG = """\
EXAMPLES
  nx status 7f8c2a31-4b1        # one job
  nx status --project my_proj   # all jobs in my project
"""

_LOGS_DESCRIPTION = """\
Fetch captured stdout and stderr for a job.

Logs are snapshotted; rerun this command to see newer output. For long-
running jobs, prefer `nx wait` over polling `nx logs` in a loop. Binary
or non-UTF8 output may appear garbled: redirect to files for that.

Output is served while the job is queued, running or at a checkpoint.
After it has ended, read jobs/<run_id>/stdout.log and stderr.log in the
workspace instead.
"""

_LOGS_EPILOG = """\
OUTPUT
  JOB <run_id> (<status>)
    command: <cmd>
  -- stdout --
    <captured stdout>
  -- stderr --
    <captured stderr>

EXAMPLES
  nx logs 7f8c2a31-4b1
"""

_CONTINUE_DESCRIPTION = """\
Resume a job that is in 'awaiting_response' state.

A job enters this state at a periodic checkpoint, after a fixed interval
of run time set by the operator. The job itself keeps running on the
compute node; the checkpoint is the moment to read its logs and decide
whether it should go on. Call 'continue' (proceed) or 'cancel' (stop).
Without an answer, the job is continued automatically after the
checkpoint timeout. Check for such jobs via 'nx wait' output or
'nx status'.
"""

_CONTINUE_EPILOG = """\
EXAMPLES
  nx continue 7f8c2a31-4b1
"""

_CANCEL_DESCRIPTION = """\
Cancel a queued or running job.

Queued jobs are removed from the queue. Running jobs are killed on the
compute node. Cancelling a finished job has no effect.

Use this when a job is taking too long to queue, when results are
clearly bad, or at a checkpoint instead of 'continue'.
"""

_CANCEL_EPILOG = """\
EXAMPLES
  nx cancel 7f8c2a31-4b1
"""

_WAIT_DESCRIPTION = """\
Block until the next event in your project (or a timeout).

Uses long-polling on the daemon: returns as soon as an event occurs.
More efficient and responsive than polling 'nx status'.

EVENTS (the event name is the status the job has just reached)
  completed           A job finished successfully.
  failed              A job exited non-zero or timed out.
  awaiting_response   A job reached a periodic checkpoint.
  cancelled           A job was cancelled (by you or an operator).

If nothing happens in time, nx prints one line starting with "TIMEOUT:"
and exits 0. The daemon ends one wait after at most 600 seconds, so a
longer --timeout behaves like 600: run nx wait again.

With --run-id, a job that has already ended or is at a checkpoint is
reported at once. Otherwise only events that occur while nx wait is
blocking are reported. In both cases an event of any job in the project
ends the wait.
"""

_WAIT_EPILOG = """\
OUTPUT
  EVENT: <event>
  RUN_ID: <run_id>
  STATUS: <status>
  (at a checkpoint: also a hint to run 'nx continue' or 'nx cancel')

EXAMPLES
  nx wait --timeout 600                         # next event in my project
  nx wait --run-id 7f8c2a31-4b1 --timeout 600   # also reports this job if it already ended
"""

_INFO_DESCRIPTION = """\
Show your project's current quota usage, budget, and allowed resources.

Useful before submitting a batch of jobs, to check you have room inside
max_gpus and are not about to blow through your soft GPU-hour budget.
"""

_INFO_EPILOG = """\
OUTPUT
  PROJECT <id>
    status: registered | active
    resources: <comma-separated resources this project can run on>  |  any
    gpus: <in_use>/<max> in use
    gpu_budget: <used> / <total> gpu-hours used (soft limit)  |  unlimited
    budget_exhausted: YES          # only shown once the budget is used up
    job_timeout: <S>s              # default per-job timeout
    jobs: <N> running, <N> completed, <N> failed

NOTES
  - gpu_budget is a SOFT cap: a running job is never stopped for
    exceeding it, but once it is used up and no job is active, new
    submissions are rejected until the operator extends the budget.
  - max_gpus is a HARD cap on concurrent GPU usage.

EXAMPLES
  nx info
  nx info --project my_proj
"""


def build_parser():
    """Build the ``nx`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="nx",
        description=_TOP_DESCRIPTION,
        epilog=_TOP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- nx submit ---------------------------------------------------------
    p_submit = subparsers.add_parser(
        "submit",
        help="Submit a GPU job to your project's queue",
        description=_SUBMIT_DESCRIPTION,
        epilog=_SUBMIT_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_submit.add_argument("command",
                          help="Shell command to run on the compute node (quote it)")
    _add_project_arg(p_submit)
    p_submit.add_argument("--resource", "-r", action="append", default=None,
                          metavar="NAME",
                          help="Pin to a Hive resource (repeatable; omit to use any allowed)")
    p_submit.add_argument("--gpus", type=int, default=1, metavar="N",
                          help="Number of GPUs (default: 1; capped by project's max_gpus)")
    p_submit.add_argument("--cpus", type=int, default=1, metavar="N",
                          help="Number of CPUs (default: 1)")
    p_submit.add_argument("--timeout", type=int, default=None, metavar="S",
                          help="Per-job wall-clock timeout in seconds "
                               "(default: project's default_job_timeout)")
    p_submit.add_argument("--log-dir", type=str, default=None, metavar="DIR",
                          help="Write job logs to <DIR>/<run_id>/{stdout,stderr}.log. "
                               "Files are refreshed on every subsequent nx status/logs/wait "
                               "for this run_id while the job is active. "
                               "Path is a container-local path.")
    p_submit.set_defaults(func=cmd_submit)

    # -- nx status [run_id] ------------------------------------------------
    p_status = subparsers.add_parser(
        "status",
        help="Show one job (by run_id) or every job in the project",
        description=_STATUS_DESCRIPTION,
        epilog=_STATUS_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_status.add_argument("run_id", nargs="?", default=None,
                          help="Job run ID (omit to list all jobs in the project)")
    _add_project_arg(p_status)
    p_status.add_argument("--log-dir", type=str, default=None, metavar="DIR",
                          help="Override the stored log-dir for this run_id; "
                               "logs are refreshed into <DIR>/<run_id>/")
    p_status.set_defaults(func=cmd_status)

    # -- nx logs <run_id> --------------------------------------------------
    p_logs = subparsers.add_parser(
        "logs",
        help="Show captured stdout/stderr for a job",
        description=_LOGS_DESCRIPTION,
        epilog=_LOGS_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_logs.add_argument("run_id", help="Job run ID")
    p_logs.add_argument("--log-dir", type=str, default=None, metavar="DIR",
                        help="Override the stored log-dir for this run_id; "
                             "logs are refreshed into <DIR>/<run_id>/")
    p_logs.set_defaults(func=cmd_logs)

    # -- nx continue <run_id> ----------------------------------------------
    p_continue = subparsers.add_parser(
        "continue",
        help="Resume a job after a checkpoint (awaiting_response state)",
        description=_CONTINUE_DESCRIPTION,
        epilog=_CONTINUE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_continue.add_argument("run_id", help="Job run ID to continue")
    p_continue.set_defaults(func=cmd_continue)

    # -- nx cancel <run_id> ------------------------------------------------
    p_cancel = subparsers.add_parser(
        "cancel",
        help="Cancel a queued or running job",
        description=_CANCEL_DESCRIPTION,
        epilog=_CANCEL_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_cancel.add_argument("run_id", help="Job run ID to cancel")
    p_cancel.set_defaults(func=cmd_cancel)

    # -- nx wait -----------------------------------------------------------
    p_wait = subparsers.add_parser(
        "wait",
        help="Block until the next event in your project",
        description=_WAIT_DESCRIPTION,
        epilog=_WAIT_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_project_arg(p_wait)
    p_wait.add_argument("--run-id", type=str, default=None, metavar="ID",
                        help="Job to report at once if it has already ended or is "
                             "at a checkpoint (default: only new events)")
    p_wait.add_argument("--timeout", type=int, default=3600, metavar="S",
                        help="Max seconds to block (default: 3600; the daemon ends "
                             "one wait after at most 600). On timeout, exits 0.")
    p_wait.set_defaults(func=cmd_wait)

    # -- nx info -----------------------------------------------------------
    p_info = subparsers.add_parser(
        "info",
        help="Show project quota, budget, and allowed resources",
        description=_INFO_DESCRIPTION,
        epilog=_INFO_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_project_arg(p_info)
    p_info.set_defaults(func=cmd_info)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
