"""SLURM allocation pool manager."""

import logging
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    HiveConfig,
    ResourceSpec,
    HIVE_DIR,
    STATE_PATH,
    SLURM_TRACKING_DIR,
    get_ray_tmpdir,
    parse_hive_duration_to_seconds,
)

log = logging.getLogger(__name__)


@dataclass
class SlurmJob:
    job_id: str
    state: str  # PENDING, RUNNING, COMPLETED, CANCELLED, FAILED, TIMEOUT
    node: Optional[str] = None
    gpus: int = 0
    partition: str = ""
    resource_name: str = ""  # name of the ResourceSpec that spawned this job


class PoolManager:
    """Manages a pool of SLURM jobs that provide GPU nodes to the Ray cluster."""

    def __init__(self, config: HiveConfig, ray_head_address: str):
        self.config = config
        self.ray_head_address = ray_head_address
        self.jobs: Dict[str, SlurmJob] = {}
        self._sbatch_template_paths: List[Path] = []

    def submit_jobs_multi(self, resources: List[ResourceSpec], default_walltime: str) -> List[str]:
        """Submit SLURM jobs across multiple resources.

        Each ResourceSpec must have num_jobs > 0 (caller distributes the total).
        Per-resource walltime is used if set, otherwise falls back to default_walltime.
        Returns list of all submitted SLURM job IDs.
        """
        all_job_ids = []
        sbatch_dir = HIVE_DIR / "sbatch"
        sbatch_dir.mkdir(parents=True, exist_ok=True)
        SLURM_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
        for pspec in resources:
            if pspec.num_jobs <= 0:
                continue
            walltime = pspec.walltime or default_walltime
            template = self._build_sbatch_script(pspec=pspec, walltime=walltime)
            # Save to persistent path (referenced by the EXIT trap for auto-requeue)
            tpath = sbatch_dir / f"{pspec.name}.sbatch"
            tpath.write_text(template)
            tpath.chmod(0o755)
            self._sbatch_template_paths.append(tpath)

            for i in range(pspec.num_jobs):
                job_id = self._submit_one(tpath)
                if job_id:
                    self.jobs[job_id] = SlurmJob(
                        job_id=job_id,
                        state="PENDING",
                        gpus=pspec.gpus_per_job,
                        partition=pspec.name,
                        resource_name=pspec.name,
                    )
                    # Track job ID -> resource name for dashboard
                    (SLURM_TRACKING_DIR / job_id).write_text(pspec.name)
                    all_job_ids.append(job_id)
                    log.info(
                        f"Submitted SLURM job {job_id} to {pspec.name} "
                        f"({i+1}/{pspec.num_jobs})"
                    )
                else:
                    log.error(
                        f"Failed to submit SLURM job {i+1}/{pspec.num_jobs} "
                        f"to {pspec.name}"
                    )

        return all_job_ids

    def refresh_job_states(self):
        """Query SLURM for current state of all tracked jobs."""
        if not self.jobs:
            return

        job_ids = ",".join(self.jobs.keys())
        try:
            result = subprocess.run(
                ["sacct", "-j", job_ids, "--format=JobID,State,NodeList", "--noheader", "--parsable2"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                log.warning(f"sacct failed: {result.stderr}")
                return

            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) < 3:
                    continue
                job_id, state, nodelist = parts[0], parts[1], parts[2]
                # sacct returns sub-job entries like "12345.batch" — skip those
                if "." in job_id:
                    continue
                if job_id in self.jobs:
                    # Normalize state: sacct returns e.g. "CANCELLED by 374202"
                    raw_state = state.strip()
                    normalized = raw_state.split()[0] if raw_state else raw_state
                    self.jobs[job_id].state = normalized
                    node = nodelist.strip() if nodelist.strip() else None
                    self.jobs[job_id].node = node
        except subprocess.TimeoutExpired:
            log.warning("sacct timed out")
        except FileNotFoundError:
            log.warning("sacct not found — not on a SLURM cluster?")

    def get_status_summary(self) -> dict:
        """Return a summary of pool state."""
        self.refresh_job_states()
        states = {}
        for job in self.jobs.values():
            states[job.state] = states.get(job.state, 0) + 1

        running_jobs = [j for j in self.jobs.values() if j.state == "RUNNING"]
        pending_jobs = [j for j in self.jobs.values() if j.state == "PENDING"]

        return {
            "total_jobs": len(self.jobs),
            "running": len(running_jobs),
            "pending": len(pending_jobs),
            "state_counts": states,
            "jobs": {
                jid: {"state": j.state, "node": j.node, "gpus": j.gpus}
                for jid, j in self.jobs.items()
            },
        }

    def cancel_all(self):
        """Cancel all tracked SLURM jobs."""
        for job_id in list(self.jobs.keys()):
            self._cancel_one(job_id)

        # Clean up sbatch templates
        for tpath in self._sbatch_template_paths:
            if tpath.exists():
                tpath.unlink()
        self._sbatch_template_paths.clear()

    def _submit_one(self, sbatch_path: Path) -> Optional[str]:
        """Submit a single sbatch script, return job ID or None."""
        try:
            result = subprocess.run(
                ["sbatch", str(sbatch_path)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                log.error(f"sbatch failed: {result.stderr}")
                return None
            # Parse "Submitted batch job 12345"
            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if match:
                return match.group(1)
            log.error(f"Could not parse sbatch output: {result.stdout}")
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            log.error(f"sbatch error: {e}")
            return None

    def cancel_jobs(self, job_ids: List[str]):
        """Cancel specific SLURM jobs and remove them from tracking.

        Uses --batch --signal=SIGKILL to prevent the EXIT trap in sbatch
        scripts from auto-requeuing the job.
        """
        for job_id in job_ids:
            # Remove tracking file FIRST — prevents requeue if trap fires
            tracking_file = SLURM_TRACKING_DIR / job_id
            if tracking_file.exists():
                tracking_file.unlink()

            try:
                subprocess.run(
                    ["scancel", "--batch", "--signal=SIGKILL", job_id],
                    capture_output=True, text=True, timeout=10,
                )
                log.info(f"Cancelled SLURM job {job_id}")
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                log.warning(f"scancel error for {job_id}: {e}")

            self.jobs.pop(job_id, None)

    def _cancel_one(self, job_id: str):
        """Cancel a single SLURM job."""
        try:
            subprocess.run(
                ["scancel", job_id],
                capture_output=True, text=True, timeout=10,
            )
            log.info(f"Cancelled SLURM job {job_id}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            log.warning(f"scancel error for {job_id}: {e}")

    def _build_sbatch_script(self, pspec: ResourceSpec, walltime: str) -> str:
        """Generate sbatch script content for a resource spec.

        The script includes an EXIT trap that auto-requeues the job when it
        ends (TIMEOUT, preemption, etc.), but only if the cluster is still
        running (state.json exists).  `hive stop` removes state.json before
        cancelling jobs, so the trap becomes a no-op on clean shutdown.
        """
        sc = self.config.slurm
        ec = self.config.env
        resource_label = pspec.name
        gpus_per_job = pspec.gpus_per_job
        state_path = shlex.quote(str(STATE_PATH))
        resolve_head_py = shlex.quote(
            "import json, pathlib, sys; "
            "p = pathlib.Path(sys.argv[1]); "
            "data = json.loads(p.read_text()) if p.exists() else {}; "
            "print(data.get('ray_head_address', ''))"
        )

        # Save the sbatch script to a persistent location so the EXIT trap
        # can re-submit it without knowing the original temp path.
        sbatch_dir = HIVE_DIR / "sbatch"
        sbatch_path = sbatch_dir / f"{resource_label}.sbatch"

        lines = [
            "#!/bin/bash",
            f"#SBATCH --nodes=1",
            f"#SBATCH --time={walltime}",
            f"#SBATCH --job-name=hive-{resource_label}",
            f"#SBATCH --output={HIVE_DIR / 'logs' / 'worker-%j.out'}",
            f"#SBATCH --error={HIVE_DIR / 'logs' / 'worker-%j.err'}",
        ]
        drain_before_timeout = parse_hive_duration_to_seconds(sc.drain_before_timeout)
        if drain_before_timeout > 0:
            lines.append(f"#SBATCH --signal=B:USR1@{drain_before_timeout}")
        if gpus_per_job > 0:
            lines.insert(2, f"#SBATCH --gres=gpu:{gpus_per_job}")
        if pspec.partition:
            lines.append(f"#SBATCH --partition={pspec.partition}")
        if pspec.account:
            lines.append(f"#SBATCH --account={pspec.account}")
        if pspec.constraint:
            lines.append(f"#SBATCH --constraint={pspec.constraint}")
        if pspec.qos:
            lines.append(f"#SBATCH --qos={pspec.qos}")
        if pspec.mem:
            lines.append(f"#SBATCH --mem={pspec.mem}")
        if pspec.cpus_per_task:
            lines.append(f"#SBATCH --cpus-per-task={pspec.cpus_per_task}")
        if pspec.exclude_nodes:
            lines.append(f"#SBATCH --exclude={pspec.exclude_nodes}")
        if sc.extra_args:
            for arg in shlex.split(sc.extra_args):
                lines.append(f"#SBATCH {arg}")

        lines.append("")

        # Auto-requeue: when this job exits (timeout, preemption, failure),
        # resubmit itself — but only if the cluster is still running.
        # hive stop removes state.json BEFORE cancelling jobs, so the trap
        # sees no state file and skips the requeue.
        # Also updates slurm_tracking: creates file for new job, removes old.
        tracking_dir = SLURM_TRACKING_DIR
        lines.append("_hive_requeue() {")
        lines.append("    # Stop Ray worker so it doesn't linger as a ghost after SLURM exits")
        lines.append("    ray stop --force >/dev/null 2>&1 || true")
        lines.append(f"    if [ -f {state_path} ]; then")
        lines.append(f"        NEW_ID=$(sbatch {sbatch_path} 2>/dev/null "
                      f"| grep -o '[0-9]*')")
        lines.append(f"        if [ -n \"$NEW_ID\" ]; then")
        lines.append(f"            echo \"{resource_label}\" > {tracking_dir}/$NEW_ID 2>/dev/null || true")
        lines.append(f"            rm -f {tracking_dir}/$SLURM_JOB_ID 2>/dev/null || true")
        lines.append(f"            echo \"Auto-requeued {resource_label} worker -> $NEW_ID\"")
        lines.append(f"        fi")
        lines.append("    fi")
        lines.append("}")
        lines.append("trap _hive_requeue EXIT")
        lines.append("")
        if drain_before_timeout > 0:
            drain_cmd = (
                "python -m hive.node_drain "
                '--state-path "$_STATE_PATH" '
                f"--resource-name {shlex.quote(resource_label)} "
                '--slurm-job-id "${SLURM_JOB_ID:-}"'
            )
            lines.append("_hive_drain() {")
            lines.append(
                f"    echo \"Hive drain signal received for {resource_label} "
                f"(${{SLURM_JOB_ID:-unknown}}) - draining before walltime\""
            )
            lines.append(f"    if ! {drain_cmd}; then")
            lines.append("        echo 'WARN: node_drain failed; leaving worker running'")
            lines.append("        return 0")
            lines.append("    fi")
            lines.append("    ray stop --grace-period=30 >/dev/null 2>&1 || true")
            lines.append("    exit 0")
            lines.append("}")
            lines.append("trap _hive_drain USR1")
            lines.append("")

        lines.append("set -e")
        lines.append("")

        # Self-register tracking file so the dashboard always knows about
        # this SLURM job, even if the original submit didn't create one
        # (e.g. after hive relaunch, manual sbatch, or lost tracking dir).
        lines.append(f"echo \"{resource_label}\" > {tracking_dir}/$SLURM_JOB_ID 2>/dev/null || true")
        lines.append("")

        # Activate virtual environment if configured
        if ec.venv_path:
            lines.append(f"source {ec.venv_path}/bin/activate")
            lines.append("")

        # Site-specific setup (module loads, proxies, ...) from env.setup_commands
        for setup_cmd in (ec.setup_commands or []):
            lines.append(str(setup_cmd))
        # Workers must resolve the same state directory as the head node.
        lines.append(f"export HIVE_DIR={shlex.quote(str(HIVE_DIR))}")
        lines.append(f"export HIVE_RAY_TMPDIR={get_ray_tmpdir()}")
        # Send Ray worker logs to stderr. With file-based logging, Ray's log
        # monitor can fail on a shared filesystem when it stats a worker log
        # that was already cleaned up after the worker died.
        lines.append("export RAY_LOG_TO_STDERR=1")
        lines.append("")
        lines.append(f"_STATE_PATH={state_path}")
        lines.append("_hive_current_head() {")
        lines.append(f"    python -c {resolve_head_py} \"$_STATE_PATH\" 2>/dev/null || true")
        lines.append("}")
        lines.append("")

        # Start Ray worker in a reconnect loop so that `hive restart` (which
        # restarts the Ray head) doesn't lose these SLURM jobs.  When the head
        # goes down the local Ray worker dies; we stop it cleanly, wait for the
        # head to come back, then rejoin.
        restart_signal = HIVE_DIR / "restart_signal"
        lines.append(f"_RESTART_SIGNAL={restart_signal}")
        lines.append(f"_RESTART_TS=$(cat $_RESTART_SIGNAL 2>/dev/null || echo 0)")
        lines.append("_RESTART_GRACE_SEC=330")
        lines.append("_RESTART_GRACE_UNTIL=0")
        lines.append("_CONNECT_FAILS=0")
        lines.append("_MAX_CONNECT_FAILS=12")
        lines.append(f"echo \"Starting Ray worker on $(hostname) with {gpus_per_job} GPUs ({resource_label})\"")
        lines.append("while true; do")
        lines.append("    _RAY_HEAD=$(_hive_current_head)")
        lines.append("    if [ -z \"$_RAY_HEAD\" ]; then")
        lines.append("        if [ ! -f \"$_STATE_PATH\" ]; then")
        lines.append("            echo \"Hive stopped — exiting.\"")
        lines.append("            exit 0")
        lines.append("        fi")
        lines.append("        echo \"No ray_head_address in Hive state — retrying in 5s...\"")
        lines.append("        sleep 5")
        lines.append("        continue")
        lines.append("    fi")
        lines.append("    if ! " + (
            f"ray start --address=\"$_RAY_HEAD\" --num-gpus={gpus_per_job} "
            f"--temp-dir=$HIVE_RAY_TMPDIR "
            f"--resources='{{\"partition:{resource_label}\": 100}}'"
        ) + "; then")
        lines.append("        _CUR_TS=$(cat $_RESTART_SIGNAL 2>/dev/null || echo 0)")
        lines.append("        _NEXT_HEAD=$(_hive_current_head)")
        lines.append("        if [ \"$_CUR_TS\" != \"$_RESTART_TS\" ] || { [ -n \"$_NEXT_HEAD\" ] && [ \"$_NEXT_HEAD\" != \"$_RAY_HEAD\" ]; }; then")
        lines.append("            echo \"ray start failed for stale head $_RAY_HEAD — retrying in 5s...\"")
        lines.append("            _RESTART_TS=$_CUR_TS")
        lines.append("            _RESTART_GRACE_UNTIL=$(($(date +%s) + $_RESTART_GRACE_SEC))")
        lines.append("            _CONNECT_FAILS=0")
        lines.append("            sleep 5")
        lines.append("            continue")
        lines.append("        fi")
        lines.append("        _NOW=$(date +%s)")
        lines.append("        if [ \"$_NOW\" -lt \"$_RESTART_GRACE_UNTIL\" ]; then")
        lines.append("            echo \"ray start still failing during restart grace for $_RAY_HEAD — retrying in 5s...\"")
        lines.append("            sleep 5")
        lines.append("            continue")
        lines.append("        fi")
        lines.append("        _CONNECT_FAILS=$((_CONNECT_FAILS + 1))")
        lines.append("        if [ \"$_CONNECT_FAILS\" -ge \"$_MAX_CONNECT_FAILS\" ]; then")
        lines.append("            echo \"ray start failed for $_RAY_HEAD after $_CONNECT_FAILS attempts — exiting for SLURM requeue\"")
        lines.append("            exit 1")
        lines.append("        fi")
        lines.append("        echo \"ray start failed for $_RAY_HEAD ($_CONNECT_FAILS/$_MAX_CONNECT_FAILS) — retrying in 5s...\"")
        lines.append("        sleep 5")
        lines.append("        continue")
        lines.append("    fi")
        lines.append("    _RESTART_GRACE_UNTIL=0")
        lines.append("    _CONNECT_FAILS=0")
        lines.append("    _RESTART_TS=$(cat $_RESTART_SIGNAL 2>/dev/null || echo 0)")
        lines.append("")
        lines.append("    # Wait until head dies or restart signal changes")
        lines.append("    while ray status --address=\"$_RAY_HEAD\" >/dev/null 2>&1; do")
        lines.append("        _CUR_TS=$(cat $_RESTART_SIGNAL 2>/dev/null || echo 0)")
        lines.append("        if [ \"$_CUR_TS\" != \"$_RESTART_TS\" ]; then")
        lines.append("            echo \"Restart signal detected — reconnecting...\"")
        lines.append("            _RESTART_TS=$_CUR_TS")
        lines.append("            _RESTART_GRACE_UNTIL=$(($(date +%s) + $_RESTART_GRACE_SEC))")
        lines.append("            _CONNECT_FAILS=0")
        lines.append("            break")
        lines.append("        fi")
        lines.append("        sleep 15")
        lines.append("    done")
        lines.append("")
        lines.append("    # Clean up local Ray and retry")
        lines.append(f"    echo \"Reconnecting in 5s...\"")
        lines.append(f"    ray stop --force >/dev/null 2>&1 || true")
        lines.append(f"    sleep 5")
        lines.append("")
        lines.append(f"    # If state.json was removed (hive stop), exit instead of reconnecting")
        lines.append("    if [ ! -f \"$_STATE_PATH\" ]; then")
        lines.append(f"        echo \"Hive stopped — exiting.\"")
        lines.append(f"        exit 0")
        lines.append("    fi")
        lines.append("done")
        lines.append("")

        return "\n".join(lines)
