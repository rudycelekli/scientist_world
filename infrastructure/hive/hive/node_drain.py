"""Drain a local Hive worker node before SLURM walltime."""

import argparse
import logging
import os
import socket
import subprocess
import time
from typing import Optional

import ray

from .config import STATE_PATH, load_config, load_runtime_ray_head
from .dispatcher import (
    DRAINING_NODES_DIR,
    JobRecord,
    _DRAIN_NOTE,
    find_job_actor,
    launch_job_runner,
    mark_node_draining,
)

log = logging.getLogger(__name__)


def _local_hostname_candidates() -> set[str]:
    names = set()
    try:
        fqdn = socket.getfqdn()
        if fqdn:
            names.add(fqdn)
    except Exception:
        pass
    try:
        hostname = socket.gethostname()
        if hostname:
            names.add(hostname)
            names.add(hostname.split(".", 1)[0])
    except Exception:
        pass
    return {name for name in names if name}


def _local_ip_candidates() -> set[str]:
    ips = set()
    try:
        result = subprocess.run(
            ["hostname", "-I"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            ips.update(ip.strip() for ip in result.stdout.split() if ip.strip())
    except Exception:
        pass
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip:
            ips.add(ip)
    except Exception:
        pass
    return ips


def _resolve_local_node() -> Optional[dict]:
    local_ips = _local_ip_candidates()
    local_names = _local_hostname_candidates()

    for node in ray.nodes():
        if not node.get("Alive", False):
            continue
        node_ip = node.get("NodeManagerAddress") or ""
        node_name = node.get("NodeName") or ""
        node_name_short = node_name.split(".", 1)[0]
        if node_ip in local_ips or node_name in local_names or node_name_short in local_names:
            return node
    return None


def _default_workdir() -> str:
    try:
        cfg = load_config()
        if cfg.env.working_dir:
            return cfg.env.working_dir
    except Exception:
        pass
    return os.getcwd()


def _wait_for_actor_exit(job_id: str, timeout_sec: float = 30.0, poll_interval: float = 0.2) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if find_job_actor(job_id) is None:
            return True
        time.sleep(poll_interval)
    return find_job_actor(job_id) is None


def drain_current_node(
    ray_head: str | None = None,
    *,
    state_path: str | None = None,
    resource_name: str = "",
    slurm_job_id: str = "",
) -> dict:
    """Preempt jobs on the local worker node before SLURM walltime."""
    if not ray_head:
        ray_head = load_runtime_ray_head(state_path)
    if not ray_head:
        raise RuntimeError("Could not determine current Ray head address")
    ray.init(address=ray_head, namespace="hive", ignore_reinit_error=True)
    try:
        DRAINING_NODES_DIR.mkdir(parents=True, exist_ok=True)
        node = _resolve_local_node()
        if node is None:
            raise RuntimeError("Could not resolve the local Ray node for draining")

        node_id = node["NodeID"]
        mark_node_draining(
            node_id,
            resource_name=resource_name,
            slurm_job_id=slurm_job_id,
        )

        preempted_jobs = []
        workdir = _default_workdir()
        for record in JobRecord.load_active():
            if record.node_id != node_id:
                continue
            if record.status == "running":
                result, _ = JobRecord.mark_preempted(record.job_id, note=_DRAIN_NOTE)
                if result not in ("preempted", "terminal"):
                    raise RuntimeError(f"Could not mark job {record.job_id} preempted: {result}")
                preempted_jobs.append(record.job_id)

        actor_kills = 0
        for job_id in preempted_jobs:
            actor = find_job_actor(job_id)
            if actor is None:
                continue
            try:
                ray.kill(actor)
                actor_kills += 1
            except Exception:
                pass

        requeued_jobs = []
        for job_id in preempted_jobs:
            if not _wait_for_actor_exit(job_id):
                raise RuntimeError(f"Timed out waiting for drained actor {job_id} to exit")

            result, record = JobRecord.requeue_preempted(job_id)
            if result != "queued" or record is None:
                raise RuntimeError(f"Could not requeue drained job {job_id}: {result}")

            try:
                launch_result, _updated = launch_job_runner(record, workdir=workdir)
                if launch_result == "launched":
                    pass
                elif launch_result == "already_launched":
                    log.info("Job %s was already re-dispatched while node drain was running", job_id)
                elif launch_result == "launch_pending":
                    log.info("Job %s still has a recent pending relaunch while node drain is finishing", job_id)
                else:
                    raise RuntimeError(f"launch returned {launch_result}")
            except Exception as exc:
                raise RuntimeError(f"Could not relaunch drained job {job_id}: {exc}") from exc
            requeued_jobs.append(job_id)

        log.warning(
            "Draining node %s before walltime (%s): preempted_jobs=%s requeued_jobs=%s actor_kills=%d",
            node_id[:12],
            slurm_job_id or "unknown",
            preempted_jobs,
            requeued_jobs,
            actor_kills,
        )
        return {
            "node_id": node_id,
            "preempted_jobs": preempted_jobs,
            "requeued_jobs": requeued_jobs,
            "actor_kills": actor_kills,
        }
    finally:
        try:
            ray.shutdown()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Drain a local Hive worker node before SLURM walltime")
    parser.add_argument("--ray-head", default=None, help="Ray head address")
    parser.add_argument(
        "--state-path",
        default=str(STATE_PATH),
        help="Shared Hive state.json path used to resolve the current Ray head",
    )
    parser.add_argument("--resource-name", default="", help="Hive resource name for logging")
    parser.add_argument("--slurm-job-id", default="", help="Current SLURM worker job id")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    try:
        drain_current_node(
            args.ray_head,
            state_path=args.state_path,
            resource_name=args.resource_name,
            slurm_job_id=args.slurm_job_id,
        )
        return 0
    except Exception as exc:
        log.error("Node drain failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
