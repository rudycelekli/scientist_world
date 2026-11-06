"""Nexus job lifecycle and workspace management.

A NexusJob tracks a single compute job submitted through a project.
Jobs are persisted as JSON files under JOBS_DIR (Nexus's internal state),
and also write metadata, logs, and checkpoints into the agent's workspace
so the agent can observe job progress without querying the daemon.

Workspace structure (under {workspace_dir}/jobs/{run_id}/):
    meta.json               Job metadata (command, resources, timing)
    stdout.log              Latest stdout from the running/completed job
    stderr.log              Latest stderr from the running/completed job
    result.json             Final outcome (only after completion)
    checkpoints/
        001.json            Periodic checkpoint with timestamp + log snapshot
        002.json
        ...
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from .config import JOBS_DIR


@dataclass
class NexusJob:
    """A single compute job within a Nexus project.

    Attributes:
        run_id: Unique identifier for this job run.
        project_id: The project this job belongs to.
        command: Shell command to execute on the compute node.
        num_gpus: Number of GPUs requested.
        num_cpus: Number of CPUs requested.
        timeout_sec: Per-job timeout in seconds (None = use project default).
        resource: Hive resource name(s) the job was submitted to
            (comma-separated; empty = any).
        actual_resource: Hive resource the job actually ran on, as
            reported by Hive when the job ends.
        hive_job_id: Corresponding job ID in the Hive system.
        status: Current job state (queued, running, completed, failed,
                cancelled, awaiting_response).
        submitted_at: Unix timestamp when the job was submitted.
        started_at: Unix timestamp when the job began executing.
        completed_at: Unix timestamp when the job finished.
        returncode: Process exit code (None if not yet completed).
        last_checkpoint_at: Unix timestamp of the most recent checkpoint.
        checkpoint_count: Total number of checkpoints written.
    """
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    project_id: str = ""
    command: str = ""
    num_gpus: int = 1
    num_cpus: int = 1
    timeout_sec: Optional[int] = None
    resource: Optional[str] = None
    actual_resource: Optional[str] = None
    hive_job_id: Optional[str] = None
    status: str = "queued"
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    returncode: Optional[int] = None
    last_checkpoint_at: Optional[float] = None
    checkpoint_count: int = 0

    def to_dict(self) -> dict:
        """Serialize job to a JSON-safe dictionary."""
        d = {
            "run_id": self.run_id,
            "project_id": self.project_id,
            "command": self.command,
            "num_gpus": self.num_gpus,
            "num_cpus": self.num_cpus,
            "status": self.status,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "returncode": self.returncode,
            "last_checkpoint_at": self.last_checkpoint_at,
            "checkpoint_count": self.checkpoint_count,
        }
        if self.timeout_sec is not None:
            d["timeout_sec"] = self.timeout_sec
        if self.resource is not None:
            d["resource"] = self.resource
        if self.actual_resource is not None:
            d["actual_resource"] = self.actual_resource
        if self.hive_job_id is not None:
            d["hive_job_id"] = self.hive_job_id
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "NexusJob":
        """Deserialize a job from a dictionary."""
        return cls(
            run_id=d.get("run_id", uuid.uuid4().hex[:8]),
            project_id=d.get("project_id", ""),
            command=d.get("command", ""),
            num_gpus=d.get("num_gpus", 1),
            num_cpus=d.get("num_cpus", 1),
            timeout_sec=d.get("timeout_sec"),
            resource=d.get("resource"),
            actual_resource=d.get("actual_resource"),
            hive_job_id=d.get("hive_job_id"),
            status=d.get("status", "queued"),
            submitted_at=d.get("submitted_at", 0.0),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            returncode=d.get("returncode"),
            last_checkpoint_at=d.get("last_checkpoint_at"),
            checkpoint_count=d.get("checkpoint_count", 0),
        )

    @property
    def _file_path(self) -> Path:
        """Path to this job's JSON file in Nexus internal storage."""
        return Path(JOBS_DIR) / f"{self.run_id}.json"

    def save(self):
        """Persist job state to Nexus internal storage."""
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, run_id: str) -> Optional["NexusJob"]:
        """Load a job from Nexus internal storage by run_id.

        Returns None if not found.
        """
        path = Path(JOBS_DIR) / f"{run_id}.json"
        if not path.exists():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text()))
        except (json.JSONDecodeError, KeyError):
            return None

    @classmethod
    def load_all(cls) -> List["NexusJob"]:
        """Load all jobs from Nexus internal storage."""
        d = Path(JOBS_DIR)
        if not d.exists():
            return []
        jobs = []
        for path in sorted(d.glob("*.json")):
            try:
                jobs.append(cls.from_dict(json.loads(path.read_text())))
            except Exception:
                pass
        return jobs

    @classmethod
    def load_by_project(cls, project_id: str) -> List["NexusJob"]:
        """Load all jobs belonging to a specific project."""
        return [j for j in cls.load_all() if j.project_id == project_id]

    def delete(self):
        """Remove this job's JSON file from Nexus internal storage."""
        if self._file_path.exists():
            self._file_path.unlink()

    # ── Workspace writes ────────────────────────────────────────────
    #
    # These methods write into the agent's workspace directory so the
    # agent can observe job progress without querying the Nexus daemon.

    def _workspace_job_dir(self, workspace_dir: str) -> Path:
        """Path to this job's directory inside the agent workspace."""
        return Path(workspace_dir) / "jobs" / self.run_id

    def write_meta(self, workspace_dir: str):
        """Write job metadata to the agent workspace.

        Creates {workspace_dir}/jobs/{run_id}/meta.json with the current
        job state. Called once at submission and updated on status changes.
        """
        job_dir = self._workspace_job_dir(workspace_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        meta_path = job_dir / "meta.json"
        meta_path.write_text(json.dumps(self.to_dict(), indent=2))

    def write_logs(self, workspace_dir: str, stdout: str, stderr: str):
        """Write current stdout/stderr to the agent workspace.

        Creates or overwrites {workspace_dir}/jobs/{run_id}/stdout.log
        and stderr.log. Called periodically during job execution.
        """
        job_dir = self._workspace_job_dir(workspace_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "stdout.log").write_text(stdout)
        (job_dir / "stderr.log").write_text(stderr)

    def write_checkpoint(self, workspace_dir: str, stdout: str, stderr: str):
        """Write a numbered checkpoint to the agent workspace.

        Increments checkpoint_count and writes a checkpoint file at
        {workspace_dir}/jobs/{run_id}/checkpoints/{NNN}.json containing
        the timestamp and current log content.
        """
        self.checkpoint_count += 1
        self.last_checkpoint_at = time.time()

        job_dir = self._workspace_job_dir(workspace_dir)
        ckpt_dir = job_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_num = f"{self.checkpoint_count:03d}"
        checkpoint = {
            "checkpoint": self.checkpoint_count,
            "timestamp": self.last_checkpoint_at,
            "status": self.status,
            "stdout": stdout,
            "stderr": stderr,
        }
        (ckpt_dir / f"{checkpoint_num}.json").write_text(
            json.dumps(checkpoint, indent=2)
        )

    def write_result(self, workspace_dir: str, stdout: str, stderr: str,
                     returncode: int):
        """Write final job result to the agent workspace.

        Creates {workspace_dir}/jobs/{run_id}/result.json with the final
        status, exit code, duration, and log content. Called once when the
        job completes (successfully or not).
        """
        job_dir = self._workspace_job_dir(workspace_dir)
        job_dir.mkdir(parents=True, exist_ok=True)

        duration = None
        if self.started_at is not None:
            end = self.completed_at if self.completed_at is not None else time.time()
            duration = end - self.started_at

        result = {
            "run_id": self.run_id,
            "project_id": self.project_id,
            "command": self.command,
            "status": self.status,
            "returncode": returncode,
            "duration": duration,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "stdout": stdout,
            "stderr": stderr,
        }
        (job_dir / "result.json").write_text(json.dumps(result, indent=2))
