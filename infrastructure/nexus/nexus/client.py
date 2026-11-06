"""Nexus Python client: programmatic access to a Nexus daemon.

Thin HTTP client for the project and read-only job routes of the daemon
REST API (register, inspect and delete projects; list and inspect jobs;
daemon status). It does not submit, wait for, continue or cancel jobs.
No dependencies beyond the Python standard library.

Usage:
    from nexus.client import NexusClient

    client = NexusClient("http://localhost:39000")
    client.register_project("my-project", max_gpus=2, workspace_dir="/ws", sif_path="/sif")
    jobs = client.list_jobs("my-project")
    client.delete_project("my-project")
"""

import json
import urllib.error
import urllib.request
from typing import List, Optional


class NexusClient:
    """Client for a remote Nexus daemon."""

    def __init__(self, url: str):
        """Connect to a Nexus daemon.

        Args:
            url: Daemon URL, e.g. 'http://localhost:39000'
        """
        self.url = url.rstrip("/")

    def _get(self, path: str) -> dict:
        """GET a JSON endpoint."""
        req = urllib.request.Request(f"{self.url}{path}")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read().decode())
                return body
            except Exception:
                raise RuntimeError(f"HTTP {e.code}: {e.reason}")

    def _post(self, path: str, data: dict) -> dict:
        """POST JSON to an endpoint."""
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{self.url}{path}", data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read().decode())
                return body
            except Exception:
                raise RuntimeError(f"HTTP {e.code}: {e.reason}")

    def _delete(self, path: str) -> dict:
        """DELETE a JSON endpoint."""
        req = urllib.request.Request(f"{self.url}{path}", method="DELETE")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read().decode())
                return body
            except Exception:
                raise RuntimeError(f"HTTP {e.code}: {e.reason}")

    def register_project(
        self,
        project_id: str,
        max_gpus: int = 4,
        workspace_dir: str = "",
        sif_path: str = "",
        default_job_timeout: int = 3600,
        soft_gpu_budget_hours: float = 0,
        shared_dirs: Optional[List] = None,
        max_cpus: int = 0,
        max_memory_mb: int = 0,
    ) -> dict:
        """Register a new project with the Nexus daemon.

        Args:
            project_id: Unique project identifier.
            max_gpus: Maximum GPUs that can be in use simultaneously.
            workspace_dir: Agent workspace directory (bind-mounted into containers).
            sif_path: Path to the Apptainer SIF image.
            default_job_timeout: Default per-job timeout in seconds.
            soft_gpu_budget_hours: Soft GPU-time budget in hours (0 = unlimited).
            shared_dirs: Additional directories to bind-mount.
            max_cpus: Maximum CPUs per job (0 = unlimited).
            max_memory_mb: Maximum memory per job in MB (0 = unlimited).

        Returns:
            Response dict with project_id and status.
        """
        payload = {
            "project_id": project_id,
            "max_gpus": max_gpus,
            "workspace_dir": workspace_dir,
            "sif_path": sif_path,
            "default_job_timeout": default_job_timeout,
            "soft_gpu_budget_hours": soft_gpu_budget_hours,
        }
        if shared_dirs:
            payload["shared_dirs"] = shared_dirs
        if max_cpus:
            payload["max_cpus"] = max_cpus
        if max_memory_mb:
            payload["max_memory_mb"] = max_memory_mb

        result = self._post("/api/projects", payload)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def delete_project(self, project_id: str) -> dict:
        """Delete a project and cancel all its running jobs.

        Args:
            project_id: Project to delete.

        Returns:
            Response dict with project_id and status.
        """
        result = self._delete(f"/api/projects/{project_id}")
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def get_project(self, project_id: str) -> dict:
        """Get project details.

        Args:
            project_id: Project to look up.

        Returns:
            Full project dict including job_counts.
        """
        result = self._get(f"/api/projects/{project_id}")
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def list_jobs(self, project_id: str) -> list:
        """List all jobs for a project.

        Args:
            project_id: Project whose jobs to list.

        Returns:
            List of job dicts.
        """
        result = self._get(f"/api/projects/{project_id}/jobs")
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(result["error"])
        if isinstance(result, dict) and "jobs" in result:
            return result["jobs"]
        if isinstance(result, list):
            return result
        return []

    def get_job(self, run_id: str) -> dict:
        """Get job details.

        Args:
            run_id: Job run ID.

        Returns:
            Job dict with status, timing and return code. While the job
            is active it also carries the current ``stdout`` and
            ``stderr`` from Hive.
        """
        result = self._get(f"/api/jobs/{run_id}")
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def status(self) -> dict:
        """Get daemon status.

        Returns:
            Dict with the sections ``daemon`` (uptime, pid), ``hive``
            (url, connected, GPU and node counts), ``projects`` and
            ``jobs`` (counts).
        """
        return self._get("/api/status")
