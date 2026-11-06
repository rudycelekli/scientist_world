"""Hive integration -- submit jobs, poll status, build apptainer commands.

Wraps the HTTP API of the Hive dashboard. Nexus uses this to submit
jobs, poll their status, and build ``apptainer exec`` commands from
project configuration. Stdlib only (urllib) -- no external dependencies.
"""

import json
import logging
import os
import time
import urllib.error
import urllib.request

log = logging.getLogger(__name__)


# Hive handles /api/submit synchronously, and one submit can take several
# seconds (longer when many clients submit at once). A short socket timeout
# would turn a slow submit into a spurious failure, so allow 60 s.
_HIVE_POST_TIMEOUT_SEC = 60

# Retry budget for transient Hive failures from submit() (5xx, URLError,
# socket.timeout, OSError). Bounded so that a Hive that stays unresponsive
# still surfaces as a clean failure to the caller.
_SUBMIT_RETRY_WALL_BUDGET_SEC = 600.0
_SUBMIT_RETRY_BACKOFF_SCHEDULE = (5, 10, 20, 30, 60)


def _shell_quote(s: str) -> str:
    """Quote a string for shell if it contains special characters.

    Safe characters (alphanumeric plus ``-_/=.:``) are passed through
    unquoted.  Everything else is wrapped in single quotes with internal
    single quotes escaped.
    """
    if s and all(c.isalnum() or c in "-_/=.:" for c in s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


class HiveBridge:
    """Client for the HTTP API of the Hive dashboard.

    Thin wrapper tailored to what Nexus needs: non-blocking submit,
    status poll, cancel, cluster status, and construction of the
    ``apptainer exec`` command from a project's configuration.

    Args:
        hive_url: Dashboard URL, e.g. ``http://<hive-host>:38581``.
    """

    def __init__(self, hive_url: str):
        self.url = hive_url.rstrip("/")

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str) -> dict:
        """GET a JSON endpoint.

        Raises:
            urllib.error.URLError: On connection failure.
        """
        req = urllib.request.Request(f"{self.url}{path}")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())

    def _post(self, path: str, data: dict) -> dict:
        """POST JSON to an endpoint and return the decoded JSON body.

        Raises:
            urllib.error.HTTPError: On a non-2xx response.
            urllib.error.URLError: On connection failure.
        """
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{self.url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_HIVE_POST_TIMEOUT_SEC) as resp:
            return json.loads(resp.read().decode())

    # ------------------------------------------------------------------
    # Job operations
    # ------------------------------------------------------------------

    def submit(
        self,
        command: str,
        num_gpus: int = 1,
        num_cpus: int = 1,
        timeout_sec: int = None,
        resource: str = None,
        max_retries: int = 3,
    ) -> str:
        """Submit a raw command to Hive.

        Retries internally on transient Hive responses (HTTP 5xx, 429,
        URLError, socket.timeout, OSError) up to a wall-time cap. Any
        other HTTP 4xx is treated as permanent and raised immediately.
        The retry sits *inside* submit() (not above it in the daemon)
        because:

        - Hive creates a job only on a successful POST, so a failed
          POST left nothing behind and retrying is safe.
        - The daemon's quota acquire/release straddles this call; if the
          caller retried instead, every attempt would re-acquire quota.

        Args:
            command: Shell command to run on the compute node.
            num_gpus: Number of GPUs to allocate.
            num_cpus: Number of CPUs to allocate.
            timeout_sec: Maximum wall-clock seconds before the job is
                killed.  ``None`` means no limit.
            resource: Target Hive resource name (optional).
            max_retries: Hive-side automatic re-run of the job after an
                infrastructure failure (node loss). Unrelated to this
                method's own retry, which covers the submit call only.

        Returns:
            The Hive-assigned job ID string.

        Raises:
            RuntimeError: If Hive returns a permanent error (4xx or a
                JSON ``error`` body), or if transient retries exhaust
                the wall-time budget.
            urllib.error.URLError: Surfaced only if the dashboard
                remains unreachable past the retry budget.
        """
        payload = {
            "command": command,
            "num_gpus": num_gpus,
            "num_cpus": num_cpus,
            "user": f"nexus/{os.environ.get('USER', 'unknown')}",
            "max_retries": max_retries,
        }
        if timeout_sec is not None:
            payload["timeout_sec"] = timeout_sec
        if resource:
            payload["resource"] = resource

        deadline = time.monotonic() + _SUBMIT_RETRY_WALL_BUDGET_SEC
        attempt = 0
        last_exc: Exception | None = None

        while True:
            try:
                result = self._post("/api/submit", payload)
            except urllib.error.HTTPError as exc:
                # 429 and 5xx are transient; any other 4xx is permanent.
                if exc.code == 429 or 500 <= exc.code < 600:
                    last_exc = exc
                else:
                    raise RuntimeError(
                        f"Hive submit rejected (HTTP {exc.code}): {exc}"
                    ) from exc
            except (urllib.error.URLError, OSError) as exc:
                last_exc = exc
            else:
                if "error" in result:
                    # An "error" key in a 200 body is a rejection by Hive
                    # (invalid request, backend down) that a quick retry
                    # would not clear, so it is treated as permanent.
                    raise RuntimeError(f"Hive submit error: {result['error']}")

                job_id = result["job_id"]
                if attempt:
                    log.info(
                        "Submitted to Hive after %d retries: "
                        "job_id=%s gpus=%d cmd=%.80s",
                        attempt, job_id, num_gpus, command,
                    )
                else:
                    log.info(
                        "Submitted to Hive: job_id=%s gpus=%d cmd=%.80s",
                        job_id, num_gpus, command,
                    )
                return job_id

            now = time.monotonic()
            if now >= deadline:
                break
            schedule = _SUBMIT_RETRY_BACKOFF_SCHEDULE
            delay = schedule[min(attempt, len(schedule) - 1)]
            delay = min(delay, max(1.0, deadline - now))
            attempt += 1
            log.warning(
                "Hive submit retry %d after %.0fs (last: %s)",
                attempt, delay, last_exc,
            )
            time.sleep(delay)

        raise RuntimeError(
            f"Hive submit failed after {attempt + 1} attempts "
            f"(~{int(_SUBMIT_RETRY_WALL_BUDGET_SEC)}s wall) "
            f"(last error: {last_exc!r}); Hive is not responding"
        )

    def poll(self, hive_job_id: str) -> dict:
        """Poll job status from Hive.

        Returns:
            The Hive ``/api/jobs/<id>/logs`` response: a dict with
            ``status``, ``stdout``, ``stderr``, ``returncode`` and timing
            keys.  On any HTTP or connection error, returns
            ``{"status": "unknown"}`` so the caller can retry on the next
            tick without raising.
        """
        try:
            return self._get(f"/api/jobs/{hive_job_id}/logs")
        except (urllib.error.URLError, OSError) as exc:
            log.debug("Poll error for %s: %s", hive_job_id, exc)
            return {"status": "unknown"}

    def cancel(self, hive_job_id: str) -> bool:
        """Cancel a job on Hive.

        Returns ``True`` if Hive accepted the cancel request.  On an
        HTTP error, an ``error`` body or a connection failure this logs
        a warning and returns ``False``; it never raises.  The caller
        marks the job cancelled in Nexus either way.
        """
        try:
            result = self._post(f"/api/jobs/{hive_job_id}/cancel", {})
            if "error" in result:
                log.warning("Hive cancel returned error for %s: %s",
                            hive_job_id, result["error"])
                return False
            log.info("Cancelled Hive job %s", hive_job_id)
            return True
        except urllib.error.HTTPError as exc:
            log.warning("Hive cancel returned HTTP %d for %s",
                        exc.code, hive_job_id)
            return False
        except (urllib.error.URLError, OSError) as exc:
            log.warning("Could not reach Hive to cancel %s: %s",
                        hive_job_id, exc)
            return False

    def cluster_status(self) -> dict:
        """Get Hive cluster status.

        Returns:
            Dict with ``total_gpus``, ``free_gpus``, and other fields
            from the Hive ``/api/status`` endpoint.  On connection error,
            returns a minimal dict with ``error`` key.
        """
        try:
            return self._get("/api/status")
        except (urllib.error.URLError, OSError) as exc:
            log.warning("Could not reach Hive for cluster status: %s", exc)
            return {"error": str(exc), "total_gpus": 0, "free_gpus": 0}

    # ------------------------------------------------------------------
    # Apptainer command construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_apptainer_command(
        project,
        command: str,
        num_gpus: int,
        working_directory: str = "/home/user",
    ) -> str:
        """Build an ``apptainer exec`` command string from project config.

        Constructs the full container invocation with isolation flags,
        GPU passthrough (when *num_gpus* > 0), workspace bind-mount,
        and any additional shared directories from the project.  The
        workspace is always mounted at ``/home/user``, and the model
        cache variables always point at ``/home/user/shared/hf_cache``
        (inside the workspace unless ``hf_cache_path`` is mounted there).

        Args:
            project: A ``Project`` instance providing ``sif_path``,
                ``workspace_dir``, ``overlay_path``, ``shared_dirs`` and
                ``hf_cache_path``.
            command: The user command to run inside the container.
            num_gpus: Number of GPUs; ``--nv`` is added when > 0.
            working_directory: ``cd`` target inside the container
                (default ``/home/user``).

        Returns:
            A shell-safe command string ready for Hive submission.
        """
        parts = ["apptainer", "exec"]

        if num_gpus > 0:
            parts.append("--nv")

        parts.extend(["--containall", "--cleanenv",
                      "--env", "HF_HOME=/home/user/shared/hf_cache",
                      "--env", "TRANSFORMERS_CACHE=/home/user/shared/hf_cache",
                      "--env", "HF_DATASETS_CACHE=/home/user/shared/hf_cache/datasets",
                      "--env", "TORCH_HOME=/home/user/shared/hf_cache/torch"])

        # Primary workspace bind
        parts.extend(["--bind", f"{project.workspace_dir}:/home/user"])

        # Overlay image holding packages installed on top of the SIF. Jobs
        # mount it read-only: nothing is installed at job time, and a
        # read-only mount avoids lock contention on a shared filesystem
        # when a job is killed.
        if project.overlay_path:
            parts.extend(["--overlay", f"{project.overlay_path}:ro"])

        # Additional shared directories (e.g., "/shared:/shared:ro")
        for shared in project.shared_dirs:
            parts.extend(["--bind", shared])

        # Shared model cache, writable so that the jobs of all projects
        # reuse downloaded models.
        if project.hf_cache_path:
            parts.extend(["--bind",
                          f"{project.hf_cache_path}:/home/user/shared/hf_cache"])

        # SIF image + inner command
        parts.extend([
            project.sif_path,
            "bash", "-c",
            f"cd {working_directory} && {command}",
        ])

        return " ".join(_shell_quote(p) for p in parts)

    # ------------------------------------------------------------------
    # High-level project-aware submission
    # ------------------------------------------------------------------

    def submit_for_project(
        self,
        project,
        command: str,
        num_gpus: int = 1,
        num_cpus: int = 1,
        timeout_sec: int = None,
        resource: str = None,
        max_retries: int = 3,
    ) -> str:
        """Submit a job for a project: wrap in apptainer, submit to Hive.

        Combines :meth:`build_apptainer_command` and :meth:`submit` into
        a single call.  The raw *command* is wrapped in an
        ``apptainer exec`` invocation using the project's SIF, workspace,
        and shared directories, then submitted to the Hive dashboard.

        Args:
            project: A ``Project`` instance.
            command: The user command to run inside the container.
            num_gpus: GPUs to request from Hive.
            num_cpus: CPUs to request from Hive.
            timeout_sec: Maximum wall-clock seconds (``None`` = no limit).
            resource: Hive resource name to target.

        Returns:
            The Hive-assigned job ID string.
        """
        apptainer_cmd = self.build_apptainer_command(
            project, command, num_gpus,
        )
        return self.submit(
            apptainer_cmd,
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            timeout_sec=timeout_sec,
            resource=resource,
            max_retries=max_retries,
        )
