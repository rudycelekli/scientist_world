"""Minimal Nexus client: per-run project lifecycle and job submit/wait.

Standard-library ``urllib`` only. The calls the harness needs:

    create_project  register a per-run project (budget, GPU cap, resources)
    update_project  point the project at a rebuilt container / overlay
    delete_project  remove it after the run
    submit_job      submit the agent's compute job
    wait_for_job    long-poll until a terminal or checkpoint event
    get_job         status, return code and log paths
    cancel_job      cancel a job
    continue_job    resume a job after a checkpoint review
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

log = logging.getLogger(__name__)


class Nexus:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    # ── HTTP helpers ─────────────────────────────────────────────────

    def _get(self, path: str, timeout: int = 15) -> dict:
        req = urllib.request.Request(f"{self.url}{path}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def _post(self, path: str, data: dict, timeout: int = 30) -> dict:
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{self.url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def _delete(self, path: str, timeout: int = 15) -> dict:
        req = urllib.request.Request(f"{self.url}{path}", method="DELETE")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def _patch(self, path: str, data: dict, timeout: int = 30) -> dict:
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{self.url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="PATCH",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    # ── projects ─────────────────────────────────────────────────────

    def create_project(
        self,
        project_id: str,
        workspace_dir: str,
        sif_path: str,
        gpu_budget: float,
        max_gpus: int,
        allowed_resources: list[str],
        overlay_path: str = "",
        checkpoint_interval: int | None = None,
        checkpoint_timeout: int | None = None,
    ) -> dict:
        payload = {
            "project_id": project_id,
            "workspace_dir": workspace_dir,
            "sif_path": sif_path,
            "overlay_path": overlay_path,
            "soft_gpu_budget_hours": gpu_budget,
            "max_gpus": max_gpus,
            "allowed_resources": allowed_resources,
        }
        if checkpoint_interval is not None:
            payload["checkpoint_interval"] = int(checkpoint_interval)
        if checkpoint_timeout is not None:
            payload["checkpoint_timeout"] = int(checkpoint_timeout)
        result = self._post("/api/projects", payload)
        if "error" in result:
            raise RuntimeError(f"Nexus create_project error: {result['error']}")
        log.info("Created Nexus project %s", project_id)
        return result

    def update_project(
        self,
        project_id: str,
        *,
        sif_path: str | None = None,
        overlay_path: str | None = None,
        gpu_seconds_used: float | None = None,
    ) -> dict:
        """PATCH project fields.

        Used to point Nexus at a freshly rebuilt container.sif / overlay.img
        after the agent edits container.def or setup.sh.
        ``gpu_seconds_used`` resets the accumulated GPU-time counter; only
        the (disabled) in-run test phase uses it.
        """
        payload: dict = {}
        if sif_path is not None:
            payload["sif_path"] = sif_path
        if overlay_path is not None:
            payload["overlay_path"] = overlay_path
        if gpu_seconds_used is not None:
            payload["gpu_seconds_used"] = float(gpu_seconds_used)
        if not payload:
            return {}
        try:
            return self._patch(f"/api/projects/{project_id}", payload)
        except (urllib.error.URLError, OSError) as exc:
            log.warning("update_project(%s) failed: %s", project_id, exc)
            return {"error": str(exc)}

    def delete_project(self, project_id: str) -> dict:
        try:
            return self._delete(f"/api/projects/{project_id}")
        except (urllib.error.URLError, OSError) as exc:
            log.warning("delete_project(%s) failed: %s", project_id, exc)
            return {"error": str(exc)}

    # ── jobs ─────────────────────────────────────────────────────────

    def submit_job(
        self,
        project_id: str,
        command: str,
        num_gpus: int = 1,
        resource: str | None = None,
    ) -> tuple[str, str]:
        """POST a job, retrying through transient scheduler backpressure.

        Submission is synchronous end to end (Nexus forwards to Hive and
        waits), so under many concurrent submissions a request can time
        out inside Nexus and come back as HTTP 500 although nothing is
        broken. Transient failures (HTTP 5xx, ``URLError``, socket
        timeout, ``OSError``) are retried with bounded backoff for up to
        ten minutes. HTTP 4xx is permanent (missing project, malformed
        payload) and raised immediately, except the two 403 cases below
        that clear on their own.

        Every retry sends the same ``client_request_id`` so the daemon
        can de-duplicate: without it, a client-side timeout that fires
        while the daemon is still processing the original POST would
        acquire GPUs twice for one job.

        Returns ``(run_id, hive_job_id)``; the latter may be empty when
        Hive has not assigned an id yet.
        """
        import time
        import uuid

        client_request_id = str(uuid.uuid4())
        payload: dict = {
            "command": command,
            "num_gpus": num_gpus,
            "client_request_id": client_request_id,
        }
        if resource:
            payload["resource"] = resource

        deadline = time.monotonic() + 600.0  # 10-minute wall cap
        backoff_schedule = [5, 10, 20, 30, 60]
        attempt = 0
        last_exc: Exception | None = None

        while True:
            try:
                result = self._post(
                    f"/api/projects/{project_id}/jobs", payload, timeout=120,
                )
            except urllib.error.HTTPError as exc:
                # 5xx is transient. 4xx is permanent, except two 403
                # cases that are retried like a 5xx:
                #   - "budget exhausted": clears when the project's
                #     budget is raised or its GPU-time counter is reset;
                #   - "N/N in use, requested K": the project is at
                #     max_gpus because one of its own jobs has not been
                #     released yet; it clears within seconds.
                if 400 <= exc.code < 500:
                    body = ""
                    try:
                        body = exc.read().decode("utf-8", errors="replace")
                    except Exception:
                        pass
                    body_l = body.lower()
                    if exc.code == 403 and (
                        "budget" in body_l or "in use" in body_l
                    ):
                        last_exc = exc
                    else:
                        raise RuntimeError(
                            f"Nexus submit_job rejected "
                            f"(HTTP {exc.code}): {body or exc}"
                        ) from exc
                else:
                    last_exc = exc
            except (urllib.error.URLError, OSError) as exc:
                last_exc = exc
            else:
                if "error" in result:
                    raise RuntimeError(f"Nexus submit error: {result['error']}")

                run_id = result.get("run_id")
                if not run_id:
                    raise RuntimeError(
                        f"Nexus submit response missing run_id: {result}"
                    )
                hive_job_id = result.get("hive_job_id") or ""
                log.info(
                    "Submitted job %s (hive=%s) to %s (gpus=%d)",
                    run_id, hive_job_id or "?", project_id, num_gpus,
                )
                return run_id, hive_job_id

            now = time.monotonic()
            if now >= deadline:
                break
            delay = backoff_schedule[min(attempt, len(backoff_schedule) - 1)]
            delay = min(delay, max(1.0, deadline - now))
            attempt += 1
            log.warning(
                "submit_job retry %d after %.0fs (last: %s)",
                attempt, delay, last_exc,
            )
            time.sleep(delay)

        raise RuntimeError(
            f"Nexus submit_job failed after {attempt + 1} attempts over "
            f"~{int(time.monotonic() - (deadline - 600.0))}s "
            f"(last error: {last_exc!r}); check the Nexus daemon and the "
            f"Hive pool behind it"
        )

    def wait_for_job(
        self,
        project_id: str,
        run_id: str,
        timeout: int = 600,
    ) -> dict:
        """Long-poll for a terminal event on a specific run."""
        path = f"/api/projects/{project_id}/wait?timeout={timeout}&run_id={run_id}"
        try:
            return self._get(path, timeout=timeout + 30)
        except (urllib.error.URLError, OSError) as exc:
            log.warning("wait_for_job(%s) failed: %s", run_id, exc)
            return {"event": "error", "error": str(exc)}

    def get_job(self, run_id: str) -> dict:
        try:
            return self._get(f"/api/jobs/{run_id}")
        except (urllib.error.URLError, OSError) as exc:
            return {"status": "unknown", "error": str(exc)}

    def cancel_job(self, run_id: str) -> bool:
        try:
            result = self._post(f"/api/jobs/{run_id}/cancel", {})
            return "error" not in result
        except (urllib.error.URLError, OSError):
            return False

    def continue_job(self, run_id: str) -> dict:
        """Tell Nexus to resume a job after a checkpoint review.

        Used when the agent has inspected the logs of a running job
        (``on_checkpoint.md``) and decided to keep it. Nexus moves the
        job back from ``awaiting_response`` to ``running``.
        """
        try:
            return self._post(f"/api/jobs/{run_id}/continue", {})
        except (urllib.error.URLError, OSError) as exc:
            log.warning("continue_job(%s) failed: %s", run_id, exc)
            return {"error": str(exc)}
