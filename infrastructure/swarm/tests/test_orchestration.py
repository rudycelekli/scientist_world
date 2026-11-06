"""One paper through the scheduler, with a scripted agent and a fake Nexus.

The agent CLI, the Nexus daemon and the container runtime are replaced by
in-process stand-ins; everything else is real: workspace creation from the
worker template, the action loop, manifest writing, the validation gate,
git commits, pushes (to a local bare repository) and the workflow gate.
``validate.py`` is not executed; its return code is scripted.
"""

import json
import shutil
import subprocess
import threading
from collections import deque
from pathlib import Path

import pytest
import yaml

from infrastructure.worker.codex import CodexWorker
from swarm import scheduler as scheduler_mod
from swarm import workspace as workspace_mod
from swarm.config import load_config
from swarm.scheduler import Scheduler

ARXIV_ID = "2512.01152"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs the git executable")


class ScriptedAgent(CodexWorker):
    """A worker whose CLI call is replaced by a script of action.yaml edits."""

    script: list = []
    prompts: list = []

    def _invoke(self, prompt: str) -> str:
        self._turn_count += 1
        ScriptedAgent.prompts.append(prompt)
        action = ScriptedAgent.script.pop(0) if ScriptedAgent.script else None
        if action is not None:
            (Path(self._workspace_path) / "action.yaml").write_text(yaml.safe_dump(action))
        self._save_raw_output(self._turn_count, json.dumps({"turn": self._turn_count}))
        self._session_id = "session-1"
        self._save_session_info()
        return "ok"


class FakeNexus:
    """Records calls and reports every job as completed with return code 0."""

    def __init__(self):
        self.calls = []
        self.gpu_seconds_used = 1800

    def delete_project(self, project_id):
        self.calls.append(("delete_project", project_id))
        return {}

    def create_project(self, **kwargs):
        self.calls.append(("create_project", kwargs))
        return {}

    def update_project(self, project_id, **kwargs):
        self.calls.append(("update_project", kwargs))
        return {}

    def submit_job(self, project_id, command, num_gpus=1, timeout_sec=None, resource=None):
        self.calls.append(("submit_job", {
            "project_id": project_id, "command": command, "num_gpus": num_gpus,
            "timeout_sec": timeout_sec, "resource": resource,
        }))
        return "run-1", "hive-1"

    def wait_for_job(self, project_id, timeout=600, run_id=""):
        return {"event": "completed", "run_id": run_id, "status": "completed"}

    def get_job(self, run_id):
        return {"returncode": 0, "started_at": 0}

    def get_project(self, project_id):
        return {"gpu_seconds_used": self.gpu_seconds_used}

    def named(self, name):
        return [payload for call, payload in self.calls if call == name]


class RecordingExecutor:
    """Stands in for the thread pool: collects jobs instead of running them."""

    _max_workers = 64

    def __init__(self):
        self.pending = []

    def submit(self, fn, job):
        self.pending.append((fn, job))

    def run_next(self):
        fn, job = self.pending.pop(0)
        fn(job)
        return job


def _git(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


def _headings(prompts):
    """The first line of each prompt sent to the agent."""
    return [prompt.lstrip().splitlines()[0] for prompt in prompts]


@pytest.fixture
def harness(tmp_path, tmp_swarm_dir, monkeypatch):
    """A scheduler without daemon side effects, wired to the fakes."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setattr(workspace_mod, "_GIT_ENV", scheduler_mod._git_env())

    data = tmp_path / "data"
    md_dir = data / "arxiv_mds" / ARXIV_ID
    md_dir.mkdir(parents=True)
    (md_dir / f"{ARXIV_ID}.md").write_text("# A Test Paper\n\nBody.\n")

    def configure(steps, max_revert_attempts=0):
        (tmp_swarm_dir / "config.yaml").write_text(yaml.safe_dump({
            "scout_data_dir": str(data),
            "workspace_base_dir": str(tmp_path / "workspaces"),
            "shared_datasets_path": str(tmp_path / "datasets"),
            "shared_models_path": str(tmp_path / "models"),
            "local_tmp_dir": str(tmp_path / "local"),
            "github_org": "",
            "default_workflow": "w",
            "profiles": {"p": {"scaffold": "codex", "model": "test-model"}},
            "workflows": {"w": {"max_revert_attempts": max_revert_attempts, "steps": steps}},
            "nexus": {
                "url": "http://localhost:1",
                "max_gpus": 2,
                "resources": {"gpu": ["gpu-a"], "gpu_test": ["gpu-a-test"],
                              "cpu": ["cpu-a"], "cpu_test": ["cpu-a-test"]},
            },
        }))
        sched = Scheduler.__new__(Scheduler)
        sched.config = load_config()
        sched.queue = deque()
        sched.active = {}
        sched._finalizing = {}
        sched.completed = {}
        sched._lock = threading.Lock()
        sched._capacity_cv = threading.Condition(sched._lock)
        sched._capacity_state = {}
        sched._cap_overrides = {}
        sched._working_priority_waiters = deque()
        sched._working_fresh_waiters = deque()
        sched._working_waiter_kind = {}
        sched._shutdown = False
        sched._cancelled = set()
        sched._active_workers = {}
        sched._nexus_projects = set()
        sched._child_pids = set()
        sched._local_sifs = {}
        sched._executor = RecordingExecutor()
        sched.nexus = FakeNexus()
        return sched

    ScriptedAgent.script = []
    ScriptedAgent.prompts = []
    validate_results = []
    container_steps = []

    real_run = subprocess.run

    def run(cmd, *args, **kwargs):
        if len(cmd) > 1 and str(cmd[1]).endswith("validate.py"):
            code = validate_results.pop(0) if validate_results else 0
            return subprocess.CompletedProcess(cmd, code, stdout="validator output", stderr="")
        if cmd[0] == "apptainer" and cmd[1] == "exec" and "*.dist-info" in cmd[-1]:
            # The package scan of the finished environment.
            container_steps.append("scan")
            return subprocess.CompletedProcess(
                cmd, 0, stdout="===DISTINFO===\nName: numpy\nVersion: 1.26.4\n", stderr="")
        return real_run(cmd, *args, **kwargs)

    def build_container(self, container_def, container_sif, build_log_path, arxiv_id=""):
        container_steps.append("build")
        Path(container_sif).write_bytes(b"sif")
        return True

    def run_setup_in_overlay(self, container_sif, overlay_img, setup_sh, log_path):
        container_steps.append("setup")
        Path(overlay_img).write_bytes(b"overlay")
        return True

    monkeypatch.setattr(scheduler_mod.subprocess, "run", run)
    monkeypatch.setattr(Scheduler, "_build_container", build_container)
    monkeypatch.setattr(Scheduler, "_run_setup_in_overlay", run_setup_in_overlay)
    monkeypatch.setattr(scheduler_mod.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        scheduler_mod, "_create_worker",
        lambda scaffold, model, config: ScriptedAgent(scaffold, model, config),
    )

    remote = tmp_path / "remote.git"

    def add_remote():
        """Give new workspaces a local bare repository as ``origin``."""
        real_run(["git", "init", "--bare", "-b", "main", str(remote)], capture_output=True, check=True)
        real_create = scheduler_mod.create_workspace

        def create_with_remote(*args, **kwargs):
            path = real_create(*args, **kwargs)
            real_run(["git", "-C", path, "remote", "add", "origin", str(remote)],
                     capture_output=True, check=True)
            return path

        monkeypatch.setattr(scheduler_mod, "create_workspace", create_with_remote)
        return remote

    return type("Harness", (), {
        "configure": staticmethod(configure),
        "add_remote": staticmethod(add_remote),
        "validate_results": validate_results,
        "container_steps": container_steps,
        "workspace": tmp_path / "workspaces" / ARXIV_ID,
        "papers": tmp_swarm_dir / "papers",
    })


def _meta(harness):
    return json.loads((harness.papers / ARXIV_ID / "meta.json").read_text())


SUBMIT = {"action": "submit", "gpus": 1, "test": False, "milestone": "method_runs", "notes": "run"}
DONE = {"action": "done", "milestone": "core_claim", "notes": "finished"}
TWO_STEPS = [
    {"profile": "p", "gpu_budget_hours": 1.0, "min_milestone": "core_claim"},
    {"profile": "p", "gpu_budget_hours": 0.5, "role": "verifier"},
]


def test_two_iterations_with_job_push_and_advance(harness):
    remote = harness.add_remote()
    sched = harness.configure(TWO_STEPS)

    # Iteration 1: the reproducer submits one job, then signals done.
    ScriptedAgent.script = [SUBMIT, DONE]
    job = sched.enqueue(ARXIV_ID, date="2025-12-01")
    assert job.workflow == "w" and job.profile == "p" and job.gpu_budget == 1.0
    sched._executor.run_next()

    # The agent is not sent a wrap-up prompt after a plain "done".
    assert _headings(ScriptedAgent.prompts) == ["# Reproducer Agent", "# Job Completed"]
    assert harness.container_steps == ["build", "setup"]

    created = sched.nexus.named("create_project")[0]
    assert created["project_id"] == f"swarm-{ARXIV_ID}"
    assert created["gpu_budget"] == 1.0 and created["max_gpus"] == 2
    assert created["allowed_resources"] == ["gpu-a", "gpu-a-test", "cpu-a", "cpu-a-test"]
    assert created["shared_dirs"][0].endswith(":/home/user/shared/datasets:ro")
    # No timeout is passed with the job: Nexus applies its own.
    assert sched.nexus.named("submit_job") == [{
        "project_id": f"swarm-{ARXIV_ID}", "command": "bash /home/user/scripts/run.sh",
        "num_gpus": 1, "timeout_sec": None, "resource": "gpu-a",
    }]
    assert sched.nexus.calls[-1] == ("delete_project", f"swarm-{ARXIV_ID}")

    subjects = _git(remote, "log", "--format=%s", "main").splitlines()
    assert subjects[0] == "End of Iteration 1: core_claim [p (reproducer)]"
    assert "Job #1 submission (1 GPU)" in subjects
    assert "Job #1 completed (run-1, rc=0)" in subjects
    assert "Start of Iteration 1: none [p (reproducer)]" in subjects
    assert subjects[-1] == "Initial workspace"
    pushed = _git(remote, "ls-tree", "-r", "--name-only", "main").splitlines()
    for path in ("manifest.json", "README.md", "validate.py", "action.yaml",
                 "trajectory/turn_0001_raw.jsonl", "trajectory/session.json",
                 f"paper/{ARXIV_ID}.md"):
        assert path in pushed, path
    assert "AGENTS.md" not in pushed and "protocol.md" not in pushed

    manifest = json.loads((harness.workspace / "manifest.json").read_text())
    assert manifest["paper"] == {
        "arxiv_id": ARXIV_ID, "title": "A Test Paper", "date": "2025-12-01", "topics": None,
    }
    agent = manifest["agents"][0]
    assert (agent["role"], agent["iteration"], agent["milestone"]) == ("reproducer", 1, "core_claim")
    assert (agent["gpu_budget"], agent["gpu_hours_used"]) == (1.0, 0.5)

    # The gate passed, so the next step is queued and dispatched.
    meta = _meta(harness)
    assert (meta["status"], meta["iteration"], meta["role"]) == ("active", 2, "verifier")
    assert (meta["milestone"], meta["previous_milestone"]) == ("core_claim", "core_claim")
    assert (meta["gpu_budget"], meta["gpu_hours_used"], meta["total_gpu_hours_used"]) == (0.5, 0.0, 0.5)

    # Iteration 2: the verifier, last step of the workflow.
    ScriptedAgent.prompts.clear()
    ScriptedAgent.script = [None, DONE]
    sched._executor.run_next()

    # A later reproducer or verifier iteration gets the start prompt again,
    # followed by the prompt of its role.
    assert _headings(ScriptedAgent.prompts) == ["# Reproducer Agent", "# Final Verification"]
    assert _git(remote, "log", "-1", "--format=%s", "main").strip() == \
        "End of Iteration 2: core_claim [p (verifier)]"
    assert sched.completed[ARXIV_ID].status == "done"
    assert _meta(harness)["status"] == "done"
    assert not sched._executor.pending and not sched.queue
    assert len(json.loads((harness.workspace / "manifest.json").read_text())["agents"]) == 2

    # The container is built once; the unchanged definition is not rebuilt.
    assert harness.container_steps.count("build") == 1
    assert harness.container_steps.count("setup") == 1

    # The frozen definition pins what the package scan found. It is written
    # after the last push and stays local.
    final_def = harness.workspace / "environment" / "container.final.def"
    assert "scan" in harness.container_steps
    assert "'numpy==1.26.4'" in final_def.read_text()
    pushed = _git(remote, "ls-tree", "-r", "--name-only", "main").splitlines()
    assert "environment/container.def" in pushed
    assert "environment/container.final.def" not in pushed


def test_without_a_remote_the_iteration_stalls_but_the_workflow_advances(harness):
    sched = harness.configure(TWO_STEPS)
    ScriptedAgent.script = [DONE]
    sched.enqueue(ARXIV_ID)
    job = sched._executor.run_next()

    assert job.status == "stalled"
    assert job.error.startswith("git push failed")
    # Work is committed locally ...
    assert _git(harness.workspace, "log", "-1", "--format=%s").strip() == \
        "End of Iteration 1: core_claim [p (reproducer)]"
    tracked = _git(harness.workspace, "ls-files").splitlines()
    assert "manifest.json" in tracked
    # ... but the failed-push fallback dropped the trajectory from the commit.
    assert "trajectory/turn_0001_raw.jsonl" not in tracked
    # "stalled" does not stop a workflow.
    meta = _meta(harness)
    assert (meta["status"], meta["iteration"], meta["role"]) == ("active", 2, "verifier")
    assert len(sched._executor.pending) == 1


def test_failed_validation_wakes_the_agent_then_stalls_without_pushing(harness):
    remote = harness.add_remote()
    sched = harness.configure(TWO_STEPS)
    ScriptedAgent.script = [DONE]
    harness.validate_results.extend([1, 1, 1])
    sched.enqueue(ARXIV_ID)
    job = sched._executor.run_next()

    # Three validation attempts, the agent is woken between them.
    assert _headings(ScriptedAgent.prompts) == [
        "# Reproducer Agent", "# Validation Failed", "# Validation Failed",
    ]
    assert "validator output" in ScriptedAgent.prompts[1]
    assert harness.validate_results == []
    assert job.status == "stalled"
    assert job.error.startswith("validate.py failed")
    subjects = _git(remote, "log", "--format=%s", "main").splitlines()
    assert not any(s.startswith("End of Iteration") for s in subjects)
    # The workflow still moves to the next step.
    assert _meta(harness)["iteration"] == 2
    assert len(sched._executor.pending) == 1


def test_validation_passes_on_a_later_attempt(harness):
    harness.add_remote()
    sched = harness.configure(TWO_STEPS)
    ScriptedAgent.script = [DONE]
    harness.validate_results.extend([1, 0])
    sched.enqueue(ARXIV_ID)
    job = sched._executor.run_next()

    assert _headings(ScriptedAgent.prompts) == ["# Reproducer Agent", "# Validation Failed"]
    assert job.status == "done" and job.error is None


def test_gate_reverts_once_then_ends_unsuccessful(harness):
    harness.add_remote()
    sched = harness.configure(
        [{"profile": "p", "min_milestone": "full", "revert_to_iteration": 1},
         {"profile": "p", "role": "verifier"}],
        max_revert_attempts=1,
    )

    # First attempt ends below the gate: iteration 1 is queued again.
    ScriptedAgent.script = [DONE]
    sched.enqueue(ARXIV_ID)
    sched._executor.run_next()
    meta = _meta(harness)
    assert (meta["status"], meta["iteration"], meta["revert_attempts"]) == ("active", 1, 1)
    assert meta["is_iteration"] is True
    # A re-run reproducer starts one level below the previous claim.
    assert (meta["previous_milestone"], meta["milestone"]) == ("core_claim", "method_runs")

    # Second attempt also ends below the gate and no revert is left.
    ScriptedAgent.prompts.clear()
    ScriptedAgent.script = [DONE, DONE]
    job = sched._executor.run_next()
    assert _headings(ScriptedAgent.prompts) == [
        "# Reproducer Agent", "# You are Continuing a Reproduction",
    ]
    assert "core_claim" in ScriptedAgent.prompts[1] and "method_runs" in ScriptedAgent.prompts[1]
    assert job.status == "unsuccessful"
    assert _meta(harness)["status"] == "unsuccessful"
    assert sched.completed[ARXIV_ID].status == "unsuccessful"
    assert not sched._executor.pending and not sched.queue


def test_unknown_step_profile_fails_the_paper_with_a_clear_error(harness):
    sched = harness.configure([{"profile": "p"}])
    job = sched.enqueue(ARXIV_ID, profile="missing")
    sched._executor.run_next()

    assert job.status == "error"
    assert "Profile 'missing' not found" in job.error
    assert sched.completed[ARXIV_ID].status == "error"
