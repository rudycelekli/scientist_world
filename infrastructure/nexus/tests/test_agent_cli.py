"""Tests for nexus.agent_cli, the agent CLI ``nx``.

The helper tests exercise URL and project resolution.  The command
tests run every ``nx`` command against the real daemon app through its
test client (see the ``nx`` fixture), so a mismatch between what the CLI
expects and what the API returns fails here.
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import pytest

from nexus import agent_cli
from nexus.jobs import NexusJob


class TestAgentCliHelpers:
    """Test helper functions in agent_cli.py."""

    def test_format_duration_seconds(self):
        assert agent_cli._format_duration(45) == "45s"

    def test_format_duration_minutes(self):
        assert agent_cli._format_duration(125) == "2m05s"

    def test_format_duration_hours(self):
        assert agent_cli._format_duration(3661) == "1h01m"

    def test_format_duration_zero(self):
        assert agent_cli._format_duration(0) == "0s"

    def test_format_duration_unknown(self):
        assert agent_cli._format_duration(None) == "-"


class TestAgentCliEnvCheck:
    """Daemon URL and project resolution."""

    def test_get_url_missing(self, monkeypatch, tmp_path):
        """No NEXUS_URL, no NEXUS_DIR, no remote.json next to the package."""
        # Without NEXUS_DIR the state directory is looked up next to the
        # package; point that lookup at an empty directory.
        monkeypatch.setattr(agent_cli, "__file__", str(tmp_path / "pkg" / "agent_cli.py"))
        with pytest.raises(SystemExit):
            agent_cli._get_url()

    def test_get_url_present(self, monkeypatch):
        monkeypatch.setenv("NEXUS_URL", "http://localhost:39000")
        assert agent_cli._get_url() == "http://localhost:39000"

    def test_get_url_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("NEXUS_URL", "http://localhost:39000/")
        assert agent_cli._get_url() == "http://localhost:39000"

    def test_get_url_missing_no_remote(self, monkeypatch, tmp_path):
        """No NEXUS_URL and no remote.json in NEXUS_DIR: exit."""
        monkeypatch.setenv("NEXUS_DIR", str(tmp_path / "state"))
        with pytest.raises(SystemExit):
            agent_cli._get_url()

    def test_get_url_from_remote_json(self, monkeypatch, tmp_path):
        """Reads daemon_url from remote.json when NEXUS_URL is unset."""
        (tmp_path / "remote.json").write_text(
            json.dumps({"daemon_url": "http://remote-node:39000"})
        )
        monkeypatch.setenv("NEXUS_DIR", str(tmp_path))
        assert agent_cli._get_url() == "http://remote-node:39000"

    def test_get_url_env_overrides_remote_json(self, monkeypatch, tmp_path):
        """NEXUS_URL env var wins over remote.json."""
        (tmp_path / "remote.json").write_text(
            json.dumps({"daemon_url": "http://remote-node:39000"})
        )
        monkeypatch.setenv("NEXUS_DIR", str(tmp_path))
        monkeypatch.setenv("NEXUS_URL", "http://explicit:39000")
        assert agent_cli._get_url() == "http://explicit:39000"

    def test_resolve_project_from_flag(self):
        args = argparse.Namespace(project="my_proj")
        assert agent_cli._resolve_project(args) == "my_proj"

    def test_resolve_project_from_env(self, monkeypatch):
        monkeypatch.setenv("NEXUS_PROJECT_ID", "env_proj")
        args = argparse.Namespace(project=None)
        assert agent_cli._resolve_project(args) == "env_proj"

    def test_resolve_project_flag_over_env(self, monkeypatch):
        monkeypatch.setenv("NEXUS_PROJECT_ID", "env_proj")
        args = argparse.Namespace(project="flag_proj")
        assert agent_cli._resolve_project(args) == "flag_proj"

    def test_resolve_project_missing_both(self):
        args = argparse.Namespace(project=None)
        with pytest.raises(SystemExit):
            agent_cli._resolve_project(args)


class TestAgentCliIsStdlibOnly:
    """The agent CLI must run from a single copied file."""

    def test_no_third_party_imports(self):
        """agent_cli.py imports neither third-party packages nor nexus."""
        tree = ast.parse(Path(agent_cli.__file__).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, "relative import in agent_cli.py"
                imported.add(node.module.split(".")[0])
        assert imported <= {"argparse", "json", "os", "sys", "time", "urllib"}


@pytest.fixture
def nx(daemon, monkeypatch, tmp_path):
    """Point ``nx`` at the daemon app's test client, with project ``p1``.

    Returns the ``daemon`` fixture object, extended with ``workspace``.
    """

    def _call(method, path, data=None):
        resp = daemon.client.open(path, method=method, json=data)
        body = resp.get_json()
        if resp.status_code >= 400:
            print("Error: %s" % body.get("error"), file=sys.stderr)
            sys.exit(1)
        return body

    monkeypatch.setattr(
        agent_cli, "_get", lambda url, path, timeout=30: _call("GET", path))
    monkeypatch.setattr(
        agent_cli, "_post", lambda url, path, data=None: _call("POST", path, data or {}))
    monkeypatch.setenv("NEXUS_URL", "http://nexus.invalid:39000")
    monkeypatch.setenv("NEXUS_PROJECT_ID", "p1")
    monkeypatch.setenv("NEXUS_LOG_STATE", str(tmp_path / "nx_log_dirs.json"))

    daemon.workspace = tmp_path / "workspace"
    daemon.workspace.mkdir()
    resp = daemon.client.post("/api/projects", json={
        "project_id": "p1",
        "max_gpus": 2,
        "workspace_dir": str(daemon.workspace),
        "sif_path": "/images/p.sif",
        "allowed_resources": ["gpu-a"],
    })
    assert resp.status_code == 201
    return daemon


def submit(capsys, *argv):
    """Run ``nx submit`` and return the run_id it printed."""
    assert agent_cli.main(["submit", "echo hi", *argv]) == 0
    out = capsys.readouterr().out
    lines = dict(line.split(": ", 1) for line in out.splitlines() if ": " in line)
    return lines["run_id"], out


class TestAgentCliCommands:
    def test_submit(self, nx, capsys):
        run_id, out = submit(capsys, "--gpus", "1", "--cpus", "2", "--timeout", "90")

        assert out.startswith("JOB SUBMITTED\n")
        assert "command: echo hi" in out
        assert "resource: gpu-a" in out
        assert "gpus: 1" in out
        assert "timeout: 90s" in out
        assert "project_gpus_in_use: 1/2" in out
        assert "gpu_budget: unlimited" in out

        job = NexusJob.load(run_id)
        assert job.command == "echo hi"
        assert job.num_cpus == 2
        assert job.timeout_sec == 90

    def test_submit_to_a_named_resource(self, nx, capsys):
        with pytest.raises(SystemExit) as exc:
            agent_cli.main(["submit", "echo hi", "--resource", "gpu-z"])
        assert exc.value.code == 1
        assert "not in allowed list" in capsys.readouterr().err

        run_id, out = submit(capsys, "-r", "gpu-a")
        assert "resource: gpu-a" in out

    def test_submit_over_quota_exits_1(self, nx, capsys):
        submit(capsys, "--gpus", "2")
        with pytest.raises(SystemExit) as exc:
            agent_cli.main(["submit", "echo hi"])
        assert exc.value.code == 1
        assert "2/2 in use" in capsys.readouterr().err

    def test_status_of_one_job_and_of_the_project(self, nx, capsys):
        run_id, _ = submit(capsys)
        nx.monitor._tick()  # queued -> running

        assert agent_cli.main(["status", run_id]) == 0
        out = capsys.readouterr().out
        assert out.startswith(f"JOB {run_id}\n")
        assert "status: running" in out
        assert "command: echo hi" in out

        assert agent_cli.main(["status"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("JOBS FOR PROJECT p1\n")
        assert run_id in out and "running" in out

    def test_logs_while_active_and_after_the_end(self, nx, capsys):
        run_id, _ = submit(capsys)
        nx.bridge.poll.return_value = {
            "status": "running", "stdout": "epoch 1\n", "stderr": "warn\n"}
        nx.monitor._tick()

        assert agent_cli.main(["logs", run_id]) == 0
        out = capsys.readouterr().out
        assert "-- stdout --\nepoch 1\n" in out
        assert "-- stderr --\nwarn\n" in out
        assert "(job is still running)" in out

        nx.bridge.poll.return_value = {
            "status": "completed", "stdout": "epoch 1\ndone\n", "stderr": "",
            "returncode": 0}
        nx.monitor._tick()

        assert agent_cli.main(["logs", run_id]) == 0
        out = capsys.readouterr().out
        assert f"JOB {run_id} (completed, exit 0)" in out
        assert "-- stdout --" not in out
        assert f"jobs/{run_id}/" in out
        # ... where the complete output is.
        assert (nx.workspace / "jobs" / run_id / "stdout.log").read_text() == "epoch 1\ndone\n"

    def test_wait_reports_a_finished_job(self, nx, capsys):
        run_id, _ = submit(capsys)
        nx.bridge.poll.return_value = {
            "status": "failed", "stdout": "", "stderr": "", "returncode": 1}
        nx.monitor._tick()

        assert agent_cli.main(["wait", "--run-id", run_id, "--timeout", "0"]) == 0
        out = capsys.readouterr().out
        assert out == f"EVENT: failed\nRUN_ID: {run_id}\nSTATUS: failed\n"

    def test_wait_timeout(self, nx, capsys):
        """The daemon answers {"timeout": true}; nx must recognise it."""
        submit(capsys)
        assert agent_cli.main(["wait", "--timeout", "0"]) == 0
        out = capsys.readouterr().out
        assert out == "TIMEOUT: no event within 0s\n"

    def test_wait_at_a_checkpoint_says_how_to_answer(self, nx, capsys):
        run_id, _ = submit(capsys)
        nx.monitor._tick()  # running
        nx.monitor.tracked_jobs()[run_id].job.started_at -= 100000
        nx.monitor._tick()  # checkpoint

        assert agent_cli.main(["wait", "--run-id", run_id, "--timeout", "0"]) == 0
        out = capsys.readouterr().out
        assert "EVENT: awaiting_response\n" in out
        assert "STATUS: awaiting_response\n" in out
        assert f"Respond with: nx continue {run_id}  OR  nx cancel {run_id}" in out

        assert agent_cli.main(["continue", run_id]) == 0
        assert NexusJob.load(run_id).status == "running"

    def test_cancel(self, nx, capsys):
        run_id, _ = submit(capsys)
        assert agent_cli.main(["cancel", run_id]) == 0
        assert NexusJob.load(run_id).status == "cancelled"
        nx.bridge.cancel.assert_called_once_with("hive-1")

    def test_info(self, nx, capsys):
        submit(capsys)
        assert agent_cli.main(["info"]) == 0
        out = capsys.readouterr().out
        assert out == (
            "PROJECT p1\n"
            "  status: active\n"
            "  resources: gpu-a\n"
            "  gpus: 1/2 in use\n"
            "  gpu_budget: unlimited\n"
            "  job_timeout: 3600s\n"
            "  jobs: 0 running, 0 completed, 0 failed\n"
        )

    def test_unknown_project(self, nx, capsys):
        with pytest.raises(SystemExit) as exc:
            agent_cli.main(["info", "--project", "nope"])
        assert exc.value.code == 1
        assert "not found" in capsys.readouterr().err

    def test_log_dir_mirror(self, nx, capsys, tmp_path):
        log_dir = tmp_path / "logs"
        run_id, out = submit(capsys, "--log-dir", str(log_dir))
        assert f"log_dir: {log_dir}/{run_id}/" in out

        nx.bridge.poll.return_value = {
            "status": "running", "stdout": "epoch 1\n", "stderr": ""}
        nx.monitor._tick()
        assert agent_cli.main(["status", run_id]) == 0

        assert (log_dir / run_id / "stdout.log").read_text() == "epoch 1\n"
        assert (log_dir / run_id / "stderr.log").read_text() == ""

    def test_log_dir_mirror_survives_the_end_of_the_job(self, nx, capsys):
        """nx must not empty the logs once the daemon stops serving them.

        With ``--log-dir jobs`` the mirror is the very directory in which
        the daemon keeps the job's final output.
        """
        log_dir = nx.workspace / "jobs"
        run_id, _ = submit(capsys, "--log-dir", str(log_dir))
        nx.bridge.poll.return_value = {
            "status": "completed", "stdout": "all done\n", "stderr": "",
            "returncode": 0}
        nx.monitor._tick()
        final_log = log_dir / run_id / "stdout.log"
        assert final_log.read_text() == "all done\n"

        for command in (["logs", run_id], ["status", run_id],
                        ["wait", "--run-id", run_id, "--timeout", "0"]):
            assert agent_cli.main(command) == 0
            assert final_log.read_text() == "all done\n"


class TestAgentCliParser:
    def test_no_command_prints_help(self, capsys):
        assert agent_cli.main([]) == 1
        assert "usage: nx" in capsys.readouterr().out

    @pytest.mark.parametrize("command", [
        None, "submit", "status", "logs", "continue", "cancel", "wait", "info",
    ])
    def test_help_renders(self, command, capsys):
        argv = ["--help"] if command is None else [command, "--help"]
        with pytest.raises(SystemExit) as exc:
            agent_cli.main(argv)
        assert exc.value.code == 0
        assert "usage: nx" in capsys.readouterr().out

    def test_help_uses_the_job_states_of_the_daemon(self, capsys):
        """The status and event names in the help are the ones the API returns."""
        with pytest.raises(SystemExit):
            agent_cli.main(["wait", "--help"])
        wait_help = capsys.readouterr().out
        with pytest.raises(SystemExit):
            agent_cli.main(["status", "--help"])
        status_help = capsys.readouterr().out
        for state in ("completed", "failed", "awaiting_response", "cancelled"):
            assert state in wait_help
            assert state in status_help
