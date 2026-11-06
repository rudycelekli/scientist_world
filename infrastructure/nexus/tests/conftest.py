"""Shared fixtures for nexus tests.

The two autouse fixtures make the whole suite hermetic: no test reads the
Nexus environment variables of the calling shell, and no test reads or
writes the real state directory or a local config.yaml.
"""

from unittest.mock import MagicMock, patch

import pytest

# Hive address used in tests. The reserved ".invalid" domain never resolves,
# so a call that escapes a mock cannot reach a real service.
FAKE_HIVE_URL = "http://hive.invalid:38581"

# Every environment variable that Nexus reads.
_NEXUS_ENV_VARS = (
    "NEXUS_URL", "NEXUS_PROJECT_ID", "NEXUS_DIR", "NEXUS_LOG_STATE", "HIVE_URL",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Remove the Nexus environment variables for the duration of a test."""
    for name in _NEXUS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def tmp_nexus_dir(tmp_path):
    """Redirect all Nexus state and config paths to a temporary directory.

    Tests that need the directory request this fixture by name.
    """
    nexus_dir = tmp_path / ".nexus"
    nexus_dir.mkdir()
    (nexus_dir / "projects").mkdir()
    (nexus_dir / "jobs").mkdir()
    with patch("nexus.config.NEXUS_DIR", nexus_dir), \
         patch("nexus.config.PROJECTS_DIR", nexus_dir / "projects"), \
         patch("nexus.config.JOBS_DIR", nexus_dir / "jobs"), \
         patch("nexus.config.DAEMON_PID_PATH", nexus_dir / "daemon.pid"), \
         patch("nexus.config.REMOTE_CONFIG_PATH", nexus_dir / "remote.json"), \
         patch("nexus.config.USER_CONFIG_PATH", nexus_dir / "config.yaml"), \
         patch("nexus.config.PACKAGE_CONFIG_PATH", nexus_dir / "package_config.yaml"), \
         patch("nexus.project.PROJECTS_DIR", nexus_dir / "projects"), \
         patch("nexus.jobs.JOBS_DIR", nexus_dir / "jobs"), \
         patch("nexus.cli.NEXUS_DIR", nexus_dir), \
         patch("nexus.cli.DAEMON_PID_PATH", nexus_dir / "daemon.pid"), \
         patch("nexus.cli.REMOTE_CONFIG_PATH", nexus_dir / "remote.json"):
        yield nexus_dir


@pytest.fixture
def sample_config():
    """NexusConfig with a fake Hive URL and a monitor that never ticks twice."""
    from nexus.config import (
        NexusConfig, HiveConnectionConfig, DaemonConfig,
        MonitorConfig, ProjectDefaults,
    )
    return NexusConfig(
        hive=HiveConnectionConfig(url=FAKE_HIVE_URL),
        daemon=DaemonConfig(port=39000, host="127.0.0.1"),
        monitor=MonitorConfig(
            status_poll_interval=3600,
            checkpoint_interval=1200,
            checkpoint_timeout=1800,
        ),
        project_defaults=ProjectDefaults(
            max_gpus=4,
            default_job_timeout=3600,
            soft_gpu_budget_hours=0,
        ),
    )


@pytest.fixture
def daemon(sample_config, tmp_nexus_dir):
    """A daemon app wired to a mocked Hive bridge.

    Yields an object with ``app``, ``client`` (Flask test client),
    ``bridge`` (the mock that stands in for HiveBridge) and ``monitor``
    (the real JobMonitor created by the app).  The monitor thread is
    started by ``create_app``; with a one-hour poll interval it runs one
    tick on an empty job set and then sleeps, so tests drive the monitor
    themselves with ``monitor._tick()``.
    """
    import threading

    import nexus.daemon as daemon_module
    from nexus.hive_bridge import HiveBridge

    bridge = MagicMock(spec=HiveBridge)
    bridge.submit_for_project.return_value = "hive-1"
    bridge.poll.return_value = {"status": "running", "stdout": "", "stderr": ""}
    bridge.cancel.return_value = True
    bridge.cluster_status.return_value = {
        "total_gpus": 8, "free_gpus": 6, "available_gpu": 6,
        "num_gpu_nodes": 2, "num_cpu_nodes": 1,
    }

    monitors = []

    class _RecordingMonitor(daemon_module.JobMonitor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ticked = threading.Event()
            monitors.append(self)

        def _tick(self):
            super()._tick()
            self.ticked.set()

    with patch.object(daemon_module, "HiveBridge", return_value=bridge), \
         patch.object(daemon_module, "JobMonitor", _RecordingMonitor):
        app = daemon_module.create_app(sample_config)

    class _Daemon:
        pass

    d = _Daemon()
    d.app = app
    d.client = app.test_client()
    d.bridge = bridge
    d.monitor = monitors[0]
    try:
        # Let the thread finish its first tick before a test adds jobs.
        assert d.monitor.ticked.wait(timeout=10)
        yield d
    finally:
        d.monitor.stop()
