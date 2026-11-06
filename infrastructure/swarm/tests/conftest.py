"""Shared test fixtures for Swarm tests.

Follows the same pattern as hive/tests/conftest.py and
nexus/tests/conftest.py: patch module-level path constants
to a temporary directory so no real state files are touched.
"""

import errno
import os
import subprocess
import urllib.error
import urllib.request

import pytest
from unittest.mock import patch

import swarm.scheduler  # noqa: F401  (puts the repository root on sys.path)
from swarm.config import SwarmConfig, WorkerConfig, NexusConnectionConfig


# Programs a test may start. Everything else Swarm shells out to must be
# replaced by the test that reaches it.
_ALLOWED_PROGRAMS = ("git", "cp")


@pytest.fixture(autouse=True)
def no_external_effects(monkeypatch):
    """Refuse HTTP requests and programs other than git and cp; fail the test.

    Swarm shells out to the container runtime, to ``pgrep`` and ``lsof``
    before it signals processes, and to a Python child for the Nexus
    daemon, and it talks to Nexus and GitHub over HTTP. A test that
    reaches one of those calls without replacing it would pull images,
    signal processes or change a running Nexus on the machine that runs
    the tests. The call fails as if the program were not installed or the
    server were down, and the test is reported as an error.
    """
    refused = []
    real_popen = subprocess.Popen

    def refuse_request(request, *args, **kwargs):
        url = getattr(request, "full_url", request)
        refused.append(f"HTTP request to {url}")
        raise urllib.error.URLError("refused by the test suite")

    class GuardedPopen(real_popen):
        def __init__(self, args, *popen_args, **kwargs):
            argv = [args] if isinstance(args, (str, bytes, os.PathLike)) else list(args)
            program = os.path.basename(os.fsdecode(argv[0])) if argv else ""
            if program not in _ALLOWED_PROGRAMS:
                refused.append(" ".join(os.fsdecode(arg) for arg in argv))
                raise FileNotFoundError(errno.ENOENT, "refused by the test suite", program)
            super().__init__(args, *popen_args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", GuardedPopen)
    monkeypatch.setattr(urllib.request, "urlopen", refuse_request)
    yield
    assert not refused, f"the test reached outside the test process: {refused}"


@pytest.fixture
def tmp_swarm_dir(tmp_path):
    """Patch Swarm path constants to use a temporary directory."""
    swarm_dir = tmp_path / ".swarm"
    papers_dir = swarm_dir / "papers"

    swarm_dir.mkdir()
    papers_dir.mkdir()

    with (
        patch("swarm.config.SWARM_DIR", swarm_dir),
        patch("swarm.config.PAPERS_DIR", papers_dir),
        patch("swarm.config.DAEMON_PID_PATH", swarm_dir / "daemon.pid"),
        patch("swarm.config.USER_CONFIG_PATH", swarm_dir / "config.yaml"),
        patch("swarm.config.PACKAGE_CONFIG_PATH", swarm_dir / "package_config.yaml"),
        patch("swarm.scheduler.PAPERS_DIR", papers_dir),
    ):
        yield swarm_dir


@pytest.fixture
def sample_config(tmp_path):
    """Return a minimal SwarmConfig for testing."""
    return SwarmConfig(
        pool_size=2,
        scout_data_dir=str(tmp_path / "data"),
        workspace_base_dir=str(tmp_path / "workspaces"),
        shared_datasets_path=str(tmp_path / "datasets"),
        shared_models_path=str(tmp_path / "models"),
        dashboard_port=39199,
        worker=WorkerConfig(
            scaffold="claude-code",
            model="claude-sonnet-4-5-20250929",
            max_turns=5,
        ),
        nexus=NexusConnectionConfig(
            url="http://localhost:39000",
            gpu_budget_hours=5.0,
            max_gpus=2,
        ),
    )
