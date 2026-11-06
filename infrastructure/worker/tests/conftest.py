"""Shared fixtures for worker tests."""

import errno
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from infrastructure.worker.claude_code import ClaudeCodeWorker

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "workspace_template"


@pytest.fixture(autouse=True)
def only_git_subprocesses(monkeypatch):
    """Refuse to start any program other than git, and fail the test.

    A worker turn starts an agent CLI, usually inside a container. A test
    that reaches that call without replacing it would run a real agent on
    the machine that runs the tests, so the call fails as if the program
    were not installed and the test is reported as an error.
    """
    refused = []
    real_popen = subprocess.Popen

    class GuardedPopen(real_popen):
        def __init__(self, args, *popen_args, **kwargs):
            argv = [args] if isinstance(args, (str, bytes, os.PathLike)) else list(args)
            program = os.path.basename(os.fsdecode(argv[0])) if argv else ""
            if program != "git":
                refused.append(" ".join(os.fsdecode(arg) for arg in argv))
                raise FileNotFoundError(errno.ENOENT, "refused by the test suite", program)
            super().__init__(args, *popen_args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", GuardedPopen)
    yield
    assert not refused, f"the test tried to start: {refused}"


@pytest.fixture(autouse=True)
def isolated_host_env(tmp_path, monkeypatch):
    """Keep tests away from the developer's credentials and git identity.

    The workers copy ``~/.claude/.credentials.json`` and ``~/.codex/auth.json``
    into the workspace and inherit ``GIT_*`` identity variables.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for var in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL",
                "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
        monkeypatch.delenv(var, raising=False)
    return home


@pytest.fixture
def tmp_workspace(tmp_path):
    """Temporary workspace seeded with workspace_template files."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    # Copy template files into workspace
    for src in TEMPLATE_DIR.iterdir():
        dst = ws / src.name
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    return ws


@pytest.fixture
def worker():
    """A ClaudeCodeWorker with default config."""
    return ClaudeCodeWorker("claude-code", "claude-opus-4-6", {})


@pytest.fixture
def configured_worker(tmp_workspace):
    """A worker with paper, workspace, and resources already set."""
    w = ClaudeCodeWorker("claude-code", "claude-opus-4-6", {})
    w.set_paper_and_workspace(
        {"paper_md_path": "/tmp/paper.md", "paper_images_dir": "/tmp/images", "paper_pdf_path": "/tmp/paper.pdf"},
        str(tmp_workspace),
    )
    w.set_resources(
        gpu_budget=20.0,
        max_gpus=2,
        job_time_limit="12:00:00",
        shared_datasets_path="/data/datasets",
        shared_models_path="/data/models",
    )
    return w
