"""Workspace plumbing: .gitignore installation, the per-paper cache, and
the end-of-run helpers of the runner."""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from evaluation import cache, container, git, git_publish, runner

PAPER = "2501.00001"

_CREDENTIAL_ENTRIES = (
    ".claude.json", ".claude.agent-original/", ".claude.json.agent-original",
    ".kimi/", ".gemini/",
)


def _workspace(data_dir: Path, name: str = "my-profile-improve-eval") -> Path:
    env_dir = data_dir / "envs" / PAPER / name
    env_dir.mkdir(parents=True)
    return env_dir


# ── .gitignore ───────────────────────────────────────────────────────────


def test_install_gitignore_patches_the_template(eval_config, tmp_path: Path) -> None:
    env_dir = _workspace(eval_config())
    template = tmp_path / "template.gitignore"
    template.write_text("# header\ntrajectory/\nlogs/\n*.sif\n.claude/\n")

    git.install_gitignore(env_dir, template)

    lines = (env_dir / ".gitignore").read_text().splitlines()
    # The harness commits trajectories and job logs with the run.
    assert "trajectory/" not in lines and "logs/" not in lines
    assert "*.sif" in lines and ".claude/" in lines and "# header" in lines
    # Credentials the harness copies into the workspace are never committed.
    for entry in _CREDENTIAL_ENTRIES:
        assert entry in lines, entry


def test_install_gitignore_with_the_shipped_template(eval_config) -> None:
    template = runner._gitignore_template({})
    if not template.exists():
        pytest.skip(f"worker workspace template not in this checkout: {template}")
    env_dir = _workspace(eval_config())
    git.install_gitignore(env_dir, template)
    lines = (env_dir / ".gitignore").read_text().splitlines()
    assert "trajectory/" not in lines and "logs/" not in lines
    for entry in _CREDENTIAL_ENTRIES + (".claude/", ".codex/", "environment/container.sif"):
        assert entry in lines, entry


def test_install_gitignore_missing_template_names_the_config_key(
    eval_config, tmp_path: Path,
) -> None:
    env_dir = _workspace(eval_config())
    with pytest.raises(FileNotFoundError, match="workspace_gitignore"):
        git.install_gitignore(env_dir, tmp_path / "absent")


def test_ensure_credentials_ignored_upgrades_an_older_gitignore(eval_config) -> None:
    env_dir = _workspace(eval_config())
    (env_dir / ".gitignore").write_text("*.sif\n.claude/\n.kimi/\n")

    assert git.ensure_credentials_ignored(env_dir) is True
    lines = (env_dir / ".gitignore").read_text().splitlines()
    assert lines[:3] == ["*.sif", ".claude/", ".kimi/"]
    for entry in _CREDENTIAL_ENTRIES:
        assert lines.count(entry) == 1, entry
    # nothing to add the second time
    before = (env_dir / ".gitignore").read_text()
    assert git.ensure_credentials_ignored(env_dir) is False
    assert (env_dir / ".gitignore").read_text() == before


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_parked_agent_credentials_are_not_committed(eval_config) -> None:
    """What the verifier does before a Claude grader starts: the agent's
    session directory is renamed, and its login file must stay untracked."""
    env_dir = _workspace(eval_config())
    (env_dir / ".gitignore").write_text(".claude/\n")  # an older workspace
    (env_dir / ".claude").mkdir()
    (env_dir / ".claude" / ".credentials.json").write_text('{"token": "x"}')
    (env_dir / ".claude.json").write_text("{}")
    (env_dir / "method").mkdir()
    (env_dir / "method" / "solution.py").write_text("# code\n")
    git.init_repo(env_dir)
    git.commit(env_dir, "eval: initial scaffold")

    git.ensure_credentials_ignored(env_dir)
    (env_dir / ".claude").rename(env_dir / ".claude.agent-original")
    (env_dir / ".claude.json").rename(env_dir / ".claude.json.agent-original")
    git.commit(env_dir, "verify: start")

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=str(env_dir), capture_output=True, text=True,
        check=True).stdout.split()
    assert "method/solution.py" in tracked and ".gitignore" in tracked
    assert not any("credentials" in name or "agent-original" in name for name in tracked)
    assert ".claude.json" not in tracked


def test_gitignore_template_override(tmp_path: Path) -> None:
    assert runner._gitignore_template({}) == runner._GITIGNORE_TEMPLATE
    assert runner._gitignore_template({"workspace_gitignore": ""}) == runner._GITIGNORE_TEMPLATE
    custom = tmp_path / "custom.gitignore"
    assert runner._gitignore_template({"workspace_gitignore": str(custom)}) == custom


@pytest.mark.parametrize("relative", [
    "envs",                              # the envs root itself
    f"envs/{PAPER}",                     # a paper directory
    f"envs/{PAPER}/source",              # the paper's source clone
    f"envs/{PAPER}/run-eval/method",     # inside a workspace
    "elsewhere/x/y",                     # outside envs/
])
def test_git_refuses_anything_but_a_run_workspace(eval_config, relative: str) -> None:
    data_dir = eval_config()
    target = data_dir / relative
    target.mkdir(parents=True)
    with pytest.raises(git.GitSafetyError):
        git._assert_per_run_workspace(target)


def test_git_never_pushes(tmp_path: Path) -> None:
    with pytest.raises(git.GitSafetyError):
        git._run(["push", "origin", "main"], tmp_path)
    with pytest.raises(git.GitSafetyError):
        git._run(["remote", "add", "origin", "https://example.com/x.git"], tmp_path)


# ── Per-paper cache ──────────────────────────────────────────────────────


def test_cache_round_trip_uses_the_sparse_copy(eval_config, monkeypatch) -> None:
    """Container image and overlay go through container._copy_artifact in
    both directions (a plain copy would write out an overlay's holes)."""
    data_dir = eval_config()
    warmed = _workspace(data_dir, "warm-staging")
    (warmed / "environment").mkdir()
    (warmed / "environment" / "container.sif").write_bytes(b"SIF" * 100)
    overlay = warmed / "environment" / "overlay.img"
    with overlay.open("wb") as fh:  # sparse: 8 MiB apparent, a few bytes written
        fh.seek(8 * 1024 * 1024 - 4)
        fh.write(b"tail")
    (warmed / "data" / "downloads" / "sub").mkdir(parents=True)
    (warmed / "data" / "downloads" / "sub" / "weights.bin").write_bytes(b"W" * 10)

    copied: list[str] = []
    real_copy = container._copy_artifact

    def spy(src: Path, dst: Path) -> None:
        copied.append(src.name)
        real_copy(src, dst)

    monkeypatch.setattr(cache, "_copy_artifact", spy)

    assert cache.is_warm(PAPER) is False
    manifest = cache.populate_from_workspace(PAPER, warmed)
    assert manifest["has_container"] and manifest["has_overlay"]
    assert manifest["files"] == [
        {"rel_path": "data/downloads/sub/weights.bin", "size": 10}]
    assert cache.is_warm(PAPER) is True
    assert copied == ["container.sif", "overlay.img"]

    run = _workspace(data_dir)
    binds = cache.apply_to_workspace(PAPER, run)
    assert copied == ["container.sif", "overlay.img"] * 2
    assert (run / "environment" / "container.sif").read_bytes() == b"SIF" * 100
    assert (run / "environment" / "overlay.img").read_bytes() == overlay.read_bytes()
    # Downloads are bound read-only over an empty placeholder, not copied.
    cached_file = cache.cache_dir(PAPER) / "data" / "downloads" / "sub" / "weights.bin"
    assert binds == [(cached_file, "/home/user/data/downloads/sub/weights.bin")]
    assert (run / "data" / "downloads" / "sub" / "weights.bin").read_bytes() == b""


def test_cache_apply_without_a_cache_is_a_no_op(eval_config) -> None:
    run = _workspace(eval_config())
    assert cache.apply_to_workspace(PAPER, run) == []
    assert list(run.iterdir()) == []


# ── End-of-run helpers ───────────────────────────────────────────────────


def test_remove_container_artifacts(tmp_path: Path) -> None:
    env = tmp_path / "environment"
    env.mkdir()
    (env / "container.sif").write_bytes(b"x")
    (env / "overlay.img").write_bytes(b"x")
    (env / "container.def").write_text("Bootstrap: docker\n")
    runner._remove_container_artifacts(tmp_path)
    assert sorted(p.name for p in env.iterdir()) == ["container.def"]
    runner._remove_container_artifacts(tmp_path)  # nothing left: still fine


def test_publish_run_disabled_does_not_call_publish(tmp_path: Path, monkeypatch) -> None:
    def boom(*a, **k):
        raise AssertionError("publish() must not be called when publishing is off")

    monkeypatch.setattr(git_publish, "publish", boom)
    record = runner._publish_run(tmp_path, PAPER, "p", "improve", {"publish": {}})
    assert record["status"] == "disabled"
    assert set(record) == {"at", "status"}


def test_publish_run_records_the_outcome(tmp_path: Path, monkeypatch) -> None:
    cfg = {"publish": {"enabled": True, "github_org": "example-org"}}

    def pushed(env_dir, paper_id, profile, mode):
        return git_publish.PublishResult(
            paper_id, profile, mode, f"{mode}-{profile}", "example-org/repo",
            pushed=True, files_committed=3, bytes_committed=42)

    monkeypatch.setattr(git_publish, "publish", pushed)
    record = runner._publish_run(tmp_path, PAPER, "p", "improve", cfg)
    assert (record["status"], record["branch"], record["repo"]) == (
        "pushed", "improve-p", "example-org/repo")
    assert (record["files"], record["bytes"]) == (3, 42)

    def skipped(env_dir, paper_id, profile, mode):
        return git_publish.PublishResult(
            paper_id, profile, mode, f"{mode}-{profile}", "",
            pushed=False, skipped_reason="no GitHub token")

    monkeypatch.setattr(git_publish, "publish", skipped)
    record = runner._publish_run(tmp_path, PAPER, "p", "improve", cfg)
    assert (record["status"], record["reason"]) == ("skipped", "no GitHub token")


def test_publish_run_never_raises(tmp_path: Path, monkeypatch) -> None:
    def failing(*a, **k):
        raise RuntimeError("git push failed (rc=1)")

    monkeypatch.setattr(git_publish, "publish", failing)
    record = runner._publish_run(
        tmp_path, PAPER, "p", "improve", {"publish": {"enabled": True}})
    assert record["status"] == "failed"
    assert "git push failed" in record["error"]


# ── warm ─────────────────────────────────────────────────────────────────


def test_warm_stages_each_paper_in_its_own_directory(eval_config, monkeypatch) -> None:
    """sciworld keeps a paper's environment in the directory ``source`` next
    to the workspace it builds. With one staging parent for all papers, the
    second paper warmed was built from the first paper's environment."""
    import sys
    import types

    data_dir = eval_config()
    calls: list[dict] = []

    def fake_research(**kwargs):
        calls.append(kwargs)
        workspace = Path(kwargs["path"])
        (workspace.parent / "source").mkdir(parents=True, exist_ok=True)
        (workspace.parent / "source" / "PAPER").write_text(kwargs["id"])
        workspace.mkdir(parents=True)
        return {"env_dir": workspace}

    monkeypatch.setitem(sys.modules, "sciworld", types.SimpleNamespace(research=fake_research))
    monkeypatch.setattr(runner._container, "ensure_container", lambda *a, **k: Path("container.sif"))
    monkeypatch.setattr(runner._container, "ensure_overlay", lambda *a, **k: Path("overlay.img"))
    populated: list[tuple] = []
    monkeypatch.setattr(cache, "populate_from_workspace", lambda paper, ws: populated.append((paper, ws)))

    for paper in ("2501.00001", "2502.00002"):
        runner.warm(paper, force=True)

    staging = data_dir / "cache-staging"
    assert [c["path"] for c in calls] == [
        str(staging / "2501.00001" / "workspace"), str(staging / "2502.00002" / "workspace")]
    assert all(c["setup_only"] and c["env"] == "container" for c in calls)
    for paper in ("2501.00001", "2502.00002"):
        assert (staging / paper / "source" / "PAPER").read_text() == paper
    assert populated == [("2501.00001", staging / "2501.00001" / "workspace"),
                         ("2502.00002", staging / "2502.00002" / "workspace")]


def test_overlay_cache_eviction_covers_package_archives(tmp_path, monkeypatch):
    """Hosts without user namespaces cache package archives instead of
    overlay images. Those were never counted or evicted, so on such hosts
    the cache grew without bound."""
    cache_dir = tmp_path / "overlay-cache"
    cache_dir.mkdir()
    long_ago = time.time() - 10 * 86400
    for name in ("a.img", "b.pkgs.tar"):
        entry = cache_dir / name
        entry.write_bytes(b"x" * 4096)
        os.utime(entry, (long_ago, long_ago))
    (cache_dir / "a.json").write_text("{}")
    (cache_dir / "c.pkgs.tar").write_bytes(b"x" * 4096)   # too young to evict
    monkeypatch.setattr(container, "OVERLAY_CACHE_BUDGET_GB", 0)

    container._evict_overlay_cache(cache_dir)

    assert sorted(p.name for p in cache_dir.iterdir()) == ["c.pkgs.tar"]
