"""Tests for :func:`evaluation.runner._scaffold`.

The fast tests replace ``sciworld.research`` and the container builders
with stubs. The one test marked ``slow`` calls the real
``sciworld.research()``, which clones an environment repository from
GitHub; it is deselected by default. To run it::

    cd evaluation
    EVAL_TEST_PAPER_ID=<paper id of a published environment> \\
        python -m pytest tests/test_scaffold.py -m slow -v
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import types
from pathlib import Path

import pytest

from evaluation import config as _config
from evaluation import container as _container
from evaluation import runner

PAPER_ID = "2501.00001"
PROFILE = "my-profile"


@pytest.fixture
def stub_builders(monkeypatch):
    """Replace the Apptainer-dependent container and overlay builders."""
    def fake_ensure_container(env_dir, **_kwargs):
        path = env_dir / "environment" / "container.sif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    monkeypatch.setattr(_container, "ensure_container", fake_ensure_container)
    monkeypatch.setattr(_container, "ensure_overlay",
                        lambda env_dir, container_sif, **_kwargs: None)


@pytest.fixture
def fake_sciworld(monkeypatch, stub_builders):
    """A stand-in ``sciworld`` module whose ``research`` lays out a minimal
    workspace the way the real one does, and records its arguments."""
    calls: list[dict] = []

    def fake_research(**kwargs):
        calls.append(kwargs)
        env_dir = Path(kwargs["path"])
        (env_dir / "READMEs").mkdir(parents=True, exist_ok=True)
        (env_dir / "idea.md").write_text("stock idea file\n")
        # Things an evaluated agent must not see.
        (env_dir / "scoring").mkdir()
        (env_dir / "scoring" / "reference.json").write_text("{}")
        (env_dir / "eval" / "test").mkdir(parents=True)
        (env_dir / "method").mkdir()
        (env_dir / "method" / "paper_method.py").write_text("# the paper's method\n")
        return {"env_dir": env_dir}

    module = types.ModuleType("sciworld")
    module.research = fake_research
    monkeypatch.setitem(sys.modules, "sciworld", module)
    return calls


def _env_dir(mode: str, variant: str = "") -> Path:
    return _config.envs_dir() / PAPER_ID / f"{PROFILE}-{mode}{variant}-eval"


def test_scaffold_wipes_existing(eval_config, fake_sciworld) -> None:
    """An existing workspace without substantial work is removed first."""
    eval_config()
    env_dir = _env_dir("innovate")
    env_dir.mkdir(parents=True)
    (env_dir / "stale.txt").write_text("leftover from a previous attempt")

    result_env, binds, sif, overlay = runner._scaffold(
        PAPER_ID, PROFILE, "claude-code", "innovate")

    assert result_env == env_dir
    assert not (env_dir / "stale.txt").exists(), "stale contents survived the wipe"
    assert binds == [] and overlay is None
    assert sif == env_dir / "environment" / "container.sif"
    assert len(fake_sciworld) == 1
    call = fake_sciworld[0]
    assert call["id"] == PAPER_ID
    assert call["path"] == str(env_dir)
    assert call["setup_only"] is True


def test_scaffold_enforces_the_workspace_whitelist(eval_config, fake_sciworld) -> None:
    eval_config()
    env_dir, *_ = runner._scaffold(PAPER_ID, PROFILE, "claude-code", "innovate")

    assert not (env_dir / "idea.md").exists(), "stock idea.md should be removed"
    assert (env_dir / "idea").is_dir(), "idea/ should be a folder"
    assert list((env_dir / "scoring").iterdir()) == [], "scoring/ must start empty"
    assert not (env_dir / "eval" / "test").exists(), "held-out evaluator leaked"
    assert list((env_dir / "method").iterdir()) == [], "innovate starts without a method"
    assert (env_dir / "scripts" / "run.sh").is_file()


def test_improve_keeps_the_starting_method(eval_config, fake_sciworld) -> None:
    eval_config()
    env_dir, *_ = runner._scaffold(PAPER_ID, PROFILE, "claude-code", "improve")
    assert (env_dir / "method" / "paper_method.py").is_file()
    assert fake_sciworld[0]["mode"] == "improve"


@pytest.mark.parametrize("scaffold, expected_agent", [
    ("claude-code", "claude-code"),
    ("codex", "codex"),
    ("claude-tmux", "claude-code"),  # same workspace layout, different driver
])
def test_scaffold_arguments_passed_to_sciworld(
    eval_config, fake_sciworld, scaffold, expected_agent,
) -> None:
    eval_config()
    runner._scaffold(PAPER_ID, PROFILE, scaffold, "innovate")
    call = fake_sciworld[0]
    assert call["agent"] == expected_agent
    assert call["mode"] == "innovate"
    assert call["env"] == "container"


def test_scaffold_refuses_to_wipe_work(eval_config, fake_sciworld) -> None:
    """Scoring files on disk mean an earlier run did real work."""
    eval_config()
    env_dir = _env_dir("innovate")
    (env_dir / "scoring").mkdir(parents=True)
    (env_dir / "scoring" / "scores_train.json").write_text("{}")

    with pytest.raises(RuntimeError, match="refusing to wipe"):
        runner._scaffold(PAPER_ID, PROFILE, "claude-code", "innovate")
    assert fake_sciworld == []
    assert (env_dir / "scoring" / "scores_train.json").exists()

    # run_one's own retry of a just-failed attempt is allowed through.
    runner._scaffold(PAPER_ID, PROFILE, "claude-code", "innovate",
                     retry_of_own_attempt=True)
    assert len(fake_sciworld) == 1


def test_variant_and_injected_idea(eval_config, fake_sciworld) -> None:
    """The idea-implementation ablation: separate workspace, idea on disk."""
    eval_config()
    env_dir, *_ = runner._scaffold(
        PAPER_ID, PROFILE, "claude-code", "innovate",
        variant="-impl-sft", injected_idea="  Use a smaller model.  ")
    assert env_dir == _env_dir("innovate", "-impl-sft")
    idea = (env_dir / "READMEs" / "IDEA.md").read_text()
    assert "Use a smaller model." in idea
    assert not _env_dir("innovate").exists()


def test_scaffold_requires_data_dir(fake_sciworld) -> None:
    with pytest.raises(_config.ConfigError, match="data_dir"):
        runner._scaffold(PAPER_ID, PROFILE, "claude-code", "innovate")


# ── With the real sciworld package, from a local source (no network) ─────


def _has_sciworld() -> bool:
    try:
        import sciworld  # noqa: F401
        return True
    except ImportError:
        return False


def _write_source(source: Path) -> None:
    """A minimal environment repository, as cloned by sciworld."""
    files = {
        "briefing/problem.md": "# Problem\n",
        "briefing/evaluation.md": "# Evaluation\n",
        "briefing/method.md": "# Method\n",
        "briefing/overview.md": "- **Title:** T\n",
        "scoring/TARGETS.md": "t\n", "scoring/CONSTRAINTS.md": "c\n",
        "scoring/DIRECTION.md": "d\n", "scoring/EXPERIMENTS.md": "## `main`\n\ne\n",
        "scoring/reference.json": json.dumps({"experiments": {"main": {
            "results": {"paper": {"type": "proposed"}, "base": {"type": "baseline"}}}}}),
        "scoring/scores_train.json": json.dumps({"experiments": {"main": {"results": {
            "paper": {"acc": 0.5}, "base": {"acc": 0.1}}}}}),
        "scoring/scores_test.json": "{}",
        "eval/__init__.py": "",
        "eval/metrics.py": "def acc():\n    return 1.0\n",
        "eval/unrelated.py": "",
        "eval/train/evaluate.py": "from eval.metrics import acc\n",
        "eval/test/evaluate.py": "from eval.metrics import acc\n",
        "scripts/evaluate_train.sh": "#!/bin/bash\ncd /home/user\n",
        "scripts/evaluate_test.sh": "#!/bin/bash\ncd /home/user\n",
        "scripts/baseline.sh": "#!/bin/bash\n", "scripts/download.sh": "#!/bin/bash\n",
        "scripts/run.sh": "#!/bin/bash\nbash scripts/evaluate_test.sh\n",
        "scripts/method.sh": "#!/bin/bash\n",
        "method/paper_method.py": "X = 1\n",
        "baseline/base.py": "Y = 1\n",
        "data/d.json": "[]",
        "environment/container.def": "Bootstrap: docker\nFrom: python:3.11-slim\n",
        "environment/setup.sh": "pip install x\n",
        "paper/paper.md": "paper\n",
        "checkpoints/w.bin": "w\n",
    }
    for rel, text in files.items():
        path = source / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


@pytest.mark.skipif(not _has_sciworld(), reason="sciworld not installed")
@pytest.mark.parametrize("mode", ["innovate", "improve"])
def test_scaffold_with_the_real_sciworld_package(
    eval_config, stub_builders, monkeypatch, tmp_path, mode,
) -> None:
    """The harness and the installed CLI agree on the workspace contract."""
    eval_config()
    # A CLI configured for the no-container backend must not change what
    # the harness gets: its workspaces keep the container paths.
    sw_home = tmp_path / "sciworld-home"
    sw_home.mkdir()
    (sw_home / "config.json").write_text(json.dumps({"container": "venv"}))
    monkeypatch.setenv("SCIWORLD_HOME", str(sw_home))
    _write_source(_config.envs_dir() / PAPER_ID / "source")

    env_dir, _binds, _sif, _overlay = runner._scaffold(PAPER_ID, PROFILE, "claude-code", mode)

    assert env_dir == _env_dir(mode)
    # The visible evaluator arrives with the module it imports; the
    # held-out evaluator and unrelated modules do not.
    assert sorted(p.name for p in (env_dir / "eval").iterdir()) == ["__init__.py", "metrics.py", "train"]
    assert sorted(p.name for p in (env_dir / "scripts").iterdir()) == [
        "baseline.sh", "download.sh", "evaluate_train.sh", "run.sh"]
    assert "action: submit" in (env_dir / "scripts" / "run.sh").read_text(), "run.sh is the harness stub"
    assert "cd /home/user" in (env_dir / "scripts" / "evaluate_train.sh").read_text()
    assert list((env_dir / "scoring").iterdir()) == []
    assert (env_dir / "idea").is_dir() and not (env_dir / "idea.md").exists()
    for gone in ("paper", "checkpoints", "environment/setup.sh"):
        assert not (env_dir / gone).exists(), gone
    readmes = sorted(p.name for p in (env_dir / "READMEs").iterdir())
    common = ["CONSTRAINTS.md", "DIRECTION.md", "EXPERIMENTS.md", "TARGETS.md", "evaluation.md", "problem.md"]
    if mode == "improve":
        assert readmes == sorted(common + ["method.md", "overview.md", "train_reference.json"])
        reference = json.loads((env_dir / "READMEs" / "train_reference.json").read_text())
        assert reference == {"experiments": {"main": {"results": {"paper": {"acc": 0.5}}}}}
        assert (env_dir / "method" / "paper_method.py").is_file()
    else:
        assert readmes == common
        assert list((env_dir / "method").iterdir()) == []


# ── End to end against the real sciworld (network) ───────────────────────


@pytest.mark.slow
@pytest.mark.skipif(not _has_sciworld(), reason="sciworld not installed")
@pytest.mark.skipif(not os.environ.get("EVAL_TEST_PAPER_ID"),
                    reason="set EVAL_TEST_PAPER_ID to a published environment's paper id")
def test_scaffold_creates_workspace(eval_config, stub_builders) -> None:
    """Real ``sciworld.research``; only the container builders are stubbed
    (they need Apptainer and a populated cache)."""
    paper_id = os.environ["EVAL_TEST_PAPER_ID"]
    eval_config()

    env_dir, _binds, _sif, _overlay = runner._scaffold(
        paper_id, PROFILE, "claude-code", "innovate")
    try:
        assert env_dir.is_dir()
        # sciworld populates these at scaffold time. method/ is created
        # empty in innovate mode; idea.md is replaced by an idea/ folder.
        for sub in ("READMEs", "eval", "scripts", "baseline", "scoring",
                    "status.json", "idea"):
            assert (env_dir / sub).exists(), f"missing {sub} in scaffolded env_dir"
        assert (env_dir / "idea").is_dir(), "idea/ should be a folder, not a file"
        assert not (env_dir / "idea.md").exists(), "stock idea.md should be removed"
        # The sibling source/ clone must exist and not be inside env_dir.
        source = env_dir.parent / "source"
        assert source.exists(), "sciworld did not produce the source/ clone"
        assert not (env_dir / "source").exists(), "source leaked into env_dir"
    finally:
        if env_dir.parent.exists():
            shutil.rmtree(env_dir.parent, ignore_errors=True)
