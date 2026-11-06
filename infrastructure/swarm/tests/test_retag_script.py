"""Tests for scripts/retag_github_topics.py with the GitHub API mocked."""

import importlib.util
import json
from pathlib import Path

import pytest

from swarm.config import SwarmConfig, WorkflowStep

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "retag_github_topics.py"


@pytest.fixture
def retag(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("retag_github_topics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    data_dir = tmp_path / "data"
    md_dir = data_dir / "arxiv_mds" / "2512.01152"
    md_dir.mkdir(parents=True)
    (md_dir / "2512.01152.md").write_text("# A Paper Title\n")

    workspace = tmp_path / "workspaces" / "2512.01152"
    (workspace / "briefing").mkdir(parents=True)
    (workspace / "briefing" / "overview.md").write_text("- **Domain:** NLP / Language Models\n")

    papers_dir = tmp_path / "papers"
    (papers_dir / "2512.01152").mkdir(parents=True)
    (papers_dir / "2512.01152" / "meta.json").write_text(json.dumps({
        "arxiv_id": "2512.01152", "status": "done", "milestone": "core_claim",
        "workspace_path": str(workspace), "date": "2025-12-01", "topics": ["custom"],
        "workflow": "standard", "workflow_step": 1, "iteration": 2,
    }))

    config = SwarmConfig(
        scout_data_dir=str(data_dir),
        github_org="example-org",
        workflows={"standard": [WorkflowStep(profile="p"), WorkflowStep(profile="p")]},
    )
    calls = []

    def fake_api(method, path, token, body=None, retries=3):
        calls.append((method, path, body))
        if method == "GET":
            return 200, {"names": ["milestone-method-runs", "iteration-1-complete", "kept"]}
        return 200, {}

    monkeypatch.setattr(module, "PAPERS_DIR", papers_dir)
    monkeypatch.setattr(module, "load_config", lambda: config)
    monkeypatch.setattr(module, "_get_github_token", lambda: "fake-token")
    monkeypatch.setattr(module, "_github_api", fake_api)
    module.calls = calls
    module.config = config
    return module


def test_repair_rebuilds_description_and_topics(retag):
    ok, message = retag._repair_one("2512.01152", missing_only=False, verbose=False)

    assert ok, message
    repo = "/repos/example-org/2512.01152-a-paper-title"
    assert retag.calls[0] == ("GET", f"{repo}/topics", None)
    assert retag.calls[1] == ("PATCH", repo, {
        "description": "core_claim: A Paper Title",
        "homepage": "https://arxiv.org/abs/2512.01152",
    })
    method, path, body = retag.calls[2]
    assert (method, path) == ("PUT", f"{repo}/topics")
    # Current milestone replaces the stale one; other existing topics are kept.
    assert body["names"] == [
        "milestone-core-claim", "nlp", "language-models", "date-2025-12-01",
        "custom", "iteration-2-complete", "done", "iteration-1-complete", "kept",
    ]


def test_repair_missing_only_skips_repos_that_have_topics(retag):
    ok, message = retag._repair_one("2512.01152", missing_only=True, verbose=False)

    assert ok and message.startswith("skipped")
    assert [call[0] for call in retag.calls] == ["GET"]


def test_repair_reports_missing_prerequisites(retag, monkeypatch):
    assert retag._repair_one("0000.00000", missing_only=False, verbose=False) == (
        False, "meta.json not found",
    )
    monkeypatch.setattr(retag, "_get_github_token", lambda: None)
    assert retag._repair_one("2512.01152", missing_only=False, verbose=False) == (
        False, "no GitHub token found",
    )
    retag.config.github_org = ""
    assert retag._repair_one("2512.01152", missing_only=False, verbose=False) == (
        False, "github_org is not configured",
    )
