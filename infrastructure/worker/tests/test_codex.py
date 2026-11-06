"""Tests for CodexWorker command building and output parsing."""

import json
from pathlib import Path

import pytest

from infrastructure.worker.codex import (
    DEFAULT_COMPACT_TOKEN_LIMIT,
    CodexWorker,
    _parse_codex_output,
)


@pytest.fixture
def codex_worker(tmp_workspace):
    w = CodexWorker("codex", "test-model", {})
    w.set_paper_and_workspace({"paper_md_path": "/tmp/paper.md"}, str(tmp_workspace))
    w.set_resources(gpu_budget=5.0, shared_datasets_path="/data/datasets",
                    shared_models_path="/data/models")
    return w


def _config_overrides(cmd):
    """Return the values of every ``-c key=value`` option in a codex command."""
    return [cmd[i + 1] for i, part in enumerate(cmd) if part == "-c"]


def test_build_command_no_container(codex_worker):
    cmd = codex_worker._build_command()
    assert cmd[1] == "exec"
    assert "--experimental-json" in cmd
    assert cmd[cmd.index("--model") + 1] == "test-model"
    assert cmd[cmd.index("-C") + 1] == codex_worker._workspace_path
    assert cmd[-1] == "-"  # prompt is read from stdin
    assert "resume" not in cmd


def test_compact_token_limit_default_and_override(codex_worker):
    assert DEFAULT_COMPACT_TOKEN_LIMIT == 120000
    assert _config_overrides(codex_worker._build_command()) == [
        "model_auto_compact_token_limit=120000",
    ]

    codex_worker.config["compact_token_limit"] = 80000
    assert _config_overrides(codex_worker._build_command()) == [
        "model_auto_compact_token_limit=80000",
    ]


def test_effort_and_resume(codex_worker):
    codex_worker.config["effort"] = "high"
    codex_worker._session_id = "sess-1"
    cmd = codex_worker._build_command()
    assert 'model_reasoning_effort="high"' in _config_overrides(cmd)
    assert cmd[cmd.index("resume") + 1] == "sess-1"
    assert cmd[-1] == "-"


def test_build_command_with_container(codex_worker):
    codex_worker.config["container_sif"] = "/path/to/reproducer.sif"
    cmd = codex_worker._build_command()
    assert cmd[:5] == ["nice", "-n", "19", "apptainer", "exec"]
    assert cmd[cmd.index("-C") + 1] == "/home/user"
    assert cmd.index("/path/to/reproducer.sif") < cmd.index("exec", 5)


def test_setup_workspace_copies_protocol_as_agents_md(codex_worker):
    codex_worker._setup_workspace()
    ws = Path(codex_worker._workspace_path)
    agents_md = ws / "AGENTS.md"
    # A copy, because protocol.md itself is hidden inside the container.
    assert agents_md.is_file() and not agents_md.is_symlink()
    assert agents_md.read_text() == (ws / "protocol.md").read_text()


def test_credentials_are_copied_only_in_container_mode(codex_worker, isolated_host_env):
    (isolated_host_env / ".codex").mkdir()
    (isolated_host_env / ".codex" / "auth.json").write_text('{"fake": "auth"}')
    ws = Path(codex_worker._workspace_path)

    codex_worker._setup_workspace()
    assert not (ws / ".codex" / "auth.json").exists()

    codex_worker.config["container_sif"] = "/path/to/reproducer.sif"
    codex_worker._setup_workspace()
    assert (ws / ".codex" / "auth.json").read_text() == '{"fake": "auth"}'
    assert ".codex/" in (ws / ".gitignore").read_text().splitlines()


def test_build_env_strips_api_keys_in_container_mode(codex_worker, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    assert codex_worker._build_env()["OPENAI_API_KEY"] == "fake-openai-key"
    codex_worker.config["container_sif"] = "/path/to/reproducer.sif"
    assert "OPENAI_API_KEY" not in codex_worker._build_env()


def test_parse_codex_output():
    lines = [
        {"type": "thread.started", "thread_id": "t-1"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "first"}},
        {"type": "item.completed", "item": {"type": "command_execution"}},
        {"type": "item.completed", "item": {"type": "file_change"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "second"}},
        {"type": "turn.completed",
         "usage": {"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 60}},
    ]
    parsed = _parse_codex_output("\n".join(json.dumps(line) for line in lines) + "\nnot json\n")

    assert parsed["session_id"] == "t-1"
    assert parsed["text"] == "first\nsecond"
    assert parsed["errors"] == []
    assert parsed["stats"]["input_tokens"] == 100
    assert parsed["stats"]["output_tokens"] == 20
    assert parsed["stats"]["cached_input_tokens"] == 60
    assert parsed["stats"]["tool_uses"] == {"shell": 1, "file_edit": 1}
    assert parsed["stats"]["num_turns"] == 2


def test_parse_codex_output_collects_errors():
    parsed = _parse_codex_output(json.dumps({"type": "error", "message": "boom"}))
    assert parsed["errors"] == ["boom"]
    assert parsed["text"] == ""
