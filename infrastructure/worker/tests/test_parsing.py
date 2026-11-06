"""Tests for shared parsing utilities."""

import json

from infrastructure.worker.parsing import (
    parse_claude_output,
    parse_current_milestone,
    parse_turn_action,
)


# ── parse_turn_action ─────────────────────────────────────────────────────


def test_parse_valid_submit(tmp_path):
    (tmp_path / "action.yaml").write_text(
        "action: submit\ngpus: 2\nmilestone: method_runs\nnotes: first run\n"
    )
    action = parse_turn_action(str(tmp_path))
    assert action is not None
    assert action.action == "submit"
    assert action.gpus == 2
    assert action.milestone == "method_runs"
    assert action.notes == "first run"


def test_parse_valid_done(tmp_path):
    (tmp_path / "action.yaml").write_text("action: done\nmilestone: core_claim\n")
    action = parse_turn_action(str(tmp_path))
    assert action is not None
    assert action.action == "done"
    assert action.milestone == "core_claim"


def test_parse_test_flag(tmp_path):
    (tmp_path / "action.yaml").write_text(
        "action: submit\ngpus: 1\ntest: true\nmilestone: method_runs\n"
    )
    action = parse_turn_action(str(tmp_path))
    assert action is not None
    assert action.test is True


def test_parse_cpu_job(tmp_path):
    (tmp_path / "action.yaml").write_text(
        "action: submit\ngpus: 0\ntest: false\nmilestone: method_runs\n"
    )
    action = parse_turn_action(str(tmp_path))
    assert action is not None
    assert action.gpus == 0
    assert action.test is False


def test_parse_defaults(tmp_path):
    (tmp_path / "action.yaml").write_text("action: submit\n")
    action = parse_turn_action(str(tmp_path))
    assert action is not None
    assert action.gpus == 1
    assert action.test is False
    assert action.milestone == "none"
    assert action.notes == ""


def test_parse_invalid_action(tmp_path):
    (tmp_path / "action.yaml").write_text("action: invalid\n")
    assert parse_turn_action(str(tmp_path)) is None


def test_parse_action_with_unusable_gpus_is_invalid(tmp_path):
    """A `gpus` value that is no number makes the file invalid; it used to
    raise instead of asking the agent to fix it."""
    for value in ("", "many"):
        (tmp_path / "action.yaml").write_text(f"action: submit\ngpus: {value}\n")
        assert parse_turn_action(str(tmp_path)) is None


def test_parse_pending_action_is_invalid(tmp_path):
    """The default template has action: pending which should be invalid."""
    (tmp_path / "action.yaml").write_text("action: pending\ngpus: 1\n")
    assert parse_turn_action(str(tmp_path)) is None


def test_parse_missing_file(tmp_path):
    assert parse_turn_action(str(tmp_path)) is None


def test_parse_bad_yaml(tmp_path):
    (tmp_path / "action.yaml").write_text("{{not yaml}}: [")
    assert parse_turn_action(str(tmp_path)) is None


def test_parse_non_dict(tmp_path):
    (tmp_path / "action.yaml").write_text("- just a list\n")
    assert parse_turn_action(str(tmp_path)) is None


def test_parse_empty_file(tmp_path):
    (tmp_path / "action.yaml").write_text("")
    assert parse_turn_action(str(tmp_path)) is None


# ── parse_claude_output ───────────────────────────────────────────────────


def _jsonl(*objs):
    return "\n".join(json.dumps(o) for o in objs)


def test_parse_result():
    stdout = _jsonl(
        {"type": "result", "result": "Done!", "session_id": "sess-1",
         "cost_usd": 0.05, "num_turns": 3, "duration_api_ms": 5000}
    )
    out = parse_claude_output(stdout)
    assert out.text == "Done!"
    assert out.session_id == "sess-1"
    assert out.stats["cost_usd"] == 0.05
    assert out.stats["num_turns"] == 3


def test_parse_assistant_tokens():
    stdout = _jsonl(
        {"type": "assistant", "message": {
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "content": [{"type": "text", "text": "hello"}],
        }},
        {"type": "result", "result": "final", "session_id": "s1"},
    )
    out = parse_claude_output(stdout)
    assert out.stats["input_tokens"] == 100
    assert out.stats["output_tokens"] == 50
    assert out.text == "final"


def test_parse_tool_use_counts():
    stdout = _jsonl(
        {"type": "assistant", "message": {
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "content": [
                {"type": "tool_use", "name": "Bash"},
                {"type": "tool_use", "name": "Bash"},
                {"type": "tool_use", "name": "Read"},
            ],
        }},
        {"type": "result", "result": "", "session_id": "s1"},
    )
    out = parse_claude_output(stdout)
    assert out.stats["tool_uses"]["Bash"] == 2
    assert out.stats["tool_uses"]["Read"] == 1


def test_parse_error():
    stdout = _jsonl(
        {"type": "error", "error": {"message": "rate limited"}}
    )
    out = parse_claude_output(stdout)
    assert "rate limited" in out.errors[0]


def test_parse_empty_stdout():
    out = parse_claude_output("")
    assert out.text == ""
    assert out.session_id is None


def test_parse_non_json_lines():
    out = parse_claude_output("not json\nalso not json\n")
    assert out.text.strip() == "not json\nalso not json"


def test_parse_single_json_fallback():
    """If stdout is a single JSON object (not JSONL), still works."""
    stdout = json.dumps({"result": "hello", "session_id": "s2", "cost_usd": 0.01})
    out = parse_claude_output(stdout)
    assert out.text == "hello"
    assert out.session_id == "s2"


# ── parse_current_milestone ───────────────────────────────────────────────


def test_parse_milestone(tmp_path):
    (tmp_path / "MILESTONES.md").write_text(
        "# Milestones\n\n**Current: core_claim**\n\n## Log\n"
    )
    assert parse_current_milestone(str(tmp_path)) == "core_claim"


def test_parse_milestone_none(tmp_path):
    (tmp_path / "MILESTONES.md").write_text("# Milestones\n\n**Current: none**\n")
    assert parse_current_milestone(str(tmp_path)) == "none"


def test_parse_milestone_missing_file(tmp_path):
    assert parse_current_milestone(str(tmp_path)) is None


def test_parse_milestone_no_match(tmp_path):
    (tmp_path / "MILESTONES.md").write_text("# Nothing here\nJust notes.\n")
    assert parse_current_milestone(str(tmp_path)) is None
