"""A run must never be frozen by state that is not the agent's own.

Three situations the resume logic has to get right:

* ``--continue`` must not be passed when the CLI never persisted a
  conversation (for example because every earlier turn died on a
  startup dialog): the relaunch would exit with "No conversation found
  to continue".
* A session file left by a one-line headless probe is not an
  interactive conversation. Counting any session file would make an
  interactive resume pass ``--continue`` and quit.
* The grader's ``verification.json`` under ``scoring/`` is not agent
  output and must not block a retry.
"""
import json
from pathlib import Path

from evaluation.agent import has_conversation
from evaluation.runner import _scoring_touched


def _session(env: Path, *events: dict, name: str = "session-a") -> Path:
    d = env / ".claude" / "projects" / "-home-user"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.jsonl"
    # claude writes compact JSON, one event per line
    p.write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in events))
    return p


def _user(entrypoint: str) -> dict:
    return {"type": "user", "entrypoint": entrypoint, "message": {"content": "task"}}


ANSWER = {"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}}


def test_no_session_store_means_nothing_to_continue(tmp_path):
    assert has_conversation(tmp_path, "claude-tmux") is False
    (tmp_path / ".claude" / "projects").mkdir(parents=True)
    assert has_conversation(tmp_path, "claude-code") is False


def test_interactive_conversation_continues_interactively(tmp_path):
    _session(tmp_path, _user("cli"), ANSWER)
    assert has_conversation(tmp_path, "claude-tmux") is True


def test_headless_conversation_continues_headless(tmp_path):
    _session(tmp_path, _user("sdk-cli"), ANSWER)
    assert has_conversation(tmp_path, "claude-code") is True


def test_headless_probe_is_not_an_interactive_conversation(tmp_path):
    # A headless --print session in the workspace of an interactive run.
    _session(tmp_path, _user("sdk-cli"), ANSWER)
    assert has_conversation(tmp_path, "claude-tmux") is False


def test_session_where_the_model_never_answered_does_not_count(tmp_path):
    _session(tmp_path, _user("cli"))
    assert has_conversation(tmp_path, "claude-tmux") is False


def test_any_matching_session_is_enough(tmp_path):
    _session(tmp_path, _user("sdk-cli"), ANSWER, name="probe")
    _session(tmp_path, _user("cli"), ANSWER, name="agent")
    assert has_conversation(tmp_path, "claude-tmux") is True


def test_other_scaffolds_keep_their_own_session_state(tmp_path):
    assert has_conversation(tmp_path, "codex") is True


def test_grader_markers_do_not_count_as_agent_scoring(tmp_path):
    scoring = tmp_path / "scoring"
    scoring.mkdir()
    (scoring / "verification.json").write_text(json.dumps({"status": "incomplete"}))
    (scoring / "verifier_scan.json").write_text("{}")
    assert _scoring_touched(tmp_path) is False


def test_agent_scores_do_count(tmp_path):
    scoring = tmp_path / "scoring"
    scoring.mkdir()
    (scoring / "verification.json").write_text("{}")
    (scoring / "scores_train.json").write_text("{}")
    assert _scoring_touched(tmp_path) is True


def test_missing_scoring_dir(tmp_path):
    assert _scoring_touched(tmp_path) is False
