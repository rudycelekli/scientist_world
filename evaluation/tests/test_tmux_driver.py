"""Tests for the interactive-tmux scaffold driver (evaluation.tmux_driver).

Two layers:
  * pure-logic unit tests for busy/completion detection (no tmux);
  * a real-tmux integration test that drives a *mock* claude TUI through
    the full ``run_tmux_turn`` path — exercising new-session, send-keys,
    capture-pane polling, the action.yaml turn signal, and clean exit —
    without touching the Anthropic API or apptainer.

The mock TUI is faithful in the one way that matters for detection: it
redraws its screen (ANSI clear) each frame, so the "esc to interrupt"
busy marker is present on the live screen while "working" and *gone*
once idle — exactly how the real claude TUI behaves, and why the driver
inspects the visible pane rather than scrollback.
"""

import shutil
import sys
import textwrap
import time
from pathlib import Path

import pytest

from evaluation import tmux_driver as td


# ── pure-logic unit tests ───────────────────────────────────────────────
def test_tui_is_busy_detects_marker():
    assert td.tui_is_busy("streaming... (esc to interrupt)")
    assert td.tui_is_busy("foo\n  ESC to interrupt  \nbar")


def test_tui_is_busy_false_when_idle():
    assert not td.tui_is_busy("turn complete.\n> ")
    assert not td.tui_is_busy("")


def test_is_bypass_prompt():
    pane = (
        "  WARNING: Claude Code running in Bypass Permissions mode\n"
        "  By proceeding, you accept all responsibility ...\n"
        "  > 1. No, exit\n    2. Yes, I accept\n  Enter to confirm\n"
    )
    assert td.is_bypass_prompt(pane)
    # a normal chat screen must not be mistaken for the confirmation
    assert not td.is_bypass_prompt("turn complete.\n> ")


def test_bypass_cursor_on_accept():
    on_no = "  ❯ 1. No, exit\n    2. Yes, I accept\n"
    on_yes = "    1. No, exit\n  ❯ 2. Yes, I accept\n"
    assert not td.bypass_cursor_on_accept(on_no)
    assert td.bypass_cursor_on_accept(on_yes)
    # ascii fallback pointer
    assert td.bypass_cursor_on_accept("  1. No, exit\n> 2. Yes, I accept\n")


def test_turn_is_complete_busy_never_completes():
    assert not td.turn_is_complete(
        action_advanced=True, busy=True, stable_for=10_000,
    )


def test_turn_is_complete_action_plus_settle():
    # action.yaml written + brief stability ⇒ done
    assert td.turn_is_complete(
        action_advanced=True, busy=False,
        stable_for=td.SETTLE_AFTER_ACTION_SEC,
    )
    # action.yaml written but not yet settled ⇒ not done
    assert not td.turn_is_complete(
        action_advanced=True, busy=False,
        stable_for=td.SETTLE_AFTER_ACTION_SEC - 0.01,
    )


def test_turn_is_complete_pure_idle_fallback():
    # no action.yaml, but long pure-idle stability ⇒ done
    assert td.turn_is_complete(
        action_advanced=False, busy=False,
        stable_for=td.SETTLE_PURE_IDLE_SEC,
    )
    # short stability without action.yaml ⇒ not done
    assert not td.turn_is_complete(
        action_advanced=False, busy=False,
        stable_for=td.SETTLE_PURE_IDLE_SEC - 0.01,
    )


# ── real-tmux integration test (mock TUI) ───────────────────────────────
_MOCK_TUI = textwrap.dedent(
    r'''
    import sys, time
    CLEAR = "\033[2J\033[3J\033[H"
    action_path = sys.argv[1]
    work_secs = float(sys.argv[2])

    def draw(s):
        sys.stdout.write(CLEAR + s + "\n")
        sys.stdout.flush()

    draw("mock claude ready\n> ")
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if line == "/exit":
            break
        if not line:
            continue
        # "generate": redraw with the busy marker for work_secs
        end = time.time() + work_secs
        while time.time() < end:
            draw("thinking tokens... (esc to interrupt)")
            time.sleep(0.15)
        # end-of-turn signal the eval loop relies on
        with open(action_path, "w") as f:
            f.write("action: done\n")
        # idle frame — busy marker is gone from the live screen
        draw("turn complete.\n> ")
    '''
)


@pytest.fixture
def empty_home(tmp_path, monkeypatch):
    """The driver seeds the workspace from the host's ``~/.claude.json``
    and tmux reads ``~/.tmux.conf``: point HOME at an empty directory so
    the test does not depend on (or copy) the caller's own files."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


@pytest.fixture
def fast_driver(monkeypatch, empty_home):
    """Shrink the driver's timing constants so the test runs in seconds."""
    monkeypatch.setattr(td, "POLL_INTERVAL_SEC", 0.2)
    monkeypatch.setattr(td, "READY_STABLE_SEC", 0.4)
    monkeypatch.setattr(td, "SETTLE_AFTER_ACTION_SEC", 0.4)
    monkeypatch.setattr(td, "SETTLE_PURE_IDLE_SEC", 3.0)
    monkeypatch.setattr(td, "READY_TIMEOUT_SEC", 15.0)
    monkeypatch.setattr(td, "INACTIVITY_TIMEOUT_SEC", 8.0)
    monkeypatch.setattr(td, "MAX_TURN_SEC", 30.0)


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux not installed")
def test_run_tmux_turn_end_to_end(tmp_path, fast_driver):
    """Full driver path against a mock TUI: ready → send → idle → done."""
    mock = tmp_path / "mock_tui.py"
    mock.write_text(_MOCK_TUI)
    env_dir = tmp_path / "ws"
    (env_dir / "trajectory").mkdir(parents=True)
    raw_path = env_dir / "trajectory" / "turn_0001_raw.jsonl"
    action_path = env_dir / "action.yaml"

    # container_cmd is normally `apptainer exec ... claude`; here it's the
    # mock TUI. The driver does not care what the pane process is.
    container_cmd = [sys.executable, str(mock), str(action_path), "1.0"]

    rc, captured, completed = td.run_tmux_turn(
        container_cmd, "DO THE THING", raw_path,
        env=None, env_dir=env_dir, turn=1, action_path=action_path,
    )

    assert completed is True
    assert rc == 0
    # the turn prompt was written for the agent to read
    assert (env_dir / ".eval" / "turn_0001_prompt.md").read_text() == "DO THE THING"
    # the agent's end-of-turn signal landed
    assert action_path.exists()
    # trajectory captured the idle frame
    assert raw_path.exists()
    assert "turn complete" in captured


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux not installed")
def test_run_tmux_turn_hang_times_out(tmp_path, monkeypatch, empty_home):
    """A TUI that never signals/changes is declared incomplete, not hung forever."""
    monkeypatch.setattr(td, "POLL_INTERVAL_SEC", 0.2)
    monkeypatch.setattr(td, "READY_STABLE_SEC", 0.4)
    monkeypatch.setattr(td, "READY_TIMEOUT_SEC", 10.0)
    monkeypatch.setattr(td, "INACTIVITY_TIMEOUT_SEC", 2.0)  # quick hang verdict
    monkeypatch.setattr(td, "SETTLE_PURE_IDLE_SEC", 999.0)  # disable idle-complete
    monkeypatch.setattr(td, "MAX_TURN_SEC", 20.0)

    # A TUI that draws a ready screen then goes silent forever (never
    # writes action.yaml, never changes the screen).
    silent = tmp_path / "silent_tui.py"
    silent.write_text(textwrap.dedent(r'''
        import sys, time
        sys.stdout.write("\033[2J\033[H idle screen\n> ")
        sys.stdout.flush()
        time.sleep(600)
    '''))
    env_dir = tmp_path / "ws"
    (env_dir / "trajectory").mkdir(parents=True)
    raw_path = env_dir / "trajectory" / "turn_0001_raw.jsonl"
    action_path = env_dir / "action.yaml"

    rc, _captured, completed = td.run_tmux_turn(
        [sys.executable, str(silent)], "x", raw_path,
        env=None, env_dir=env_dir, turn=1, action_path=action_path,
    )
    assert completed is False
    assert rc == 1
    assert not action_path.exists()
