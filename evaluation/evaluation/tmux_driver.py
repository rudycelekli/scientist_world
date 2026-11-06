"""Drive an interactive ``claude`` TUI through tmux, one eval turn at a time.

Experimental: this backs the ``claude-tmux`` scaffold. It depends on the
text the TUI draws, so a new CLI release can break it, and token usage
is not extracted for runs that use it (see ``evaluation.usage``).

Why this exists
---------------
The headless path (``claude --print``) feeds a prompt on stdin, reads
``stream-json`` events, and exits after one turn. When ``--print`` is
unavailable we drive the *interactive* TUI instead: a host-side tmux
session runs ``apptainer exec ... claude`` (interactive) in a pty, we
type the turn's prompt with ``send-keys``, wait for the turn to finish,
and scrape the pane for the trajectory log.

The eval loop's control flow does not change: the agent still signals
its intent by writing ``<env_dir>/action.yaml`` (see
``evaluation.parsing``), and the loop reads that file after every turn.
So this driver only has to do two things the headless path got for
free: (1) drive the TUI, and (2) decide when a turn is done.

Turn-completion signal
----------------------
The interactive TUI emits no ``stream-json``, so there is no
``turn.completed`` event to watch. Instead we use the eval's own
protocol: the agent edits ``action.yaml`` at the end of every turn.
A turn is complete when ``action.yaml``'s mtime has advanced past the
turn's start *and* the pane has stopped changing for a short settle
window. A longer pure-stability fallback covers a turn that never
writes ``action.yaml`` (a misbehaving agent), and a no-output
inactivity timeout matches the headless path's hang detection.

The only TUI-string dependency is :data:`BUSY_MARKERS` (text the TUI
shows while generating); it is used as a guard, never as the sole
signal, so imperfect markers degrade to the stability heuristic rather
than breaking turn detection.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Onboarding/config flags that mark claude-code's first-run flow as done.
# The interactive TUI shows a login-method / theme / onboarding sequence
# on first run; the headless ``--print`` path skips it. We seed these
# from the host's ~/.claude.json (a fully-onboarded install) into the
# container workspace so the TUI opens straight on the chat box (modulo
# the bypass-permissions confirmation, handled separately).
_ONBOARDING_KEYS = (
    "hasCompletedOnboarding",
    "lastOnboardingVersion",
    "theme",
    "tipsHistory",
    "hasIdeOnboardingBeenShown",
    "hasAcknowledgedCostThreshold",
    "firstStartTime",
    # The account identity must be present too: on turn 1 the fresh
    # workspace has no .claude.json yet, so without these claude sees
    # "onboarded but no account" and exits before rendering the TUI.
    "oauthAccount",
    "userID",
)

# ── Tunables ────────────────────────────────────────────────────────────
# Wide pane so the TUI wraps less and capture-pane text stays readable.
TMUX_COLS = 220
TMUX_ROWS = 50

POLL_INTERVAL_SEC = 2.0          # how often we capture-pane while waiting
READY_STABLE_SEC = 4.0           # pane stable this long ⇒ TUI finished drawing
# The TUI boots through up to three screens (trust folder, bypass
# permissions, release notes) before the chat box, and inside the
# container it runs at low priority: on a busy host each redraw can
# take tens of seconds, so a fresh session can need several minutes.
READY_TIMEOUT_SEC = 600.0        # give up waiting for the TUI to come up
DIALOG_CONFIRM_SEC = 60.0        # after Enter on a dialog, wait for the screen to move
SETTLE_AFTER_ACTION_SEC = 3.0    # stability needed once action.yaml is written
SETTLE_PURE_IDLE_SEC = 60.0      # stability needed when no action.yaml was written
INACTIVITY_TIMEOUT_SEC = 1200.0  # no pane change for this long ⇒ hang (20 min)
MAX_TURN_SEC = 3600.0            # absolute per-turn ceiling (1 h)
QUIT_TIMEOUT_SEC = 15.0          # wait for a clean /exit before kill-session
PROMPT_ENTER_DELAY_SEC = 1.0     # gap between typing the prompt and Enter
SUBMIT_CONFIRM_SEC = 20.0        # wait for the TUI to react to that Enter
SUBMIT_ENTER_ATTEMPTS = 3        # re-press Enter this many times before giving up

# Text the claude-code TUI shows *while generating*. Used only to avoid
# declaring a turn done mid-stream; the action.yaml + stability signals
# carry the real decision, so an incomplete list here is safe. Tune
# against a real capture (capture-pane during generation) when possible.
BUSY_MARKERS = (
    "esc to interrupt",
    "ESC to interrupt",
)

# Interactive ``claude --dangerously-skip-permissions`` opens with a
# "Bypass Permissions mode" confirmation whose default cursor is on
# "No, exit" — pressing Enter there quits claude. The headless --print
# path never shows this. We detect it and select "Yes, I accept" before
# sending the turn prompt.
BYPASS_CONFIRM_MARKERS = (
    "Bypass Permissions mode",
    "accept all responsibility",
)

# The interactive TUI also shows a "trust this folder?" confirmation on a
# new working directory (the headless path never does). Its default
# cursor is on "Yes, I trust this folder", so a bare Enter accepts.
TRUST_CONFIRM_MARKERS = (
    "trust this folder",
    "Is this a project you created or one you trust",
)

CAPTURE_SCROLLBACK = 4000        # lines of history to include in a capture


# ── tmux primitives ─────────────────────────────────────────────────────
def _tmux(socket: str, args: list[str], *, env: dict | None = None,
          check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run ``tmux -L <socket> <args>``.

    A dedicated socket (``-L``) isolates this run's tmux server so the
    first call's ``env`` becomes the server's environment — which the
    ``apptainer`` command inherits (it needs the ``APPTAINERENV_*`` vars
    the eval sets). Later calls just talk to that server; their env is
    irrelevant.
    """
    cmd = ["tmux", "-L", socket, *args]
    return subprocess.run(
        cmd, env=env, check=check,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def _capture(socket: str, session: str, *, full: bool = False) -> str:
    """Return the pane's text, no escape codes.

    ``full=False`` (default) captures only the *visible* screen — what
    the TUI is showing right now. This is what busy/stability detection
    must use: a real TUI redraws its status line in place, so a stale
    "esc to interrupt" lingers in scrollback but not on the live screen.
    ``full=True`` includes scrollback for the per-turn trajectory log.
    """
    args = ["capture-pane", "-p", "-t", session]
    if full:
        args += ["-S", f"-{CAPTURE_SCROLLBACK}"]
    try:
        cp = _tmux(socket, args, check=False, capture=True)
    except FileNotFoundError:
        return ""
    if cp.returncode != 0:
        return ""
    return cp.stdout.decode("utf-8", errors="replace")


def _session_alive(socket: str, session: str) -> bool:
    cp = _tmux(socket, ["has-session", "-t", session], check=False, capture=True)
    return cp.returncode == 0


# ── pure helpers (unit-tested) ──────────────────────────────────────────
def tui_is_busy(pane: str) -> bool:
    """True if the captured pane shows a 'generating' indicator."""
    tail = pane[-2000:]
    return any(marker in tail for marker in BUSY_MARKERS)


def is_bypass_prompt(pane: str) -> bool:
    """True if the 'Bypass Permissions mode' confirmation is on screen."""
    return (
        any(m in pane for m in BYPASS_CONFIRM_MARKERS)
        and "Yes, I accept" in pane
    )


def is_trust_prompt(pane: str) -> bool:
    """True if the 'trust this folder?' confirmation is on screen."""
    return any(m in pane for m in TRUST_CONFIRM_MARKERS) and "trust this folder" in pane.lower()


def is_whatsnew_prompt(pane: str) -> bool:
    """True if the version-change 'What's new' / release-notes screen is
    up. New claude CLI versions render this once after an upgrade; it is
    not one of the onboarding keys we seed, so it must be dismissed with
    Escape or the driver mistakes it for the ready chat box and the
    prompt never lands (the TUI then exits rc=1)."""
    low = pane.lower()
    return "/release-notes for more" in low or "what's new" in low


def bypass_cursor_on_accept(pane: str) -> bool:
    """True if the bypass-confirmation cursor sits on 'Yes, I accept'."""
    for line in pane.splitlines():
        if ("❯" in line or line.lstrip().startswith(">")) and "Yes, I accept" in line:
            return True
    return False


def trust_cursor_on_accept(pane: str) -> bool:
    """True if the trust-folder cursor sits on 'Yes, I trust this folder'."""
    for line in pane.splitlines():
        if ("❯" in line or line.lstrip().startswith(">")) and "trust this folder" in line.lower():
            return True
    return False


def turn_is_complete(
    *, action_advanced: bool, busy: bool, stable_for: float,
) -> bool:
    """Decide whether the current turn has finished.

    Primary: the agent wrote ``action.yaml`` this turn and the pane has
    settled briefly. Fallback: the pane has been idle (not busy, no
    change) for a long window even though no ``action.yaml`` appeared.
    """
    if busy:
        return False
    if action_advanced and stable_for >= SETTLE_AFTER_ACTION_SEC:
        return True
    if stable_for >= SETTLE_PURE_IDLE_SEC:
        return True
    return False


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _ensure_onboarded(env_dir: Path) -> None:
    """Seed the workspace ``.claude.json`` so the TUI skips first-run onboarding.

    Merges the host's onboarding flags into the container workspace's
    ``.claude.json`` (preserving its ``oauthAccount`` and migration keys)
    and forces ``hasCompletedOnboarding``. Without this the interactive
    TUI opens on the "Select login method" screen and never reaches chat.
    """
    cj = env_dir / ".claude.json"
    try:
        data = json.loads(cj.read_text()) if cj.exists() else {}
    except (OSError, ValueError):
        data = {}
    try:
        host = json.loads((Path.home() / ".claude.json").read_text())
    except (OSError, ValueError):
        host = {}
    for k in _ONBOARDING_KEYS:
        if k in host:
            data[k] = host[k]
    data["hasCompletedOnboarding"] = True
    try:
        cj.parent.mkdir(parents=True, exist_ok=True)
        cj.write_text(json.dumps(data))
    except OSError as exc:
        log.warning("could not seed onboarding flags into %s: %s", cj, exc)


# ── readiness + turn loop ───────────────────────────────────────────────
def _wait_ready(socket: str, session: str, timeout: float) -> bool:
    """Wait until the TUI has drawn its screen (non-empty + stable).

    Marker-independent: we just wait for the pane to become non-empty
    and stop changing, which is true once claude finishes rendering its
    welcome/input screen. Works for the real TUI and the test mock alike.
    """
    start = time.monotonic()
    last = None
    last_change = start
    while time.monotonic() - start < timeout:
        if not _session_alive(socket, session):
            return False
        pane = _capture(socket, session)
        now = time.monotonic()
        if pane != last:
            last = pane
            last_change = now
        elif pane.strip() and (now - last_change) >= READY_STABLE_SEC:
            return True
        time.sleep(POLL_INTERVAL_SEC)
    return bool(last and last.strip())


def _confirm_dialog(socket: str, session: str, shown: str) -> None:
    """Press Enter on a startup dialog, then wait for the screen to move on.

    The TUI can take tens of seconds to redraw inside the container (it
    runs at low priority and the host may be busy), and the two startup dialogs
    — trust-folder, then bypass-permissions — both default to "No, exit".
    Re-sending Enter every second while the first dialog is still on
    screen queues extra keystrokes, and the surplus Enter lands on the
    NEXT dialog's default: claude exits and the pane goes blank. So press
    Enter exactly once and hold until the pane has changed (or a generous
    deadline passes, after which the caller's loop takes over again).
    """
    _tmux(socket, ["send-keys", "-t", session, "Enter"], check=False)
    deadline = time.monotonic() + DIALOG_CONFIRM_SEC
    while time.monotonic() < deadline:
        time.sleep(1.0)
        if not _session_alive(socket, session):
            return
        if _capture(socket, session) != shown:
            return


def _prepare_session(socket: str, session: str, timeout: float) -> bool:
    """Drive the TUI from launch to a ready chat box.

    Walks the interactive startup confirmations the headless ``--print``
    path never shows — "trust this folder?" (default Enter accepts) and
    "Bypass Permissions mode" (default would quit, so move down to
    "Yes, I accept") — dismissing each as it appears. Returns True once a
    non-confirmation screen has been stable for the settle window (chat
    box ready), False on timeout or if the session dies.

    Returns ``(ok, last_pane)``: the last non-empty full capture is
    preserved so a TUI that exits immediately (auth failure, refusal,
    crash) still leaves its final screen for diagnostics — capturing
    after the session dies returns nothing.
    """
    start = time.monotonic()
    last = None
    last_change = start
    last_full = ""
    while time.monotonic() - start < timeout:
        if not _session_alive(socket, session):
            return False, last_full
        vis = _capture(socket, session)
        full = _capture(socket, session, full=True)
        if full.strip():
            last_full = full
        now = time.monotonic()
        if vis != last:
            last = vis
            last_change = now
        if is_whatsnew_prompt(full) or is_whatsnew_prompt(vis):
            # Release-notes overlay after a CLI upgrade — Escape closes it.
            _tmux(socket, ["send-keys", "-t", session, "Escape"], check=False)
            time.sleep(1.0)
            last = None
            continue
        if is_trust_prompt(vis):
            # CLI >=2.1.251 defaults the cursor to "No, exit"; a bare
            # Enter there QUITS claude (rc=1). Move to "Yes, I trust
            # this folder" and only confirm once the cursor is on it.
            if trust_cursor_on_accept(vis):
                _confirm_dialog(socket, session, vis)
            else:
                _tmux(socket, ["send-keys", "-t", session, "Down"], check=False)
                time.sleep(0.8)
            last = None
            continue
        if is_bypass_prompt(vis):
            # The default cursor is on "No, exit" — Enter there quits
            # claude. Under load the Down keystroke can land late, so
            # only confirm after the cursor has visibly moved to
            # "Yes, I accept"; otherwise loop and retry the Down.
            if bypass_cursor_on_accept(vis):
                _confirm_dialog(socket, session, vis)
            else:
                _tmux(socket, ["send-keys", "-t", session, "Down"], check=False)
                time.sleep(0.8)
            last = None
            continue
        # No known confirmation on screen. Once it's been non-empty and
        # unchanged for the settle window, it's the chat box.
        if vis.strip() and (now - last_change) >= READY_STABLE_SEC:
            return True, last_full
        time.sleep(POLL_INTERVAL_SEC)
    return False, last_full


def _newest_session_file(env_dir: Path) -> Path | None:
    """Newest claude session jsonl under the workspace's .claude/projects/."""
    proj = env_dir / ".claude" / "projects"
    if not proj.is_dir():
        return None
    files = [
        p
        for d in proj.iterdir() if d.is_dir()
        for p in d.glob("*.jsonl")
    ]
    if not files:
        return None
    return max(files, key=_mtime)


def _capture_session_delta(env_dir: Path, turn: int) -> None:
    """Save this turn's slice of claude's own session jsonl.

    The pane scrape in ``turn_NNNN_raw.jsonl`` is rendered TUI text —
    wrapped lines, no structured events. Claude persists the *real*
    trajectory (structured messages, tool calls) to
    ``.claude/projects/<dir>/<session>.jsonl`` in the workspace; we copy
    the lines appended since the previous turn to
    ``trajectory/turn_NNNN_session.jsonl`` so tmux runs keep structured
    per-turn records like the headless scaffolds. Byte offset is tracked
    in ``.eval/session_capture.json``; a new session file resets it.
    """
    src = _newest_session_file(env_dir)
    if src is None:
        return
    state_path = env_dir / ".eval" / "session_capture.json"
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        state = {}
    offset = int(state.get("offset") or 0) if state.get("path") == str(src) else 0
    try:
        data = src.read_bytes()
    except OSError:
        return
    if offset > len(data):
        offset = 0
    new = data[offset:]
    if new.strip():
        from evaluation.agent import redact_secrets
        out = env_dir / "trajectory" / f"turn_{turn:04d}_session.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(redact_secrets(new.decode("utf-8", errors="replace")))
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"path": str(src), "offset": len(data)}))
    except OSError as exc:
        log.warning("could not persist session capture offset: %s", exc)


def _send_prompt(socket: str, session: str, container_prompt_path: str) -> None:
    """Type a short instruction telling the agent to read the prompt file.

    The real prompt (~13 KB, full of markdown/special chars) is written
    to a file rather than typed: ``send-keys`` cannot reliably push that
    much text, and the TUI's input box mangles large multiline pastes.
    """
    instruction = (
        f"Your instructions for this turn are in the file "
        f"{container_prompt_path}. Read that file now and carry out "
        f"exactly what it says."
    )
    # ``-l`` sends the text literally (no key-name interpretation), then a
    # separate Enter submits it.
    #
    # The Enter must not ride in the same terminal read as the text. The
    # TUI treats a burst of input arriving at once as a paste and folds an
    # embedded newline INTO the message instead of submitting it; on a
    # busy host tmux's two writes coalesce often enough to hit this. The
    # prompt then sits unsent in the input box, the pane never goes busy,
    # and the turn is scored as "action.yaml missing/invalid", so a whole
    # run can use up its turn budget without the model ever being asked
    # anything. So: pause, hit Enter, and confirm the TUI reacted before
    # handing off to the poller.
    _tmux(socket, ["send-keys", "-t", session, "-l", instruction], check=False)
    time.sleep(PROMPT_ENTER_DELAY_SEC)
    typed = _capture(socket, session)
    for attempt in range(1, SUBMIT_ENTER_ATTEMPTS + 1):
        _tmux(socket, ["send-keys", "-t", session, "Enter"], check=False)
        deadline = time.monotonic() + SUBMIT_CONFIRM_SEC
        while time.monotonic() < deadline:
            time.sleep(POLL_INTERVAL_SEC)
            if not _session_alive(socket, session):
                return
            pane = _capture(socket, session)
            # Any reaction at all — spinner, busy marker, streamed output —
            # means the prompt was accepted. A prompt still sitting in the
            # input box leaves the pane byte-identical.
            if pane != typed or tui_is_busy(pane):
                return
        log.warning("tmux turn: prompt not submitted after Enter "
                    "(attempt %d/%d) — resending",
                    attempt, SUBMIT_ENTER_ATTEMPTS)
    log.error("tmux turn: TUI never accepted the prompt after %d Enter "
              "attempts — turn will be reported incomplete",
              SUBMIT_ENTER_ATTEMPTS)


def _poll_turn(
    socket: str, session: str, action_path: Path, action_mtime0: float,
) -> tuple[str, bool]:
    """Capture-poll until the turn completes, hangs, or hits the ceiling.

    Returns ``(final_pane, completed)``. ``completed`` is False on a hang
    (no output for :data:`INACTIVITY_TIMEOUT_SEC`) or the absolute
    :data:`MAX_TURN_SEC` ceiling.
    """
    start = time.monotonic()
    last_pane = ""
    last_change = start
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        if not _session_alive(socket, session):
            # Pane process exited on its own (agent quit / crashed). Treat
            # as complete iff it managed to write action.yaml this turn.
            return last_pane, _mtime(action_path) > action_mtime0
        pane = _capture(socket, session)
        now = time.monotonic()
        if pane != last_pane:
            last_pane = pane
            last_change = now
        stable_for = now - last_change

        if turn_is_complete(
            action_advanced=_mtime(action_path) > action_mtime0,
            busy=tui_is_busy(pane),
            stable_for=stable_for,
        ):
            return pane, True

        # Hang: nothing changed for the inactivity window and the agent
        # never signalled via action.yaml.
        if (now - last_change) >= INACTIVITY_TIMEOUT_SEC:
            log.warning("tmux turn: no output for %.0fs — treating as hang",
                        INACTIVITY_TIMEOUT_SEC)
            return pane, False
        # Absolute backstop for a TUI that keeps redrawing forever.
        if (now - start) >= MAX_TURN_SEC:
            log.warning("tmux turn: hit MAX_TURN_SEC ceiling")
            return pane, _mtime(action_path) > action_mtime0


def _quit_claude(socket: str, session: str) -> None:
    """Exit the TUI cleanly so ``--continue`` can resume next turn.

    ``/exit`` makes claude persist the session to ~/.claude/projects/
    before quitting. We wait briefly for the pane process to die; the
    caller's ``finally`` kills the session unconditionally as a backstop.
    """
    if not _session_alive(socket, session):
        return
    # Clear the input box first. If the turn's prompt never got submitted
    # it is still sitting there, and typing /exit onto the end submits the
    # two glued together as one message — which both wastes a real model
    # call and poisons the next turn's transcript with a bogus prompt.
    _tmux(socket, ["send-keys", "-t", session, "C-u"], check=False)
    time.sleep(0.5)
    _tmux(socket, ["send-keys", "-t", session, "-l", "/exit"], check=False)
    time.sleep(PROMPT_ENTER_DELAY_SEC)
    _tmux(socket, ["send-keys", "-t", session, "Enter"], check=False)
    deadline = time.monotonic() + QUIT_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if not _session_alive(socket, session):
            return
        time.sleep(0.5)


# ── public entry point ──────────────────────────────────────────────────
def run_tmux_turn(
    container_cmd: list[str],
    prompt: str,
    raw_path: Path,
    *,
    env: dict,
    env_dir: Path,
    turn: int,
    action_path: Path | None = None,
) -> tuple[int, str, bool]:
    """Run one eval turn by driving an interactive ``claude`` via tmux.

    ``container_cmd`` is the fully-built ``apptainer exec ... claude``
    (interactive) argv. It is launched inside a fresh, isolated tmux
    server; the turn's prompt is written to ``<env_dir>/.eval/`` and the
    agent is told to read it. The captured pane is written to
    ``raw_path`` (the per-turn trajectory file).

    Returns ``(returncode, captured_pane, completed)`` where
    ``returncode`` is 0 on a completed turn, 1 otherwise — matching the
    headless path's ``(rc, stdout, _)`` so the caller is unchanged.
    """
    if action_path is None:
        action_path = env_dir / "action.yaml"
    socket = f"evaltmux_{os.getpid()}_{turn}"
    session = "agent"

    # Write the turn prompt to a file the agent can read from inside the
    # container (workspace is bound at /home/user).
    rel = Path(".eval") / f"turn_{turn:04d}_prompt.md"
    (env_dir / rel).parent.mkdir(parents=True, exist_ok=True)
    (env_dir / rel).write_text(prompt)
    container_prompt_path = f"/home/user/{rel.as_posix()}"

    action_mtime0 = _mtime(action_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip the interactive first-run login/onboarding screens.
    _ensure_onboarded(env_dir)

    # Clear any stale server on this (pid,turn) socket, then start the
    # session running the container command. ``new-session`` takes a
    # shell string, so we shlex-join the argv.
    _tmux(socket, ["kill-server"], check=False, capture=True)
    shell_cmd = shlex.join(container_cmd)
    try:
        _tmux(
            socket,
            ["new-session", "-d", "-s", session,
             "-x", str(TMUX_COLS), "-y", str(TMUX_ROWS), shell_cmd],
            env=env,
        )
    except Exception as exc:
        log.error("tmux new-session failed: %s", exc)
        return 1, f"tmux new-session failed: {exc}", False

    try:
        ready, last_full = _prepare_session(socket, session, READY_TIMEOUT_SEC)
        if not ready:
            # Startup failure (TUI exited or never drew the chat box).
            # The last screen goes to logs/, NOT trajectory/: a 0-byte or
            # diagnostic-only trajectory file would trip the runner's
            # wipe-on-requeue guard and block retries of a run that never
            # did any work.
            full = _capture(socket, session, full=True) or last_full
            fail_log = env_dir / "logs" / f"tmux_turn_{turn:04d}_failure.log"
            fail_log.parent.mkdir(parents=True, exist_ok=True)
            fail_log.write_text(full or "(no pane output captured)\n")
            _capture_session_delta(env_dir, turn)
            log.warning("tmux turn %d: TUI never became ready — last screen in %s",
                        turn, fail_log)
            return 1, full, False

        _send_prompt(socket, session, container_prompt_path)
        _visible, completed = _poll_turn(socket, session, action_path, action_mtime0)
        # Persist the full scrollback (not just the visible screen) as the
        # turn's trajectory record. Fall back to the last poll capture if
        # the session already died, and never leave a 0-byte file behind.
        full = _capture(socket, session, full=True) or _visible
        if full.strip():
            raw_path.write_text(full)
        _quit_claude(socket, session)
        # Also save claude's own structured session jsonl for this turn —
        # the pane scrape is rendered text only.
        _capture_session_delta(env_dir, turn)
        return (0 if completed else 1), full, completed
    finally:
        _tmux(socket, ["kill-server"], check=False, capture=True)
