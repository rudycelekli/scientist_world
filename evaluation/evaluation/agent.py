"""Invoke an agent CLI against a per-run workspace, one turn at a time.

Each call to :func:`invoke` runs exactly one agent turn in a fresh
container subprocess. Session continuity between turns is carried by
the agent CLI's own flags (``--continue``, ``resume --last``,
``--resume latest``). The within-run loop lives in
:mod:`evaluation.loop`.

Scaffolds: ``claude-code`` and ``codex`` (headless, stream-json output),
``kimi`` and ``gemini`` (headless, experimental), and ``claude-tmux``
(experimental: the interactive ``claude`` TUI driven through tmux by
:mod:`evaluation.tmux_driver`).

Container isolation is delegated to ``sciworld.launcher.build_container_cmd``
so the isolation contract (read-only binds, blocked binaries, hidden
trajectory folder) stays in one place.
"""

from __future__ import annotations

import json
import logging
import os
import re
import select
import subprocess
import threading
import time
from pathlib import Path

from evaluation import config as _config
from evaluation.models import Profile

log = logging.getLogger(__name__)

# Per-turn kill switch for the headless scaffolds. There is no cap on the
# length of a turn, only on silence: a turn with no stdout for this long
# is SIGKILL'd and retried. The window has to be longer than the
# exponential backoff an agent CLI applies while a provider is
# rate-limiting it (several minutes between visible events). A profile
# can override it with ``idle_timeout_sec``.
INACTIVITY_TIMEOUT_SEC = 30 * 60
POST_TURN_GRACE_SEC = 30                # after turn.completed, give the CLI this long to exit
MAX_IDLE_RETRIES = 2                    # relaunches of a turn that hit the inactivity kill


# Secret redaction, applied to every line of agent output before it is
# written to disk, since trajectories are committed to the per-run git
# repository. Same patterns as the worker scaffold. The wider list in
# git_publish.py is applied again when a run is published.
_SECRET_PATTERNS = (
    (re.compile(r"ghp_[A-Za-z0-9]{30,}"), "[REDACTED_GITHUB_PAT]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{60,}"), "[REDACTED_GITHUB_PAT]"),
    (re.compile(r"\bhf_[A-Za-z0-9]{30,}"), "[REDACTED_HF_TOKEN]"),
    (re.compile(r"(https?://)[^/\s\"'@]+:[^/\s\"'@]+@"), r"\1[REDACTED]@"),
)


# Lock that serializes OAuth refresh attempts across threads. Without
# it, concurrent `agent.invoke()` callers would all spawn `claude --print`
# at once, each seeing the same stale host token.
_OAUTH_REFRESH_LOCK = threading.Lock()
# Buffer below the actual expiresAt: refresh proactively while there's
# still time to fall back to the existing token if the refresh fails.
_OAUTH_REFRESH_BUFFER_SEC = 30 * 60   # 30 min


def _refresh_claude_oauth_if_stale() -> None:
    """Force-refresh the host's Claude OAuth access_token if near expiry.

    The `claude` CLI rotates its own access_token using the refresh_token
    in `~/.claude/.credentials.json` whenever it makes a request. One
    short `claude --print` request (model: ``claude_oauth_refresh_model``
    in the config) is made **on the host** to trigger that rotation; the
    next `copy_agent_credentials` snapshot then picks up the fresh token.

    Does nothing when there is no credentials file (API-key or gateway
    setups) or the token is still comfortably valid. Best-effort: if the
    refresh subprocess fails, log and proceed; the existing token might
    still work, and the container reports a real 401 if it does not.
    """
    creds = Path.home() / ".claude" / ".credentials.json"
    if not creds.exists():
        return
    try:
        data = json.loads(creds.read_text())
    except (OSError, json.JSONDecodeError):
        return
    expires = (data.get("claudeAiOauth") or {}).get("expiresAt", 0)
    if expires and expires > 1e12:
        expires = expires / 1000.0
    if not expires:
        return
    if expires - time.time() > _OAUTH_REFRESH_BUFFER_SEC:
        return
    with _OAUTH_REFRESH_LOCK:
        # Double-check under lock — another thread may have just refreshed.
        try:
            data = json.loads(creds.read_text())
            expires = (data.get("claudeAiOauth") or {}).get("expiresAt", 0)
            if expires and expires > 1e12:
                expires = expires / 1000.0
            if expires - time.time() > _OAUTH_REFRESH_BUFFER_SEC:
                return
        except (OSError, json.JSONDecodeError):
            return
        log.info("Refreshing host Claude OAuth token (expires in %.1f min)",
                 (expires - time.time()) / 60.0 if expires else 0)
        refresh_model = str(
            _config.load().get("claude_oauth_refresh_model")
            or "claude-haiku-4-5-20251001"
        )
        try:
            # `claude --print` exits cleanly after one round-trip; the
            # round-trip itself triggers OAuth refresh in-place.
            subprocess.run(
                ["claude", "--print", "--model", refresh_model,
                 "--dangerously-skip-permissions", "ok"],
                input="", check=False, capture_output=True, timeout=60,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.warning("OAuth refresh subprocess failed: %s — proceeding with existing token", exc)


def _host_cli_binds(scaffold: str) -> list[tuple[Path, str]]:
    """Bind the host's agent CLI installs over the ones in the agent container.

    The agent image ships whatever CLI versions existed when it was
    built, and the CLIs are released more often than the image is
    rebuilt (codex, for one, refuses model names newer than itself).
    Binding the host install over the container's runs the host's
    version without rebuilding the image. Nothing is bound when the CLI
    is not installed on the host; the container then uses its own copy.

    The container-side paths assume the layout of sciworld's agent image
    (npm packages under ``/usr/lib/node_modules``).

    Returns a list of (host_path, container_path) tuples; sciworld's
    launcher emits ``--bind host:container:ro`` for each.
    """
    out: list[tuple[Path, str]] = []
    if scaffold == "claude-code":
        # The claude installer does not produce a single package
        # directory that could be bound, so the container's own copy is
        # used.
        pass
    elif scaffold == "codex":
        # codex installed through npm: the `codex` on PATH is a symlink
        # to <pkg>/bin/codex.js, so two parents up is the @openai/codex
        # package root.
        import shutil
        host_codex_bin = shutil.which("codex")
        if host_codex_bin:
            real = Path(host_codex_bin).resolve()  # npm: bin/codex.js
            host_codex_pkg = real.parent.parent     # npm: @openai/codex
            if (host_codex_pkg / "package.json").exists():
                # npm layout: override the whole package dir.
                out.append((host_codex_pkg, "/usr/lib/node_modules/@openai/codex"))
            elif real.is_file():
                # Standalone binary release (no package.json): bind the
                # host binary over the container's codex entrypoint.
                out.append((real, "/usr/bin/codex"))
                # The standalone release is not a single file: codex
                # spawns a helper, `codex-code-mode-host`, that it looks
                # up next to the binary it is running, and without it
                # the agent cannot execute any command. In the agent
                # image /usr/bin/codex is a symlink to the npm
                # entrypoint, so the bind above lands on the symlink's
                # target and codex looks for the helper next to that.
                # Bind the helper at both locations so an image where
                # /usr/bin/codex is a regular file works too. (Binding
                # the release's whole bin/ directory instead would hide
                # the npm entrypoint.)
                helper = real.parent / "codex-code-mode-host"
                if helper.exists():
                    out.append((helper, "/usr/bin/codex-code-mode-host"))
                    out.append((
                        helper,
                        "/usr/lib/node_modules/@openai/codex/bin/codex-code-mode-host",
                    ))
    elif scaffold == "gemini":
        # gemini-cli ships as @google/gemini-cli on npm. The `gemini`
        # bin is a symlink to <pkg>/bundle/gemini.js. Resolve to the
        # package root (3 parents up from the symlink target since
        # bundle/gemini.js is two levels under the package root).
        import shutil
        host_gemini_bin = shutil.which("gemini")
        if host_gemini_bin:
            real = Path(host_gemini_bin).resolve()  # <pkg>/bundle/gemini.js
            host_gemini_pkg = real.parent.parent      # @google/gemini-cli
            if (host_gemini_pkg / "package.json").exists():
                out.append((host_gemini_pkg, "/usr/lib/node_modules/@google/gemini-cli"))
    elif scaffold == "kimi":
        # kimi-cli is a uv-managed Python tool. The `kimi` entrypoint's
        # shebang hardcodes the absolute path of the venv's python,
        # which itself symlinks to a uv-managed CPython under
        # ``~/.local/share/uv/python/<version>/``. Bind the whole
        # ``~/.local/share/uv/`` parent at the same absolute host path
        # so both the shebang and its python symlink resolve inside the
        # container without rewriting anything.
        import shutil
        host_kimi_bin = shutil.which("kimi")
        if host_kimi_bin:
            real = Path(host_kimi_bin).resolve()  # <tool_dir>/bin/kimi
            tool_dir = real.parent.parent          # <tool_dir>
            uv_share = tool_dir.parent.parent      # <uv share dir>
            if (tool_dir / "pyvenv.cfg").exists() and uv_share.name == "uv":
                out.append((uv_share, str(uv_share)))
    return out


def redact_secrets(text: str) -> str:
    """Strip known credential patterns from a string, preserving structure.

    Applied to all trajectory + job-log writes. Cheap (4 regex passes).
    Safe on JSON/JSONL: only matched substrings change, surrounding
    quotes and delimiters stay intact.
    """
    if not text:
        return text
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# Messages that mean "this profile's current API key hit a usage limit"
# (same list as the worker scaffold). When one appears in the agent's
# output, rotate to the next key in profile.api_keys rather than retry
# with the same key.
_LIMIT_ERROR_PATTERNS = (
    "weekly/monthly limit",
    "limit exhausted",
    "monthly limit",
    "weekly limit",
    "usage limit reached",
    "you've hit your limit",
)
# Per-profile rotation index, keyed by profile name and shared by all
# runs of that profile in this process. Lock-protected so concurrent
# invokes() do not all advance the index at once.
_API_KEY_INDEX: dict[str, int] = {}
_API_KEY_LOCK = threading.Lock()

# Round-robin counter, separate from rotation. Keys are assigned at
# invocation time so concurrent runs of one profile spread evenly over
# its api_keys instead of all using api_keys[0] and exhausting that
# key's rate limit.
_API_KEY_RR_COUNTER: dict[str, int] = {}
_API_KEY_RR_LOCK = threading.Lock()

# Kimi OAuth uses single-use refresh_tokens. When N concurrent kimi-cli
# processes start near a token-refresh boundary, they all read the same
# refresh_token from ~/.kimi/credentials/kimi-code.json, all POST it to
# auth.kimi.com, and the server accepts only one — the other N-1 get
# "authorization grant is invalid" and exit rc=1. kimi-cli has cross-
# process file locking for the cred-file *write*, but not for the OAuth
# *HTTP* call, so concurrent launches race.
#
# Mitigation: serialize kimi launches with a minimum spacing > the
# refresh HTTP roundtrip (~1-2s in practice; 5s safety margin). After
# the first launch in a window refreshes, the next launch reads the
# fresh access_token from disk and skips refresh entirely (it has the
# full 15-min TTL).
_KIMI_LAUNCH_LOCK = threading.Lock()
_KIMI_LAST_LAUNCH_T = 0.0
_KIMI_MIN_LAUNCH_INTERVAL = 5.0


def _kimi_throttle() -> None:
    """Stagger kimi-cli launches to avoid OAuth refresh-token races."""
    global _KIMI_LAST_LAUNCH_T
    with _KIMI_LAUNCH_LOCK:
        wait = _KIMI_MIN_LAUNCH_INTERVAL - (time.monotonic() - _KIMI_LAST_LAUNCH_T)
        if wait > 0:
            time.sleep(wait)
        _KIMI_LAST_LAUNCH_T = time.monotonic()


def _sync_kimi_credentials_to_host(env_dir: Path) -> None:
    """Persist kimi-cli's rotated credential back to the host.

    Kimi's refresh_token is SINGLE-USE. When kimi-cli refreshes the 15-min
    access token mid-run it POSTs the refresh_token to auth.kimi.com; the
    server invalidates that token and issues a rotated pair, which kimi-cli
    writes to the WORKSPACE copy only (``copy_agent_credentials`` seeds the
    workspace on turn 1 and then deliberately leaves it alone so the
    in-workspace refresh chain survives the run).

    Without a write-back the host file keeps the consumed refresh token,
    and every later run would start from a dead credential and fail with
    HTTP 401.

    Copying the rotated file back keeps ONE continuous refresh chain across
    runs. Only writes when the workspace file is strictly newer (i.e.
    kimi-cli actually rotated it), and only the JSON credential — never the
    ``.lock`` sidecar, which is kimi-cli's own cross-process mutex.

    Serialized under ``_KIMI_LAUNCH_LOCK``. Exactly correct at concurrency
    1; at higher concurrency the last writer wins, which is still strictly
    better than every run seeding a known-dead token.
    """
    import shutil

    src_dir = env_dir / ".kimi" / "credentials"
    dst_dir = Path.home() / ".kimi" / "credentials"
    if not src_dir.is_dir() or not dst_dir.is_dir():
        return
    with _KIMI_LAUNCH_LOCK:
        for child in sorted(src_dir.iterdir()):
            if not child.is_file() or child.suffix != ".json":
                continue
            dst = dst_dir / child.name
            try:
                if dst.exists() and child.stat().st_mtime <= dst.stat().st_mtime:
                    continue  # host is already as fresh — nothing rotated
                shutil.copy2(str(child), str(dst))
                os.chmod(str(dst), 0o600)
                log.info("kimi: synced rotated credential back to host (%s)",
                         child.name)
            except OSError as exc:
                log.warning("kimi: credential write-back failed (%s): %s",
                            child.name, exc)


def _next_round_robin_key(profile: Profile) -> str | None:
    """Pick the next api_key for this profile by round-robin.

    Returns None if the profile has no api_keys list (single-key
    profile — caller falls back to ``profile.env_vars["ANTHROPIC_AUTH_TOKEN"]``
    as before).
    """
    keys = list(profile.api_keys or [])
    if not keys:
        return None
    with _API_KEY_RR_LOCK:
        i = _API_KEY_RR_COUNTER.get(profile.name, 0)
        _API_KEY_RR_COUNTER[profile.name] = i + 1
    return keys[i % len(keys)]


def _detect_limit_error(text: str) -> bool:
    """True if the agent output contains a hard API usage-limit signal."""
    if not text:
        return False
    lowered = text.lower()
    return any(p in lowered for p in _LIMIT_ERROR_PATTERNS)


def _try_rotate_api_key(profile: Profile) -> bool:
    """Rotate ``profile.env_vars["ANTHROPIC_AUTH_TOKEN"]`` to the next
    key in ``profile.api_keys``. Returns True if rotated. The rotation
    index is global per profile name, so all concurrent runs of a
    profile share it.
    """
    api_keys = list(profile.api_keys or [])
    if len(api_keys) <= 1:
        return False
    with _API_KEY_LOCK:
        cur = _API_KEY_INDEX.get(profile.name, 0)
        nxt = (cur + 1) % len(api_keys)
        if nxt == cur:
            return False
        _API_KEY_INDEX[profile.name] = nxt
        new_key = api_keys[nxt]
        # The Profile dataclass exposes env_vars; mutate in place so
        # subsequent _build_env() calls pick up the new key.
        profile.env_vars["ANTHROPIC_AUTH_TOKEN"] = new_key
        log.warning(
            "Profile %s: rotated to API key index %d/%d",
            profile.name, nxt, len(api_keys),
        )
        return True


def _build_agent_argv(profile: Profile, resume: bool) -> list[str]:
    """Build the agent CLI argv. The prompt is fed via stdin, not argv.

    Passing a prompt as a command-line argument is unreliable once it
    grows past argv limits or contains characters the CLI re-parses, so
    it is always streamed on stdin (``--print`` for claude, ``-`` for
    codex)."""
    if profile.scaffold == "claude-code":
        argv = ["claude"]
        if resume:
            argv.append("--continue")
        argv += [
            "--print",
            "--dangerously-skip-permissions",
            "--verbose",
            "--output-format", "stream-json",
            "--model", profile.model,
            # When the agent authenticates with an account login (the
            # credentials are copied into the workspace), the CLI would
            # also load that account's connectors as MCP tools inside the
            # agent container: the evaluated model would get access to
            # the operator's accounts, and the connector schemas cost
            # tens of thousands of tokens per request. With this flag
            # only explicitly configured MCP servers are loaded.
            "--strict-mcp-config",
        ]
        # The compaction window is not passed on the command line: the
        # claude CLI has no such option (``-c`` is its short flag for
        # ``--continue``). It reads CLAUDE_CODE_AUTO_COMPACT_WINDOW, set
        # by _build_env.
        return argv

    if profile.scaffold == "claude-tmux":
        # Interactive TUI variant of claude-code, driven via tmux when
        # ``--print`` is unavailable. No --print / --output-format /
        # --verbose: those are headless-only. Context across turns comes
        # from --continue (session persisted in the workspace's
        # ~/.claude), exactly as the headless path. Orchestrated by
        # evaluation.tmux_driver.run_tmux_turn.
        argv = ["claude"]
        if resume:
            argv.append("--continue")
        argv += [
            "--dangerously-skip-permissions",
            "--model", profile.model,
            # Same as the headless path: never load the account's
            # connectors into an evaluated agent.
            "--strict-mcp-config",
        ]
        # As in the headless path, the compaction window is passed
        # through CLAUDE_CODE_AUTO_COMPACT_WINDOW, never as ``-c ...``:
        # the TUI would read ``-c`` as "continue a conversation" and exit
        # when there is none.
        return argv

    if profile.scaffold == "codex":
        argv = ["codex", "exec"]
        if resume:
            argv += ["resume", "--last"]
        argv += [
            "--experimental-json",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--model", profile.model,
        ]
        if profile.effort:
            argv += ["-c", f"reasoning_effort={profile.effort}"]
        if profile.compact_token_limit:
            argv += ["-c", f"model_auto_compact_token_limit={profile.compact_token_limit}"]
        # Read prompt from stdin.
        argv.append("-")
        return argv

    if profile.scaffold == "kimi":
        # kimi-cli. Resolve `kimi` from PATH; the tool dir is
        # bind-mounted at the same absolute host path so the
        # entrypoint's hardcoded python shebang resolves inside the
        # container. Stream JSON output, --yolo to auto-approve tool
        # calls (the sandbox is the container, not kimi), --thinking to
        # use the model's reasoning mode.
        import shutil
        host_bin = shutil.which("kimi") or "kimi"
        argv = [str(Path(host_bin).resolve())]
        if resume:
            argv.append("--continue")
        argv += [
            "--print",
            "--input-format", "text",
            "--output-format", "stream-json",
            "--yolo",
            "--thinking",
            "-m", profile.model,
        ]
        return argv

    if profile.scaffold == "gemini":
        # Invoke the bundled gemini.js directly via node — avoids needing
        # to bind the host's symlink (whose relative target wouldn't
        # resolve inside the container) and bypasses any PATH munging.
        argv = [
            "node",
            "/usr/lib/node_modules/@google/gemini-cli/bundle/gemini.js",
            "-p", "-",                  # headless mode, prompt from stdin
            "-o", "stream-json",
            "--skip-trust",              # skip workspace-trust prompt (no TTY)
            "--yolo",                    # auto-approve all tools (no TTY)
            "-m", profile.model,
        ]
        if resume:
            argv += ["--resume", "latest"]
        return argv

    raise ValueError(f"Unknown scaffold: {profile.scaffold!r}")


def _build_env(
    profile: Profile, nexus_project_id: str, cpu_threads: int,
) -> dict[str, str]:
    """Compose the subprocess environment — host env + profile overrides."""
    env = dict(os.environ)

    # Strip anything that would confuse a fresh session inside the container.
    for key in list(env):
        if key == "CLAUDECODE" or key.startswith("CLAUDE_CODE_"):
            del env[key]

    # The harness authenticates agents with the credential files copied
    # from the host's ~/.claude, ~/.codex, ... into the workspace, not
    # with API keys inherited from the operator's shell. Keys for a
    # profile are passed explicitly through its ``env_vars``.
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)

    # Nexus endpoint and project id of this run, exported into the agent
    # container. The harness submits every job itself; the agent
    # protocol does not use these.
    nexus_url = (_config.load().get("nexus") or {}).get("url") or ""
    if nexus_url:
        env["APPTAINERENV_NEXUS_URL"] = nexus_url
    env["APPTAINERENV_NEXUS_PROJECT_ID"] = nexus_project_id

    # BLAS / threading caps. A CPU cgroup is often not available where
    # the agent container runs, so these environment variables bound the
    # thread fan-out of torch/numpy/scipy/numexpr instead.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[f"APPTAINERENV_{var}"] = str(cpu_threads)

    # Profile env overrides — pass into the container with APPTAINERENV_ prefix
    # so --cleanenv still lets them through.
    for k, v in profile.env_vars.items():
        env[f"APPTAINERENV_{k}"] = v

    # Effort (claude-code / claude-tmux). Codex's effort is passed via -c.
    if profile.scaffold in ("claude-code", "claude-tmux") and profile.effort:
        env["APPTAINERENV_CLAUDE_CODE_EFFORT_LEVEL"] = str(profile.effort)

    # Auto-compact window (claude-code / claude-tmux). This is the knob the
    # claude binary actually reads — its own text: "To enforce it, set
    # CLAUDE_CODE_AUTO_COMPACT_WINDOW=". Without it the CLI uses the window
    # it assumes for the model name it was handed, which for a proxied
    # provider behind an "opus" alias can sit far above that provider's
    # real cap, so the session grows until the upstream rejects it as too
    # long. Codex takes its limit via -c instead.
    if profile.scaffold in ("claude-code", "claude-tmux") and profile.compact_token_limit:
        env["APPTAINERENV_CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(profile.compact_token_limit)

    # Round-robin api_key assignment. For profiles with several
    # api_keys, override the static ANTHROPIC_AUTH_TOKEN with the next
    # key so concurrent runs spread evenly over the keys. The on-error
    # rotation in invoke() remains as a fallback when a key still hits
    # its limit. Profiles without api_keys keep the env_vars value.
    rr_key = _next_round_robin_key(profile)
    if rr_key:
        env["APPTAINERENV_ANTHROPIC_AUTH_TOKEN"] = rr_key

    # Retry / timeout tuning for brief upstream overloads. The claude
    # CLI's own defaults are exhausted quickly when an endpoint is
    # overloaded for a few minutes; the values come from the evaluation
    # config (claude_code_max_retries, api_timeout_ms). The host's
    # ~/.claude/settings.json does not reach the container (the
    # environment is cleaned and the file is not bound), so the values
    # are passed explicitly.
    cfg = _config.load()
    max_retries = str(cfg.get("claude_code_max_retries") or 30)
    api_timeout = str(cfg.get("api_timeout_ms") or 3000000)
    env["APPTAINERENV_CLAUDE_CODE_MAX_RETRIES"] = max_retries
    env["APPTAINERENV_API_TIMEOUT_MS"] = api_timeout

    # Raise the claude CLI's per-response output cap. With the CLI
    # default, a model that produces one very long reasoning response
    # makes the CLI abort the whole turn ("response exceeded the ...
    # output token maximum"). A higher ceiling is harmless for models
    # that cap lower on their own.
    max_output = str(cfg.get("claude_code_max_output_tokens") or 64000)
    env["APPTAINERENV_CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = max_output

    return env


def _stream_output(
    proc: subprocess.Popen,
    raw_path: Path | None,
    idle_timeout_sec: int | None = None,
) -> tuple[str, bool, bool]:
    """Drain proc.stdout to memory + optional file with activity timeouts.

    Returns ``(stdout, turn_completed_seen, idle_killed)``.

    ``turn_completed_seen`` flips True the first time we see a line
    containing ``"turn.completed"`` (codex's end-of-turn event). At that
    point all the agent's work for this turn (action.yaml, file edits,
    trajectory) is already on disk, so the inactivity timeout switches
    to a short grace period: codex can hang in post-turn teardown, and
    there is nothing more to wait for. After the grace period the
    process is killed without flagging the turn as failed.

    ``idle_killed`` flips True only when the full inactivity timeout
    passes *without* turn.completed having been seen: a real hang that
    the caller retries.

    All written lines are passed through :func:`redact_secrets` so
    GitHub PATs / HF tokens / URL userinfo never reach disk.
    """
    idle_timeout = idle_timeout_sec or INACTIVITY_TIMEOUT_SEC
    stdout_fd = proc.stdout.fileno()
    turn_start = time.monotonic()
    buf = b""
    chunks: list[str] = []
    raw_fh = open(raw_path, "w") if raw_path else None
    turn_completed_seen = False
    idle_killed = False
    grace_deadline: float | None = None

    try:
        while True:
            # Active timeout depends on phase:
            #   - before turn.completed: full INACTIVITY_TIMEOUT_SEC (real work may still be coming)
            #   - after turn.completed: short POST_TURN_GRACE_SEC (work is done; just waiting for exit)
            if grace_deadline is not None:
                wait_for = max(0.0, grace_deadline - time.monotonic())
            else:
                wait_for = idle_timeout

            ready, _, _ = select.select([stdout_fd], [], [], wait_for)
            if not ready:
                if turn_completed_seen:
                    log.info(
                        "post-turn grace (%ds) elapsed; killing codex — turn output already saved",
                        POST_TURN_GRACE_SEC,
                    )
                else:
                    log.warning("inactivity timeout (%ds), killing", idle_timeout)
                    idle_killed = True
                proc.kill()
                break

            chunk = os.read(stdout_fd, 65536)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line_bytes, buf = buf.split(b"\n", 1)
                line = line_bytes.decode("utf-8", errors="replace") + "\n"
                redacted = redact_secrets(line)
                chunks.append(redacted)
                if raw_fh:
                    raw_fh.write(redacted)
                    raw_fh.flush()
                if not turn_completed_seen and '"turn.completed"' in redacted:
                    turn_completed_seen = True
                    grace_deadline = time.monotonic() + POST_TURN_GRACE_SEC
        if buf:
            line = buf.decode("utf-8", errors="replace")
            redacted = redact_secrets(line)
            chunks.append(redacted)
            if raw_fh:
                raw_fh.write(redacted)
                raw_fh.flush()
    finally:
        if raw_fh:
            raw_fh.close()

    return "".join(chunks), turn_completed_seen, idle_killed


# How claude records the way a session was started. ``--continue`` only
# resumes sessions of its own kind: the interactive TUI picks up "cli"
# sessions, headless ``--print`` picks up "sdk-cli" ones.
_CLAUDE_ENTRYPOINT = {"claude-code": "sdk-cli", "claude-tmux": "cli"}


def has_conversation(env_dir: Path, scaffold: str) -> bool:
    """True if this workspace holds a conversation the agent can continue.

    The CLI stores sessions under ``$HOME/.claude/projects/<cwd-key>/``
    as one jsonl per session; the workspace is ``$HOME`` inside the
    container. A file there is not enough: ``--continue`` only resumes
    sessions started the same way (interactive or headless), and with
    nothing to resume claude prints "No conversation found to continue"
    and exits, so every later turn would fail without the agent ever
    seeing its task. A session counts only if it was started the way
    this scaffold starts claude and the model actually answered in it.

    Scaffolds other than claude keep their own session state and are
    assumed continuable.
    """
    entry = _CLAUDE_ENTRYPOINT.get(scaffold)
    if entry is None:
        return True
    projects = env_dir / ".claude" / "projects"
    if not projects.is_dir():
        return False
    kind = f'"entrypoint":"{entry}"'
    answered = '"type":"assistant"'
    for session in projects.glob("*/*.jsonl"):
        right_kind = has_answer = False
        try:
            with open(session, errors="replace") as fh:
                for line in fh:
                    right_kind = right_kind or kind in line
                    has_answer = has_answer or answered in line
                    if right_kind and has_answer:
                        return True
        except OSError:
            continue
    return False


def invoke(
    env_dir: Path,
    profile: Profile,
    prompt: str,
    *,
    turn: int,
    resume: bool,
    nexus_project_id: str,
    test_phase: bool = False,
    test_phase_unlock_method: bool = False,
    extra_binds: list[tuple[Path, str]] | None = None,
) -> tuple[int, str, str]:
    """Run exactly one agent turn.

    Returns ``(returncode, stdout, stderr)``. Raw stream-json output is
    streamed to ``<env_dir>/trajectory/turn_NNNN_raw.jsonl`` as the agent
    emits it.

    ``test_phase`` and ``test_phase_unlock_method`` are only used by the
    disabled in-run test phase (``runner._run_test_phase``).
    ``test_phase=True`` adds read-only bind mounts over ``method/``,
    ``eval/test/``, and ``scripts/evaluate_test.sh`` so the held-out
    scoring cannot be tampered with by any infrastructure fix.
    ``test_phase_unlock_method=True`` is the final-tier recovery mode:
    ``method/`` becomes writable so the agent can rename or adapt files
    to satisfy the test scorer's expected interface. Has effect only
    when ``test_phase`` is also True.
    """
    cfg = _config.load()
    sif = cfg.get("agent_sif") or ""
    if not sif:
        raise _config.ConfigError(
            "No agent container configured. Set agent_sif in "
            f"{_config.CONFIG_PATH} to the agent image built by the "
            "sciworld CLI."
        )

    # Lazy import — sciworld is an optional dep at import time, not at runtime
    # of config.py (e.g. dashboard-only deployments can skip it).
    from sciworld.launcher import build_container_cmd, copy_agent_credentials  # type: ignore

    # Refresh the host's Claude OAuth token if it is near or past expiry
    # before it is copied into the workspace. Access tokens live for a
    # few hours, so in a long batch every turn started after the expiry
    # would otherwise copy a dead token and fail with HTTP 401.
    if profile.scaffold in ("claude-code", "claude-tmux"):
        _refresh_claude_oauth_if_stale()

    # The interactive tmux scaffold authenticates and binds exactly like
    # claude-code (same `claude` CLI, same OAuth creds); only the turn
    # *execution* differs. Map it to "claude-code" for credential copy
    # and host-CLI binds so sciworld's launcher stays scaffold-agnostic.
    cred_scaffold = "claude-code" if profile.scaffold == "claude-tmux" else profile.scaffold

    # Copy host credentials into the workspace so the container sees them
    # at $HOME/.claude/ or $HOME/.codex/. Container uses --containall so
    # HOME becomes the bind-mounted workspace.
    for note in copy_agent_credentials(cred_scaffold, env_dir):
        log.info(note)

    # ``--continue`` needs a conversation to continue. A turn that dies
    # before claude accepts its first message (startup confirmation,
    # auth failure) persists nothing, yet the run still counts that
    # turn, so a later resume would ask claude to continue a
    # conversation that does not exist and claude would exit at once.
    # Start a new session in that case: the workspace and trajectory
    # are intact, so nothing is lost.
    if resume and not has_conversation(env_dir, profile.scaffold):
        log.warning(
            "Turn %d: resume requested but the workspace holds no claude "
            "conversation to continue (earlier turns died before the chat "
            "box) — starting a new session over the existing workspace", turn,
        )
        resume = False

    agent_argv = _build_agent_argv(profile, resume=resume)
    # Empty directory bound over trajectory/ so the agent cannot read
    # its own recorded transcript.
    empty_traj_overlay = _config.data_dir() / "containers" / "empty_trajectory"
    empty_traj_overlay.mkdir(parents=True, exist_ok=True)

    # Run the host's agent CLI install instead of the one in the agent
    # image when it is available (see _host_cli_binds).
    binds = list(extra_binds) if extra_binds else []
    binds.extend(_host_cli_binds(cred_scaffold))

    # Interactive scaffold only: bind a pinned `claude` binary over the
    # image's. The tmux driver depends on the start-up screens of the
    # CLI version it was written against, so that version is pinned
    # through ``claude_tmux_binary``. Headless claude-code is untouched.
    if profile.scaffold == "claude-tmux":
        pinned = cfg.get("claude_tmux_binary") or ""
        if pinned and Path(pinned).exists():
            binds.append((Path(pinned), "/usr/local/bin/claude"))
        else:
            log.warning("claude-tmux: claude_tmux_binary missing (%r); "
                        "container will use its bundled claude", pinned)

    container_cmd = build_container_cmd(
        "apptainer",
        sif,
        env_dir,
        cmd=agent_argv,
        empty_traj_overlay=empty_traj_overlay,
        test_phase=test_phase,
        test_phase_unlock_method=test_phase_unlock_method,
        extra_binds=binds,
    )

    # Apply per-container resource caps (memory cgroup hard cap + nice
    # priority). Thread caps are environment variables set by _build_env.
    mem_limit = str(cfg.get("agent_memory") or "8G")
    nice_val = str(cfg.get("agent_nice") or 19)
    cpu_threads = int(cfg.get("agent_cpu_threads") or 2)
    exec_idx = container_cmd.index("exec")
    container_cmd = (
        ["nice", "-n", nice_val]
        + container_cmd[:exec_idx + 1]
        + ["--memory", mem_limit]
        + container_cmd[exec_idx + 1:]
    )

    traj_dir = env_dir / "trajectory"
    traj_dir.mkdir(parents=True, exist_ok=True)
    raw_path = traj_dir / f"turn_{turn:04d}_raw.jsonl"

    env = _build_env(profile, nexus_project_id, cpu_threads)

    # Interactive (tmux) scaffold: drive the claude TUI instead of the
    # headless --print loop below. The agent still signals via
    # action.yaml, so the caller is unchanged; we return (rc, captured,
    # "") to match the headless ``(rc, stdout, stderr)`` contract.
    if profile.scaffold == "claude-tmux":
        from evaluation.tmux_driver import run_tmux_turn
        rc, captured, _completed = run_tmux_turn(
            container_cmd, prompt, raw_path,
            env=env, env_dir=env_dir, turn=turn,
            action_path=env_dir / "action.yaml",
        )
        return rc, captured, ""

    # Idle-retry loop. An agent CLI can hang without any output (usually
    # a transient API or network problem): kill it after the inactivity
    # window and relaunch the turn up to MAX_IDLE_RETRIES times; only
    # then is the non-zero exit reported to the caller.
    idle_kill_count = 0
    rotated_keys = 0
    max_key_rotations = max(0, len(profile.api_keys) - 1)  # try every other key once
    while True:
        if profile.scaffold == "kimi":
            _kimi_throttle()
        log.info(
            "Turn %d: launching %s %s%s",
            turn, profile.scaffold, profile.model,
            f" (idle-retry {idle_kill_count}/{MAX_IDLE_RETRIES})" if idle_kill_count else "",
        )
        proc = subprocess.Popen(
            container_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(env_dir),
            env=env,
            # Put each agent in its own process group so the dashboard can
            # kill the whole Apptainer process tree with one
            # ``os.killpg`` on shutdown. Apptainer double-forks, so a
            # signal to the harness alone would leave the container
            # processes running as orphans.
            start_new_session=True,
        )
        # Feed the prompt over stdin and close it; the CLI does not start
        # processing until EOF.
        try:
            proc.stdin.write(prompt.encode("utf-8"))
            proc.stdin.close()
        except BrokenPipeError:
            pass

        try:
            stdout, turn_completed_seen, idle_killed = _stream_output(
                proc, raw_path,
                idle_timeout_sec=profile.idle_timeout_sec,
            )
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            stderr = redact_secrets(stderr)
            proc.wait(timeout=60)
        except Exception:
            proc.kill()
            proc.wait()
            raise

        # If the turn already wrote ``turn.completed``, the agent's work is
        # on disk and the SIGKILL we sent during post-turn-grace is benign.
        # Normalize the returncode so the caller doesn't treat the kill as a
        # failure.
        rc = proc.returncode or 0
        if turn_completed_seen and rc != 0:
            rc = 0

        # kimi-cli may have rotated its single-use refresh token during this
        # turn, writing the new pair into the workspace only. Persist it back
        # to the host HERE — before any retry/``continue`` below — so both the
        # next turn's re-seed and the next run start from a live credential
        # rather than the consumed one.
        if profile.scaffold == "kimi":
            _sync_kimi_credentials_to_host(env_dir)

        # Real hang (no end-of-turn event seen, full inactivity timeout
        # fired): relaunch the turn rather than fail the run. The
        # relaunch uses the same argv, so with ``resume`` set the agent
        # continues its session.
        if idle_killed and not turn_completed_seen:
            idle_kill_count += 1
            if idle_kill_count <= MAX_IDLE_RETRIES:
                log.warning(
                    "Turn %d: agent produced no output for the inactivity "
                    "window; retrying (%d/%d)",
                    turn, idle_kill_count, MAX_IDLE_RETRIES,
                )
                continue
            log.warning(
                "Turn %d: agent hung %d times in a row; giving up",
                turn, idle_kill_count,
            )

        # Usage-limit detection: scan stdout+stderr for the messages in
        # _LIMIT_ERROR_PATTERNS. On a match, and if the profile has more
        # than one api_key, rotate to the next key and retry the turn.
        if rc != 0 and _detect_limit_error(stdout + "\n" + stderr):
            if rotated_keys < max_key_rotations and _try_rotate_api_key(profile):
                rotated_keys += 1
                # The new key needs to land in env for the retry — rebuild env.
                env = _build_env(profile, nexus_project_id, cpu_threads)
                log.warning(
                    "Turn %d: hit API usage limit, rotated key (%d/%d); retrying",
                    turn, rotated_keys, max_key_rotations,
                )
                # Reset idle counter — a fresh key is a fresh attempt.
                idle_kill_count = 0
                continue

        return rc, stdout, stderr
