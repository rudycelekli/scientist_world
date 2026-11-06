"""CodexWorker — drives a builder agent via the Codex CLI."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from infrastructure.worker.worker import CONTAINER_WORKSPACE, Worker

logger = logging.getLogger(__name__)

# Context size (tokens) at which Codex compacts the conversation, unless the
# profile sets ``compact_token_limit``.
DEFAULT_COMPACT_TOKEN_LIMIT = 120000

CONFIG_DEFAULTS = {
    "max_turns": 200,     # accepted but unused: the Codex command takes no turn limit
    "effort": None,       # model_reasoning_effort (low/medium/high)
    "compact_token_limit": None,  # model_auto_compact_token_limit; None = DEFAULT_COMPACT_TOKEN_LIMIT
    "idle_timeout": 1200,  # kill process if no stdout for this many seconds (20 min)
    "idle_retries": 2,     # retries when a turn is killed for inactivity
    "env_vars": {},        # extra env vars passed to subprocess
    "action_retries": 2,   # retries when action.yaml is invalid after a turn
    "api_retries": 8,      # retries on transient API errors
    "container_sif": None, # path to reproducer SIF; None = run without container
}


# ── Codex output parsing ────────────────────────────────────────────────────

def _parse_codex_output(stdout: str) -> dict:
    """Parse Codex ``--experimental-json`` JSONL output.

    Returns a dict with keys: text, session_id, stats, errors.
    """
    text = ""
    session_id = None
    errors: list[str] = []
    stats = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "num_turns": 0,
        "tool_uses": {},
    }

    for line in stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = data.get("type", "")

        if msg_type == "session.created":
            session_id = data.get("session_id")

        elif msg_type == "thread.started":
            session_id = session_id or data.get("thread_id")

        elif msg_type == "error":
            errors.append(data.get("message", str(data)))

        elif msg_type == "session_meta":
            payload = data.get("payload", {})
            if not session_id:
                session_id = payload.get("id")

        elif msg_type == "event_msg":
            payload = data.get("payload", {})
            event = payload.get("type", "")

            if event == "agent_message":
                msg = payload.get("message", "")
                if msg:
                    if text:
                        text += "\n"
                    text += msg

            elif event == "token_count":
                info = payload.get("info") or {}
                total = info.get("total_token_usage", {})
                if total:
                    stats["input_tokens"] = total.get("input_tokens", 0)
                    stats["output_tokens"] = total.get("output_tokens", 0)
                    stats["reasoning_tokens"] = total.get("reasoning_output_tokens", 0)

            elif event in ("error", "stream_error"):
                errors.append(payload.get("message", str(payload)))

            elif event == "exec_command":
                stats["tool_uses"]["shell"] = stats["tool_uses"].get("shell", 0) + 1

        elif msg_type == "response_item":
            payload = data.get("payload", {})
            item_type = payload.get("type", "")

            if item_type == "message" and payload.get("role") == "assistant":
                for block in payload.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        msg = block.get("text", "")
                        if msg:
                            if text:
                                text += "\n"
                            text += msg

            elif item_type == "function_call":
                name = payload.get("name", "unknown")
                stats["tool_uses"][name] = stats["tool_uses"].get(name, 0) + 1
                stats["num_turns"] += 1

        elif msg_type == "item.completed":
            item = data.get("item", {})
            item_type = item.get("type") or item.get("item_type", "")

            if item_type in ("assistant_message", "agent_message"):
                msg = item.get("text", "")
                if msg:
                    if text:
                        text += "\n"
                    text += msg

            elif item_type == "command_execution":
                stats["tool_uses"]["shell"] = stats["tool_uses"].get("shell", 0) + 1
                stats["num_turns"] += 1

            elif item_type == "function_call":
                name = item.get("name", "unknown")
                stats["tool_uses"][name] = stats["tool_uses"].get(name, 0) + 1
                stats["num_turns"] += 1

            elif item_type == "file_change":
                stats["tool_uses"]["file_edit"] = stats["tool_uses"].get("file_edit", 0) + 1
                stats["num_turns"] += 1

        elif msg_type == "usage.updated":
            usage = data.get("usage", {})
            stats["input_tokens"] = usage.get("input_tokens", stats["input_tokens"])
            stats["output_tokens"] = usage.get("output_tokens", stats["output_tokens"])
            stats["reasoning_tokens"] = usage.get("reasoning_tokens", stats["reasoning_tokens"])

        elif msg_type == "turn.completed":
            usage = data.get("usage", {})
            stats["input_tokens"] = usage.get("input_tokens", stats["input_tokens"])
            stats["output_tokens"] = usage.get("output_tokens", stats["output_tokens"])
            stats["cached_input_tokens"] = usage.get("cached_input_tokens", 0)

    if not text and not session_id and not errors:
        text = stdout

    return {
        "text": text,
        "session_id": session_id,
        "stats": stats,
        "errors": errors,
    }


# ── Worker implementation ────────────────────────────────────────────────────

class CodexWorker(Worker):
    """Worker implementation backed by the Codex CLI subprocess."""

    # Patterns that indicate the context window is exhausted.
    _CONTEXT_EXCEEDED_PATTERNS = (
        "exceeds the context window",
        "context window exceeded",
        "above the token limit after automatic summarization",
    )

    # ── abstract hook implementations ──────────────────────────────────────

    @staticmethod
    def _config_defaults() -> dict[str, Any]:
        return CONFIG_DEFAULTS

    @staticmethod
    def _initial_cumulative_stats() -> dict:
        return {
            "cost_usd": None,          # not reported by codex
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "cache_creation_tokens": None,  # not reported by codex
            "reasoning_tokens": 0,
            "total_input_tokens": 0,
            "num_turns": 0,
            "duration_api_ms": None,    # not reported by codex
            "tool_uses": {},
        }

    def _accumulate_stats(self, turn_stats: dict) -> None:
        """Merge stats. Codex reports cumulative token counts (take max)."""
        for key in ("input_tokens", "output_tokens", "cached_input_tokens",
                     "reasoning_tokens"):
            self._cumulative_stats[key] = max(
                self._cumulative_stats[key], turn_stats.get(key, 0)
            )
        # Codex input_tokens already includes cached, so total = input_tokens
        self._cumulative_stats["total_input_tokens"] = self._cumulative_stats["input_tokens"]
        self._cumulative_stats["num_turns"] += turn_stats.get("num_turns", 0)
        for tool, count in turn_stats.get("tool_uses", {}).items():
            self._cumulative_stats["tool_uses"][tool] = (
                self._cumulative_stats["tool_uses"].get(tool, 0) + count
            )

    # ── workspace setup ─────────────────────────────────────────────────

    def _setup_workspace(self) -> None:
        ws = Path(self._workspace_path)
        self._link_protocol_as_agents_md(ws)
        self._copy_credentials(ws)

    @staticmethod
    def _link_protocol_as_agents_md(ws: Path) -> None:
        protocol_src = ws / "protocol.md"
        agents_md = ws / "AGENTS.md"
        if protocol_src.exists() and not agents_md.exists():
            # Copy instead of symlink: codex's sandbox maps protocol.md
            # to /dev/null, so a symlink would be unreadable.
            shutil.copy2(str(protocol_src), str(agents_md))

    def _copy_credentials(self, ws: Path) -> None:
        if not self.config.get("container_sif"):
            return
        host_auth = Path.home() / ".codex" / "auth.json"
        if host_auth.exists():
            codex_dir = ws / ".codex"
            codex_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(host_auth), str(codex_dir / "auth.json"))

    # ── container / env ───────────────────────────────────────────────────

    def _add_scaffold_container_binds(self, cmd: list[str]) -> None:
        # Bind-mount Node.js prefix (node binary + codex package)
        node_prefix = self._find_node_prefix()
        if node_prefix:
            cmd.extend(["--bind", f"{node_prefix}:{node_prefix}"])
            cmd.extend(["--env", f"PATH={node_prefix}/bin:/usr/local/bin:/usr/bin:/bin"])
            cmd.extend(["--env", "TMPDIR=/tmp"])

        # Bind host CA certificates for HTTPS
        for ca_path in (
            "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
            "/etc/ssl/certs/ca-certificates.crt",
        ):
            if os.path.exists(ca_path):
                cmd.extend(["--bind", f"{ca_path}:{ca_path}:ro"])
                break

    def _configure_env(self, env: dict[str, str]) -> None:
        if self.config.get("container_sif"):
            env.pop("OPENAI_API_KEY", None)
            env.pop("ANTHROPIC_API_KEY", None)
        elif self._workspace_path:
            # Without a container, codex uses a 64 MB FUSE overlay at /.
            # Redirect pip/uv installs to the workspace so they don't
            # fill that tiny filesystem.
            pkg_dir = os.path.join(self._workspace_path, ".python_packages")
            env.setdefault("PIP_TARGET", pkg_dir)
            env.setdefault("UV_PYTHON_PREFERENCE", "system")
            # Ensure the installed packages are importable
            existing = env.get("PYTHONPATH", "")
            if pkg_dir not in existing:
                env["PYTHONPATH"] = f"{pkg_dir}:{existing}" if existing else pkg_dir

    @staticmethod
    def _find_node_prefix() -> str | None:
        """Discover the Node.js installation prefix from the ``codex`` binary."""
        codex_path = shutil.which("codex")
        if not codex_path:
            return None
        resolved = Path(codex_path).resolve()
        for parent in resolved.parents:
            if (parent / "bin" / "node").exists():
                return str(parent)
        return None

    # ── CLI invocation ────────────────────────────────────────────────────

    def _build_command(self) -> list[str]:
        sif = self.config.get("container_sif")
        codex_bin = shutil.which("codex") or "codex"

        codex_cmd = [
            codex_bin, "exec",
            "--experimental-json",
            "--model", self.model,
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "-C", (CONTAINER_WORKSPACE if self.config.get("container_sif") else self._workspace_path) or ".",
        ]
        compact_limit = self.config.get("compact_token_limit") or DEFAULT_COMPACT_TOKEN_LIMIT
        codex_cmd.extend(["-c", f"model_auto_compact_token_limit={compact_limit}"])

        effort = self.config.get("effort")
        if effort:
            codex_cmd.extend(["-c", f'model_reasoning_effort="{effort}"'])

        if self._session_id:
            codex_cmd.extend(["resume", self._session_id])

        codex_cmd.append("-")

        if not sif:
            return codex_cmd

        cmd = self._build_container_cmd(sif)
        cmd.extend(codex_cmd)
        return cmd

    @staticmethod
    def _is_context_exceeded(errors: list[str], stderr: str) -> bool:
        """Check if the context window has been exhausted."""
        combined = " ".join(errors).lower() + " " + stderr.lower()
        return any(pat in combined for pat in CodexWorker._CONTEXT_EXCEEDED_PATTERNS)

    def _run_with_idle_detection(
        self,
        cmd: list[str],
        env: dict[str, str],
        prompt: str,
        idle_timeout: float,
    ) -> tuple[str, str, int, bool]:
        """Run the Codex CLI and kill it if stdout goes silent.

        Allows arbitrarily long turns as long as the process keeps producing
        output. Only kills when no new stdout line arrives for
        ``idle_timeout`` seconds.

        Returns (stdout, stderr, returncode, was_idle_killed).
        """
        import select

        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
                cwd=self._workspace_path, env=env,
                start_new_session=True,
            )
            self._active_proc = proc
        except Exception as exc:
            return "", str(exc), -99, False

        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
        except Exception:
            pass

        stdout_lines: list[str] = []
        last_activity = time.monotonic()
        was_idle_killed = False

        try:
            while True:
                ready, _, _ = select.select([proc.stdout], [], [], 10.0)
                if ready:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    stdout_lines.append(line)
                    last_activity = time.monotonic()
                else:
                    if proc.poll() is not None:
                        break
                    if idle_timeout and (time.monotonic() - last_activity) > idle_timeout:
                        logger.warning(
                            "Codex process idle for %ds (pid=%d), killing process group",
                            idle_timeout, proc.pid,
                        )
                        try:
                            os.killpg(proc.pid, 9)
                        except OSError:
                            proc.kill()
                        was_idle_killed = True
                        break
        except Exception as exc:
            logger.warning("Error reading Codex stdout: %s", exc)

        try:
            remaining_out, stderr = proc.communicate(timeout=30)
            if remaining_out:
                stdout_lines.append(remaining_out)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, 9)
            except OSError:
                proc.kill()
            remaining_out, stderr = proc.communicate()
            if remaining_out:
                stdout_lines.append(remaining_out)
            was_idle_killed = True
        except Exception:
            stderr = ""

        self._active_proc = None
        stdout = "".join(stdout_lines)
        returncode = proc.returncode if proc.returncode is not None else -1
        return stdout, stderr, returncode, was_idle_killed

    def _invoke(self, prompt: str) -> str:
        if self._workspace_path:
            self._copy_credentials(Path(self._workspace_path))

        self._turn_count += 1
        turn_num = self._turn_count

        max_api_retries = self.config.get("api_retries", 8)
        max_idle_retries = self.config.get("idle_retries", 2)
        idle_timeout = self.config.get("idle_timeout", 1200)
        base_delay = 30
        max_total_retry_sec = self.config.get("max_total_retry_sec", 900)  # 15 min cap
        retry_start = time.monotonic()
        idle_kill_count = 0
        api_retry_count = 0

        while True:
            cmd = self._build_command()
            env = self._build_env()

            total_attempts = idle_kill_count + api_retry_count
            if total_attempts == 0:
                logger.info("Turn %d: invoking Codex CLI: %s", turn_num, " ".join(cmd[:6]))
            else:
                logger.info("Turn %d: retry (api=%d, idle_kill=%d)", turn_num, api_retry_count, idle_kill_count)

            start = time.monotonic()

            stdout, stderr, returncode, was_idle_killed = self._run_with_idle_detection(
                cmd, env, prompt, idle_timeout,
            )

            if was_idle_killed:
                idle_kill_count += 1
                if idle_kill_count <= max_idle_retries:
                    logger.warning(
                        "Turn %d: no output for %ds, killed and retrying (%d/%d)",
                        turn_num, idle_timeout, idle_kill_count, max_idle_retries,
                    )
                    continue
                logger.warning("Turn %d: idle-killed %d times, giving up", turn_num, idle_kill_count)

            if returncode == -99:
                logger.error("Turn %d: CLI failed: %s", turn_num, stderr)
                self._status = "error"
                self._error_message = f"CLI failed (rc=-99): {stderr[:500]}"
                self._trajectory.append({
                    "turn": turn_num, "error": stderr,
                    "duration_s": round(time.monotonic() - start, 2),
                })
                return f"Error invoking Codex: {stderr}"

            elapsed = round(time.monotonic() - start, 2)
            parsed = _parse_codex_output(stdout)

            if returncode != 0 and parsed["errors"] and self._is_retryable_error(parsed["errors"], stderr):
                api_retry_count += 1
                if api_retry_count <= max_api_retries:
                    total_elapsed = time.monotonic() - retry_start
                    if total_elapsed >= max_total_retry_sec:
                        logger.warning(
                            "Turn %d: total retry time %.0fs exceeds cap %ds, giving up",
                            turn_num, total_elapsed, max_total_retry_sec,
                        )
                        break
                    delay = base_delay * api_retry_count + random.uniform(0, 10)
                    delay = min(delay, max_total_retry_sec - total_elapsed)
                    logger.warning(
                        "Turn %d: transient error (%d/%d), retrying in %ds: %s",
                        turn_num, api_retry_count, max_api_retries, delay, parsed["errors"][0][:200],
                    )
                    if self._interruptible_sleep(delay):
                        logger.info("Turn %d: retry sleep interrupted by cancel", turn_num)
                        break
                    continue
                else:
                    logger.error(
                        "Turn %d: transient error persists after %d retries: %s",
                        turn_num, max_api_retries, parsed["errors"][0][:200],
                    )

            # API key rotation: if hard limit hit and backup keys exist, swap and retry
            if returncode != 0 and parsed["errors"] and self._is_limit_error(parsed["errors"], stderr):
                if self._try_rotate_api_key():
                    logger.warning(
                        "Turn %d: hard API limit, rotated key — retrying",
                        turn_num,
                    )
                    continue
            break

        self._save_raw_output(turn_num, stdout)
        if parsed["session_id"]:
            self._session_id = parsed["session_id"]
            self._save_session_info()
        self._accumulate_stats(parsed["stats"])

        # Context window exhausted — graceful termination
        if parsed["errors"] and self._is_context_exceeded(parsed["errors"], stderr):
            logger.warning("Turn %d: context window exhausted, ending session gracefully", turn_num)
            self._status = "done"
        elif parsed["errors"] and returncode != 0 and not parsed["text"]:
            self._status = "error"
            self._error_message = f"rc={returncode}: {'; '.join(parsed['errors'][:3])}"[:500]

        self._trajectory.append({
            "turn": turn_num, "text": parsed["text"], "stats": parsed["stats"],
            "errors": parsed["errors"], "returncode": returncode,
            "duration_s": elapsed, "session_id": parsed["session_id"],
        })

        if parsed["errors"]:
            logger.warning("Turn %d errors: %s", turn_num, parsed["errors"])
        if stderr and returncode != 0:
            logger.warning("Turn %d stderr (rc=%d): %s", turn_num, returncode, stderr[:500])

        return parsed["text"]
