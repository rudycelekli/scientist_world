"""Shared parsing utilities for Worker implementations.

Three groups:
1. YAML action file parser — reads action.yaml from workspace
2. Claude Code output parser — parses stream-json CLI output
3. MILESTONES.md parser — reads current milestone from workspace
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


# ── YAML action file parsing ──────────────────────────────────────────────

ACTION_FILENAME = "action.yaml"

VALID_ACTIONS = ("submit", "done")

VALID_MILESTONES = (
    "not_possible",
    "none",
    "method_runs",
    "core_claim",
    "core_claim_plus",
    "secondary_claims",
    "majority",
    "near_complete",
    "full",
)

# Map common agent misspellings / variations to canonical milestone names.
# Keys are lowercase, spaces/hyphens collapsed to underscores.
_MILESTONE_ALIASES: dict[str, str] = {
    "impossible": "not_possible",
    "infeasible": "not_possible",
    "not_feasible": "not_possible",
    "method_run": "method_runs",
    "methods_run": "method_runs",
    "methods_runs": "method_runs",
    "method_running": "method_runs",
    "core_claims": "core_claim",
    "coreclaim": "core_claim",
    "core_claim_plus": "core_claim_plus",
    "coreclaimplus": "core_claim_plus",
    "core_claims_plus": "core_claim_plus",
    "secondary_claim": "secondary_claims",
    "secondaryclaims": "secondary_claims",
    "near_completed": "near_complete",
    "nearcomplete": "near_complete",
    "nearly_complete": "near_complete",
    "full_complete": "full",
    "complete": "full",
}


def _normalize_milestone(raw: str) -> str:
    """Normalize a milestone string to a canonical name.

    Strips whitespace, lowercases, replaces hyphens/spaces with underscores,
    then checks exact match → alias table → falls back to ``"none"``.
    """
    cleaned = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if cleaned in VALID_MILESTONES:
        return cleaned
    if cleaned in _MILESTONE_ALIASES:
        return _MILESTONE_ALIASES[cleaned]
    return "none"


@dataclass
class TurnAction:
    """Parsed action from the workspace action.yaml file."""

    action: Literal["submit", "done"]
    gpus: int = 1
    test: bool = False
    milestone: str = "none"
    notes: str = ""


def parse_turn_action(workspace_path: str) -> TurnAction | None:
    """Read and parse ``action.yaml`` from the workspace directory.

    The agent edits this file to signal its next action.
    Returns None if the file doesn't exist or has no valid action.
    """
    action_file = Path(workspace_path) / ACTION_FILENAME
    if not action_file.exists():
        return None

    try:
        data = yaml.safe_load(action_file.read_text())
    except yaml.YAMLError:
        return None

    if not isinstance(data, dict):
        return None

    action = data.get("action")
    if action not in VALID_ACTIONS:
        return None

    try:
        gpus = int(data.get("gpus", 1))
    except (TypeError, ValueError):
        return None

    return TurnAction(
        action=action,
        gpus=gpus,
        test=bool(data.get("test", False)),
        milestone=_normalize_milestone(str(data.get("milestone", "none"))),
        notes=str(data.get("notes", "")),
    )


# ── Claude Code output parsing ─────────────────────────────────────────────


@dataclass
class ClaudeOutput:
    """Parsed output from a Claude Code CLI invocation."""

    text: str = ""
    session_id: str | None = None
    stats: dict = field(default_factory=lambda: {
        "cost_usd": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "num_turns": 0,
        "duration_api_ms": 0,
        "tool_uses": {},
    })
    errors: list[str] = field(default_factory=list)


def parse_claude_output(stdout: str) -> ClaudeOutput:
    """Parse Claude Code ``--output-format stream-json`` output."""
    text = ""
    session_id = None
    errors: list[str] = []
    stats = {
        "cost_usd": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "num_turns": 0,
        "duration_api_ms": 0,
        "tool_uses": {},
    }

    for line in stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = data.get("type")

        if msg_type == "error":
            err = data.get("error", {})
            msg = err.get("message") if isinstance(err, dict) else None
            msg = msg or data.get("message") or str(err or data)
            errors.append(msg)

        elif msg_type == "result":
            text = data.get("result", "")
            session_id = data.get("session_id")
            if data.get("is_error"):
                errors.append(text)
            for key in ("cost_usd", "num_turns", "duration_api_ms"):
                if data.get(key) is not None:
                    stats[key] = data[key]
            # Also check total_cost_usd (newer Claude Code format)
            if data.get("total_cost_usd") is not None:
                stats["cost_usd"] = data["total_cost_usd"]
            # Result message has authoritative final token counts
            usage = data.get("usage", {})
            if usage.get("input_tokens") is not None:
                stats["input_tokens"] = usage["input_tokens"]
                stats["output_tokens"] = usage.get("output_tokens", 0)
                stats["cached_input_tokens"] = usage.get("cache_read_input_tokens", 0)
                stats["cache_creation_tokens"] = usage.get("cache_creation_input_tokens", 0)

        elif msg_type == "assistant":
            message = data.get("message", data)
            if data.get("error"):
                for block in message.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        errors.append(block["text"])
            usage = message.get("usage", {})
            stats["input_tokens"] += usage.get("input_tokens", 0)
            stats["output_tokens"] += usage.get("output_tokens", 0)
            stats["cached_input_tokens"] += usage.get("cache_read_input_tokens", 0)
            for block in message.get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name", "unknown")
                    stats["tool_uses"][name] = stats["tool_uses"].get(name, 0) + 1

    # Fallback: try entire stdout as single JSON (plain --output-format json)
    if not text and not session_id:
        try:
            data = json.loads(stdout)
            text = data.get("result", stdout)
            session_id = data.get("session_id")
            for key in ("cost_usd", "num_turns", "duration_api_ms"):
                if data.get(key) is not None:
                    stats[key] = data[key]
            if data.get("total_cost_usd") is not None:
                stats["cost_usd"] = data["total_cost_usd"]
            usage = data.get("usage", {})
            if usage.get("input_tokens") is not None:
                stats["input_tokens"] = usage["input_tokens"]
                stats["output_tokens"] = usage.get("output_tokens", 0)
                stats["cached_input_tokens"] = usage.get("cache_read_input_tokens", 0)
                stats["cache_creation_tokens"] = usage.get("cache_creation_input_tokens", 0)
        except json.JSONDecodeError:
            text = stdout

    return ClaudeOutput(text=text, session_id=session_id, stats=stats, errors=errors)


# ── MILESTONES.md parsing ──────────────────────────────────────────────────

_MILESTONE_RE = re.compile(r"\*\*Current:\s*(\S+)\*\*")


def parse_current_milestone(workspace_path: str) -> str | None:
    """Read MILESTONES.md and extract the current milestone.

    Looks for ``**Current: <milestone>**`` on the first matching line.
    Returns None if the file doesn't exist or no milestone is found.
    The raw value is normalized through :func:`_normalize_milestone`.
    """
    milestones_file = Path(workspace_path) / "MILESTONES.md"
    if not milestones_file.exists():
        return None

    for line in milestones_file.read_text().splitlines():
        m = _MILESTONE_RE.search(line)
        if m:
            return _normalize_milestone(m.group(1))

    return None
