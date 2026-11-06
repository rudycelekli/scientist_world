"""Parse ``action.yaml``, the agent's per-turn signal.

Evaluation runs are single-session, so only the within-run turn
protocol is needed: ``submit``, ``done``, and the two checkpoint
decisions ``continue`` and ``cancel``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

ACTION_FILENAME = "action.yaml"
# ``continue`` / ``cancel`` are checkpoint-only: the loop asks for them
# when Nexus moves a running job to ``awaiting_response``, so the agent
# can review the logs and decide to keep the job or cancel it.
VALID_ACTIONS = ("submit", "done", "continue", "cancel")


@dataclass
class TurnAction:
    """Parsed content of ``<env_dir>/action.yaml``."""

    action: Literal["submit", "done", "continue", "cancel"]
    gpus: int = 1
    test: bool = False
    notes: str = ""


def parse_turn_action(workspace_path: str | Path) -> TurnAction | None:
    """Read and parse ``action.yaml`` from the workspace root.

    Returns ``None`` if the file is missing, unreadable, or has no
    recognizable action.
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
        gpus = max(1, int(data.get("gpus", 1)))
    except (TypeError, ValueError):
        gpus = 1

    return TurnAction(
        action=action,
        gpus=gpus,
        test=bool(data.get("test", False)),
        notes=str(data.get("notes", "")),
    )
