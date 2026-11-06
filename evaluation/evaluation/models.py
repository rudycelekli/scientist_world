"""Model profile registry.

A profile ties a scaffold (``claude-code``, ``codex``, ``kimi``,
``gemini`` or ``claude-tmux``) to a specific model, effort, and optional
gateway credentials / environment variables. Profiles are read from the
``profiles`` key of the evaluation config; the schema is the same as
Swarm's, so a profile block can be copied between the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evaluation import config as _config


@dataclass
class Profile:
    name: str
    scaffold: str
    model: str
    effort: str | None = None
    actual_model: str | None = None
    compact_token_limit: int | None = None
    env_vars: dict[str, str] = field(default_factory=dict)
    api_keys: list[str] = field(default_factory=list)
    # Per-profile override of agent.py's per-turn inactivity kill. None =
    # the module default. Only needed for endpoints whose rate-limit
    # blocks are longer than the default inactivity window, where a patient
    # retry is indistinguishable from a hang (both are silence).
    idle_timeout_sec: int | None = None
    # External identity. Several profiles may serve the same model
    # through different endpoints; everything that leaves the machine
    # (git branches, commit author) and the normalization cohort use
    # publish_as, so such profiles count as one model.
    publish_as: str | None = None

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "Profile":
        return cls(
            name=name,
            scaffold=data["scaffold"],
            model=data["model"],
            effort=data.get("effort"),
            actual_model=data.get("actual_model"),
            compact_token_limit=data.get("compact_token_limit"),
            env_vars=dict(data.get("env_vars", {})),
            api_keys=list(data.get("api_keys", [])),
            idle_timeout_sec=data.get("idle_timeout_sec"),
            publish_as=data.get("publish_as"),
        )


def load_profiles() -> dict[str, Profile]:
    """Return all configured profiles keyed by name."""
    cfg = _config.load()
    return {
        name: Profile.from_dict(name, data)
        for name, data in (cfg.get("profiles") or {}).items()
    }


def get_profile(name: str) -> Profile:
    """Look up a profile, raising KeyError if missing."""
    profiles = load_profiles()
    if name not in profiles:
        raise KeyError(
            f"Unknown profile {name!r}. Available: {sorted(profiles)}"
        )
    return profiles[name]
