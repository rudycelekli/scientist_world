"""Write an evaluation config from a local Swarm config (``sw-eval seed``).

A convenience for installations that also run Swarm: the ``profiles``
and ``nexus`` blocks of Swarm's ``config.yaml`` use the same schema as
the evaluation config, so they are copied instead of being typed twice.
The result is the defaults of :data:`evaluation.config.DEFAULT_CONFIG`
plus those two blocks, plus ``data_dir`` and ``agent_sif`` derived from
the sciworld CLI's settings when they exist.

Non-destructive: keys already set in the evaluation config win. Pass
``overwrite=True`` to replace them. Review the written file afterwards;
paths, the grader profile and publishing are not seeded.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path

import yaml

from evaluation import config as _config

log = logging.getLogger(__name__)

_SWARM_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "infrastructure" / "swarm" / "swarm" / "config" / "config.yaml"
)


def _load_swarm_config(path: Path) -> dict:
    if not path.exists():
        raise _config.ConfigError(
            f"Swarm config not found at {path}. Pass its location with "
            "--swarm-config, or write the evaluation config by hand from "
            "config.example.yaml."
        )
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise _config.ConfigError(f"{path} did not parse to a mapping")
    return data


def _sciworld_config_path() -> Path:
    """Settings file of the sciworld CLI (``$SCIWORLD_HOME`` or ``~/.sciworld``)."""
    home = os.environ.get("SCIWORLD_HOME")
    base = Path(home).expanduser() if home else Path.home() / ".sciworld"
    return base / "config.json"


def _existing_config() -> dict:
    """What the evaluation config file itself states (no defaults)."""
    if not _config.CONFIG_PATH.exists():
        return {}
    data = yaml.safe_load(_config.CONFIG_PATH.read_text()) or {}
    if not isinstance(data, dict):
        raise _config.ConfigError(f"{_config.CONFIG_PATH} did not parse to a mapping")
    return data


def seed(overwrite: bool = False, swarm_config: Path | str | None = None) -> Path:
    """Write defaults + Swarm's profiles and nexus block to the config file.

    Returns the config path. Fields already set in the existing file are
    kept unless *overwrite* is True.
    """
    swarm_cfg = _load_swarm_config(Path(swarm_config) if swarm_config else _SWARM_CONFIG)
    swarm_profiles = dict(swarm_cfg.get("profiles") or {})
    existing = _existing_config()

    merged: dict = copy.deepcopy(_config.DEFAULT_CONFIG)
    if not overwrite:
        merged.update(existing)

    # Profiles: Swarm's are added; profiles already present in the
    # evaluation config are kept as they are.
    profiles = dict(swarm_profiles)
    if not overwrite:
        profiles.update(existing.get("profiles") or {})
    merged["profiles"] = profiles

    # Runs use the agent image built by the sciworld CLI (`sw init`), so
    # take data_dir and agent_sif from its settings when they exist.
    sciworld_cfg = _sciworld_config_path()
    if sciworld_cfg.exists():
        try:
            sw = json.loads(sciworld_cfg.read_text())
            data_dir = sw.get("data_dir")
            if data_dir:
                if overwrite or not existing.get("data_dir"):
                    merged["data_dir"] = str(Path(data_dir) / "evaluation")
                if overwrite or not existing.get("agent_sif"):
                    candidate = Path(data_dir) / "containers" / "agent.sif"
                    if candidate.exists():
                        merged["agent_sif"] = str(candidate)
        except (OSError, json.JSONDecodeError):
            pass

    # Swarm's nexus block has the same structure (url, gpu_budget_hours,
    # max_gpus, resources).
    swarm_nexus = swarm_cfg.get("nexus") or {}
    if swarm_nexus and (overwrite or "nexus" not in existing):
        merged["nexus"] = {**merged.get("nexus", {}), **swarm_nexus}

    _config.save(merged)
    log.info("Seeded %s with %d profiles", _config.CONFIG_PATH, len(profiles))
    return _config.CONFIG_PATH
