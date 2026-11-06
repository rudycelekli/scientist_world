"""Configuration for the evaluation harness.

The configuration file is ``config.yaml`` next to this module
(``evaluation/evaluation/config.yaml``). It is not tracked;
``config.example.yaml`` is the documented template. Set
``EVALUATION_DIR=/abs/dir`` to read ``<dir>/config.yaml`` instead (the
test suite does this).

Every key has a default in :data:`DEFAULT_CONFIG`. Keys with no sensible
default (``data_dir``, ``agent_sif``, the Nexus resource names, the
grader profile, the GitHub organisation) default to empty and raise
:class:`ConfigError` with an explanation at the point of use.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

_PACKAGE_DIR = Path(__file__).resolve().parent

if os.environ.get("EVALUATION_DIR"):
    CONFIG_DIR = Path(os.environ["EVALUATION_DIR"])
else:
    CONFIG_DIR = _PACKAGE_DIR
CONFIG_PATH = CONFIG_DIR / "config.yaml"


class ConfigError(RuntimeError):
    """A required configuration value is missing or unusable."""


DEFAULT_CONFIG: dict[str, Any] = {
    # ── Concurrency ────────────────────────────────────────────────────
    "max_concurrent": 10,
    # Per-profile in-flight cap, e.g. {"my-profile": 6}. Profiles not
    # listed are only limited by max_concurrent.
    "profile_concurrency": {},
    "max_turns": 200,
    "max_retries": 1,

    # ── Normalization cohort ───────────────────────────────────────────
    # The cohort defines the per-paper spread used by the scoring (see
    # docs/scoring.md). Empty profile_cohort = every configured profile.
    # profile_cohort_exclude removes names from either form.
    "profile_cohort": [],
    "profile_cohort_exclude": [],

    # ── Paths ──────────────────────────────────────────────────────────
    # data_dir: root of all run state (required). agent_sif: the agent
    # container image built by the sciworld CLI (required to run agents).
    "data_dir": "",
    "agent_sif": "",
    "container_cache_dir": "",
    "uv_cache_dir": "",
    "local_tmp_dir": "/tmp",
    # .gitignore template installed into every run workspace. Empty =
    # the worker template shipped in this repository.
    "workspace_gitignore": "",

    # ── Agent container limits ─────────────────────────────────────────
    # agent_memory is a cgroup hard cap; agent_cpu_threads bounds BLAS
    # fan-out through OMP/MKL/OPENBLAS/NUMEXPR/VECLIB environment
    # variables; agent_nice lowers scheduler priority.
    "agent_memory": "8G",
    "agent_cpu_threads": 2,
    "agent_nice": 19,

    # ── Dashboard ──────────────────────────────────────────────────────
    # The dashboard has no authentication and exposes endpoints that
    # start runs, so it binds the loopback interface by default.
    "host": "127.0.0.1",
    "port": 38582,
    # Optional JSON list of paper ids (path relative to data_dir) that
    # restricts what the dashboard offers and accepts for queueing.
    "cohort_file": "",
    "dashboard": {
        # {"label": [paper ids]}: extra coverage strips and charts
        # restricted to each named subset of papers.
        "paper_groups": {},
        # Profile names shown first, in this order, in the matrix.
        "model_order": [],
    },

    # ── Agent CLI tuning ───────────────────────────────────────────────
    # Retry count, request timeout and per-response output cap passed to
    # the claude CLI inside the agent container.
    "claude_code_max_retries": 50,
    "api_timeout_ms": 3000000,
    "claude_code_max_output_tokens": 64000,
    # Model used by the host-side call that refreshes a Claude OAuth
    # token before it is copied into a workspace.
    "claude_oauth_refresh_model": "claude-haiku-4-5-20251001",
    # claude-tmux scaffold only: host path of the claude binary to bind
    # into the agent container.
    "claude_tmux_binary": "",

    # ── Job checkpoints ────────────────────────────────────────────────
    # Every checkpoint_interval seconds Nexus pauses a running job and
    # the agent decides whether to continue or cancel it;
    # checkpoint_timeout is how long Nexus waits for that decision.
    "checkpoint_interval": 1200,
    "checkpoint_timeout": 1800,
    # Time limit for scripts/download.sh during `warm`.
    "warm_download_timeout": 7200,

    # ── Model profiles ─────────────────────────────────────────────────
    # Keyed by profile name; see models.Profile for the schema.
    "profiles": {},

    # ── Nexus / compute ────────────────────────────────────────────────
    "nexus": {
        "url": "http://localhost:39000",
        "gpu_budget_hours": 30.0,
        "max_gpus": 2,
        # Hive resource names per job category. They must match the
        # resource names of the Hive pool behind Nexus.
        "resources": {
            "gpu": [],
            "gpu_test": [],
            "cpu": [],
            "cpu_test": [],
        },
    },

    # ── Independent verification (verifier.py) ─────────────────────────
    "verifier": {
        "enabled": False,
        "grader_profile": "",
        "max_turns": 12,
        "gpu_budget_hours": 8.0,
        "max_relaunches": 2,
        "publish": True,
    },

    # ── Publishing to GitHub (git_publish.py) ──────────────────────────
    "publish": {
        "enabled": False,
        "github_org": "",
    },
}

# Top-level keys whose value is a mapping merged key by key with the
# defaults, so a config file only needs to state what it changes.
_MERGED_BLOCKS = ("nexus", "verifier", "publish", "dashboard")

# Parsed config file, keyed by (path, mtime, size). load() is on hot
# paths (normalization resolves envs_dir() once per paper), and parsing
# a large YAML file on every call is slow. Edits are still picked up
# because the file's mtime and size are part of the key.
_PARSED_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}


def _parse_config_file(path: Path) -> dict[str, Any]:
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    hit = _PARSED_CACHE.get(key)
    if hit is None:
        parsed = yaml.safe_load(path.read_text()) or {}
        if not isinstance(parsed, dict):
            raise ConfigError(f"{path} did not parse to a mapping")
        _PARSED_CACHE.clear()
        _PARSED_CACHE[key] = parsed
        hit = parsed
    # Callers get their own copy: load() builds a fresh dict on top of it
    # and some callers mutate nested blocks in place.
    return copy.deepcopy(hit)


def load() -> dict[str, Any]:
    """Return the configuration with defaults filled in.

    A missing ``config.yaml`` yields the defaults. The mapping blocks
    ``nexus``, ``verifier``, ``publish`` and ``dashboard`` are merged key
    by key with their defaults, and ``nexus.resources`` one level deeper.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    user: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        user = _parse_config_file(CONFIG_PATH)
        cfg.update(user)
    cfg["profiles"] = dict(cfg.get("profiles") or {})
    for block in _MERGED_BLOCKS:
        cfg[block] = {
            **copy.deepcopy(DEFAULT_CONFIG[block]),
            **(user.get(block) or {}),
        }
    cfg["nexus"]["resources"] = {
        **copy.deepcopy(DEFAULT_CONFIG["nexus"]["resources"]),
        **((user.get("nexus") or {}).get("resources") or {}),
    }
    return cfg


def nexus_all_resources(cfg: dict[str, Any] | None = None) -> list[str]:
    """Flat, de-duplicated union of every resource group, in declaration order.

    Used when a Nexus project is created: the project is allowed to use
    any of these resources. Per-job selection is :func:`nexus_pick`.
    """
    cfg = cfg or load()
    resources = (cfg.get("nexus") or {}).get("resources") or {}
    seen: set[str] = set()
    flat: list[str] = []
    for group in ("gpu", "gpu_test", "cpu", "cpu_test"):
        for r in resources.get(group) or []:
            if r not in seen:
                seen.add(r)
                flat.append(r)
    return flat


def nexus_pick(gpus: int, test: bool = False,
               cfg: dict[str, Any] | None = None) -> str:
    """Pick the resource group for one job.

    Returns the comma-joined resource names Nexus passes to Hive for
    this job: ``gpu`` (or ``gpu_test`` when ``test`` is set) for jobs
    that request GPUs, ``cpu`` / ``cpu_test`` otherwise.
    """
    cfg = cfg or load()
    resources = (cfg.get("nexus") or {}).get("resources") or {}
    if gpus > 0:
        category = "gpu_test" if test else "gpu"
    else:
        category = "cpu_test" if test else "cpu"
    pool = resources.get(category) or []
    if not pool:
        raise ConfigError(
            f"No resources configured for category '{category}'. "
            f"Add Hive resource names to nexus.resources.{category} in "
            f"{CONFIG_PATH}."
        )
    return ",".join(pool)


def check_run_config(cfg: dict[str, Any] | None = None) -> None:
    """Raise :class:`ConfigError` listing everything a run still needs.

    Called before a run touches the filesystem or the cluster, so a
    missing setting fails immediately instead of mid-run.
    """
    cfg = cfg or load()
    problems: list[str] = []
    if not cfg.get("data_dir"):
        problems.append("data_dir is not set (root directory for run state)")
    if not cfg.get("agent_sif"):
        problems.append("agent_sif is not set (path of the agent container "
                        "image built by the sciworld CLI)")
    resources = (cfg.get("nexus") or {}).get("resources") or {}
    for group in ("gpu", "gpu_test"):
        if not resources.get(group):
            problems.append(
                f"nexus.resources.{group} is empty (list the Hive resource "
                "names jobs of this category may use; gpu_test may repeat "
                "the gpu list)")
    verifier = cfg.get("verifier") or {}
    if verifier.get("enabled") and not verifier.get("grader_profile"):
        problems.append("verifier.enabled is true but verifier.grader_profile "
                        "is not set")
    if problems:
        raise ConfigError(
            f"Incomplete configuration in {CONFIG_PATH}:\n  - "
            + "\n  - ".join(problems)
            + "\nSee config.example.yaml for every key."
        )


def save(cfg: dict[str, Any]) -> None:
    """Persist config to disk as YAML."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
    _PARSED_CACHE.clear()


def data_dir() -> Path:
    """Root directory of all run state (``data_dir`` in the config)."""
    value = load().get("data_dir")
    if not value:
        raise ConfigError(
            f"data_dir is not set. Copy config.example.yaml to {CONFIG_PATH} "
            "and set data_dir to a directory with enough space for run "
            "workspaces and container images."
        )
    return Path(str(value)).expanduser()


def envs_dir() -> Path:
    """Where per-run workspaces live: ``<data_dir>/envs``."""
    return data_dir() / "envs"
