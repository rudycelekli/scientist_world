"""Nexus configuration management.

Configuration is loaded from the first YAML file that exists:

    1. NEXUS_DIR/config.yaml     (per-deployment override)
    2. nexus/config/config.yaml  (next to this package, gitignored)

Keys missing from the file, and all keys when no file exists, take the
dataclass defaults below.

All runtime state (projects, jobs, PID file, daemon log) lives under
NEXUS_DIR. It defaults to ``.nexus/`` next to the ``nexus`` package and
can be moved with the ``NEXUS_DIR`` environment variable. It is never
placed in the home directory.
"""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Runtime directory: the NEXUS_DIR env var, or .nexus/ next to the package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEXUS_DIR = Path(os.environ["NEXUS_DIR"]) if os.environ.get("NEXUS_DIR") else _PROJECT_ROOT / ".nexus"
PACKAGE_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "config.yaml"
USER_CONFIG_PATH = NEXUS_DIR / "config.yaml"
PROJECTS_DIR = NEXUS_DIR / "projects"
JOBS_DIR = NEXUS_DIR / "jobs"
DAEMON_PID_PATH = NEXUS_DIR / "daemon.pid"
REMOTE_CONFIG_PATH = NEXUS_DIR / "remote.json"


@dataclass
class HiveConnectionConfig:
    """Connection settings for the Hive compute pool.

    ``url`` has no default. It must be set in config.yaml or through the
    ``HIVE_URL`` environment variable.
    """
    url: str = ""


@dataclass
class DaemonConfig:
    """Nexus daemon HTTP server settings.

    The API has no authentication, so the default bind address is the
    loopback interface. Set ``host`` to ``0.0.0.0`` (or to one interface
    address) only on a network where every client is trusted.
    """
    port: int = 39000
    host: str = "127.0.0.1"


@dataclass
class MonitorConfig:
    """Job monitoring and checkpoint settings.

    Attributes:
        status_poll_interval: Seconds between Hive status polls of active jobs.
        checkpoint_interval: Seconds a job runs between checkpoints.
        checkpoint_timeout: Seconds to wait for a continue/cancel decision
            at a checkpoint before the job is continued automatically.
    """
    status_poll_interval: int = 60
    checkpoint_interval: int = 1200
    checkpoint_timeout: int = 1800


@dataclass
class ProjectDefaults:
    """Default project resource limits.

    Applied when a project is registered without explicit overrides.

    Attributes:
        max_gpus: Maximum GPUs that can be in use simultaneously.
        default_job_timeout: Default per-job timeout in seconds (1 hour).
        soft_gpu_budget_hours: Cumulative GPU-time budget in hours (0 = unlimited).
    """
    max_gpus: int = 4
    default_job_timeout: int = 3600
    soft_gpu_budget_hours: float = 0


@dataclass
class NexusConfig:
    """Top-level Nexus configuration."""
    hive: HiveConnectionConfig = field(default_factory=HiveConnectionConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    project_defaults: ProjectDefaults = field(default_factory=ProjectDefaults)


def _find_config_path() -> Optional[Path]:
    """Find config file: NEXUS_DIR/config.yaml > package/config.yaml."""
    if USER_CONFIG_PATH.exists():
        return USER_CONFIG_PATH
    if PACKAGE_CONFIG_PATH.exists():
        return PACKAGE_CONFIG_PATH
    return None


def load_config() -> NexusConfig:
    """Load config, searching NEXUS_DIR/config.yaml then package/config.yaml.

    Missing keys fall back to dataclass defaults. Unknown keys are ignored.
    """
    config = NexusConfig()
    config_path = _find_config_path()
    if config_path:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

        for section in ("hive", "daemon", "monitor", "project_defaults"):
            target = getattr(config, section)
            for k, v in (raw.get(section) or {}).items():
                if hasattr(target, k):
                    setattr(target, k, v)

    # The HIVE_URL env var is the fallback for an empty hive.url.
    if not config.hive.url:
        config.hive.url = os.environ.get("HIVE_URL", "")

    return config


def ensure_nexus_dir():
    """Create NEXUS_DIR and subdirectories if they don't exist."""
    Path(NEXUS_DIR).mkdir(parents=True, exist_ok=True)
    Path(PROJECTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(JOBS_DIR).mkdir(parents=True, exist_ok=True)
