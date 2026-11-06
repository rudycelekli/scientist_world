"""Hive configuration management."""

import json
import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# Runtime directory for state, logs, etc. Set the HIVE_DIR environment
# variable to choose it; the default is <hive checkout>/.hive/. It must be on
# a filesystem shared by the head node and the compute nodes.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
HIVE_DIR = Path(os.environ["HIVE_DIR"]) if os.environ.get("HIVE_DIR") else _PROJECT_ROOT / ".hive"
PACKAGE_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "config.yaml"
USER_CONFIG_PATH = HIVE_DIR / "config.yaml"
STATE_PATH = HIVE_DIR / "state.json"
JOBS_DIR = HIVE_DIR / "jobs"
DASHBOARD_PID_PATH = HIVE_DIR / "dashboard.pid"
SLURM_TRACKING_DIR = HIVE_DIR / "slurm_tracking"
DRAINING_NODES_DIR = HIVE_DIR / "draining_nodes"


@dataclass
class ResourceSpec:
    """A named resource entry with SLURM partition, GPU count, and optional account."""
    name: str
    partition: Optional[str] = None  # SLURM partition name (omit to use cluster default)
    gpus_per_job: int = 1
    account: Optional[str] = None
    constraint: Optional[str] = None  # SLURM --constraint (e.g. "a100")
    qos: Optional[str] = None  # SLURM --qos; omit for cluster default
    mem: Optional[str] = None  # SLURM --mem (e.g. "64G")
    cpus_per_task: Optional[int] = None  # SLURM --cpus-per-task
    walltime: Optional[str] = None  # per-resource walltime, falls back to default_walltime
    exclude_nodes: Optional[str] = None  # SLURM --exclude (e.g. "node01,node02")
    max_jobs: int = 1  # max SLURM jobs for --full mode
    test: bool = False  # if True, excluded from --full
    num_jobs: int = 0  # actual jobs to launch (set at runtime)

    def to_dict(self) -> dict:
        return {"name": self.name, "gpus_per_job": self.gpus_per_job,
                "partition": self.partition, "account": self.account,
                "constraint": self.constraint, "qos": self.qos, "mem": self.mem,
                "cpus_per_task": self.cpus_per_task,
                "walltime": self.walltime,
                "exclude_nodes": self.exclude_nodes,
                "max_jobs": self.max_jobs, "test": self.test,
                "num_jobs": self.num_jobs}

    @classmethod
    def from_dict(cls, d: dict) -> "ResourceSpec":
        return cls(name=d["name"], partition=d.get("partition"),
                   gpus_per_job=d.get("gpus_per_job", 1),
                   account=d.get("account"),
                   constraint=d.get("constraint"),
                   qos=d.get("qos"),
                   mem=d.get("mem"),
                   cpus_per_task=d.get("cpus_per_task"),
                   walltime=d.get("walltime"),
                   exclude_nodes=d.get("exclude_nodes"),
                   max_jobs=d.get("max_jobs", 1), test=d.get("test", False),
                   num_jobs=d.get("num_jobs", 0))


@dataclass
class SlurmConfig:
    gpus_per_job: int = 4
    default_walltime: str = "12:00:00"
    drain_before_timeout: str = "0:05:00"
    extra_args: str = ""
    resources: List[ResourceSpec] = field(default_factory=list)


@dataclass
class RayConfig:
    head_port: int = 6379
    dashboard_port: int = 38581  # port for the Hive web dashboard
    # Bind address of the dashboard. The dashboard has no authentication and
    # its API runs arbitrary commands as the cluster owner, so it listens on
    # loopback unless told otherwise. Use "0.0.0.0" only on a trusted network.
    dashboard_host: str = "127.0.0.1"
    head_ip: Optional[str] = None  # IP for workers to reach the head (auto-detected if not set)
    launch_pending_grace_sec: int = 30  # how long to trust a new detached launch before retrying


@dataclass
class EnvConfig:
    venv_path: Optional[str] = None
    # Default working directory of submitted jobs. Empty means the directory
    # the job was submitted from.
    working_dir: str = ""
    # Ray temp directory (overridden by $HIVE_RAY_TMPDIR). Falls back to
    # /tmp/ray_$USER. Use a short path (Unix sockets are limited to 107
    # bytes) on storage that is not cleaned periodically.
    ray_tmpdir: str = ""
    # Shell lines run on each worker node before Ray starts, e.g.
    # ["module load cuda/12.4", "export HTTPS_PROXY=http://proxy:3128"].
    setup_commands: List[str] = field(default_factory=list)


@dataclass
class HiveConfig:
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    ray: RayConfig = field(default_factory=RayConfig)
    env: EnvConfig = field(default_factory=EnvConfig)


def parse_hive_duration_to_seconds(value: Optional[str]) -> int:
    """Parse a short duration string into seconds.

    Accepted forms:
    - ``SS``
    - ``MM:SS``
    - ``HH:MM:SS``
    - ``D-HH:MM:SS``
    """
    if value is None:
        return 0
    text = str(value).strip()
    if not text:
        return 0
    if text.isdigit():
        return int(text)

    days = 0
    time_part = text
    if "-" in text:
        day_part, time_part = text.split("-", 1)
        days = int(day_part)

    parts = [int(p) for p in time_part.split(":")]
    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"Invalid duration '{value}'")

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def get_ray_tmpdir(config: HiveConfig = None) -> str:
    """Resolve Ray temp directory.  Priority: $HIVE_RAY_TMPDIR > config > /tmp fallback.

    Set ``env.ray_tmpdir`` in config.yaml to a short path on storage that is
    not cleaned periodically, so Ray's sockets survive for the lifetime of
    the cluster. The path must be short enough for Unix domain sockets
    (107 byte limit).
    """
    from_env = os.environ.get("HIVE_RAY_TMPDIR", "")
    if from_env:
        return from_env
    if config and config.env.ray_tmpdir:
        return config.env.ray_tmpdir
    # Load from config file if no config object passed
    cfg = load_config()
    if cfg.env.ray_tmpdir:
        return cfg.env.ray_tmpdir
    # Fallback to /tmp (may be cleaned periodically; prefer env.ray_tmpdir)
    return f"/tmp/ray_{os.environ.get('USER', 'hive')}"


def get_launch_pending_grace_sec(config: HiveConfig = None) -> float:
    """Return the queued launch-pending grace window in seconds."""
    cfg = config or load_config()
    return max(0.0, float(cfg.ray.launch_pending_grace_sec))


def load_runtime_state(state_path: Optional[str | Path] = None) -> Optional[dict]:
    """Load Hive runtime state from the shared state file.

    This is intentionally lightweight so worker-side helpers can resolve the
    current Ray head dynamically instead of relying on a baked-in address.
    """
    path = Path(state_path) if state_path is not None else STATE_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_runtime_ray_head(state_path: Optional[str | Path] = None) -> Optional[str]:
    """Return the current Ray head address from the shared Hive state file."""
    state = load_runtime_state(state_path)
    if not state:
        return None
    return state.get("ray_head_address") or None


def _find_config_path() -> Optional[Path]:
    """Find config file: <HIVE_DIR>/config.yaml > <repo>/config.yaml."""
    if USER_CONFIG_PATH.exists():
        return USER_CONFIG_PATH
    if PACKAGE_CONFIG_PATH.exists():
        return PACKAGE_CONFIG_PATH
    return None


def load_config() -> HiveConfig:
    """Load config, searching <HIVE_DIR>/config.yaml then <repo>/config.yaml."""
    config = HiveConfig()
    config_path = _find_config_path()
    if config_path:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

        # A section that is present but empty (all keys commented out)
        # parses as None, so fall back to an empty mapping.
        for k, v in (raw.get("slurm") or {}).items():
            if k == "resources" and isinstance(v, list):
                config.slurm.resources = [
                    ResourceSpec.from_dict(p) for p in v
                ]
            elif hasattr(config.slurm, k):
                setattr(config.slurm, k, v)
        for k, v in (raw.get("ray") or {}).items():
            if hasattr(config.ray, k):
                setattr(config.ray, k, v)
        for k, v in (raw.get("env") or {}).items():
            if hasattr(config.env, k):
                setattr(config.env, k, v)

    return config


def ensure_hive_dir():
    """Create HIVE_DIR and subdirectories if they don't exist."""
    HIVE_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    SLURM_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    DRAINING_NODES_DIR.mkdir(parents=True, exist_ok=True)
