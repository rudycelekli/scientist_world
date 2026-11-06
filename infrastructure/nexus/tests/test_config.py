"""Tests for nexus.config: config loading, dataclass defaults."""

import dataclasses
from pathlib import Path
from unittest.mock import patch

import yaml

from nexus.config import (
    NexusConfig, HiveConnectionConfig, DaemonConfig,
    MonitorConfig, ProjectDefaults, load_config,
)

EXAMPLE_CONFIG = (
    Path(__file__).resolve().parent.parent
    / "nexus" / "config" / "config.example.yaml"
)


class TestDefaults:
    """Dataclass defaults are sane."""

    def test_hive_connection_defaults(self):
        c = HiveConnectionConfig()
        assert c.url == ""  # no default, must be configured

    def test_daemon_defaults(self):
        c = DaemonConfig()
        assert c.port == 39000
        # The API is unauthenticated, so it must not be exposed by default.
        assert c.host == "127.0.0.1"

    def test_monitor_defaults(self):
        c = MonitorConfig()
        assert c.status_poll_interval == 60
        assert c.checkpoint_interval == 1200
        assert c.checkpoint_timeout == 1800

    def test_project_defaults(self):
        c = ProjectDefaults()
        assert c.max_gpus == 4
        assert c.default_job_timeout == 3600
        assert c.soft_gpu_budget_hours == 0

    def test_full_config_defaults(self):
        c = NexusConfig()
        assert isinstance(c.hive, HiveConnectionConfig)
        assert isinstance(c.daemon, DaemonConfig)
        assert isinstance(c.monitor, MonitorConfig)
        assert isinstance(c.project_defaults, ProjectDefaults)


class TestLoadConfig:
    """Config loading from YAML files."""

    def test_no_config_file_returns_defaults(self):
        """Without any config file or HIVE_URL, every key is a default."""
        assert load_config() == NexusConfig()

    def test_load_from_custom_yaml(self, tmp_path):
        custom = tmp_path / "config.yaml"
        custom.write_text(yaml.dump({
            "hive": {"url": "http://custom:9999"},
            "daemon": {"port": 12345},
            "monitor": {"checkpoint_interval": 999},
        }))
        with patch("nexus.config.PACKAGE_CONFIG_PATH", custom):
            config = load_config()
        assert config.hive.url == "http://custom:9999"
        assert config.daemon.port == 12345
        assert config.monitor.checkpoint_interval == 999

    def test_user_config_takes_precedence(self, tmp_path, tmp_nexus_dir):
        """NEXUS_DIR/config.yaml wins over the package-level file."""
        package = tmp_path / "package.yaml"
        package.write_text(yaml.dump({"daemon": {"port": 11111}}))
        (tmp_nexus_dir / "config.yaml").write_text(
            yaml.dump({"daemon": {"port": 22222}})
        )
        with patch("nexus.config.PACKAGE_CONFIG_PATH", package):
            config = load_config()
        assert config.daemon.port == 22222

    def test_load_partial_yaml(self, tmp_nexus_dir):
        """Only some sections present: the rest are defaults."""
        (tmp_nexus_dir / "config.yaml").write_text(
            yaml.dump({"daemon": {"port": 55555}})
        )
        config = load_config()
        assert config.daemon.port == 55555
        assert config.daemon.host == "127.0.0.1"
        assert config.hive.url == ""
        assert config.monitor == MonitorConfig()

    def test_empty_section_and_unknown_keys_are_ignored(self, tmp_nexus_dir):
        (tmp_nexus_dir / "config.yaml").write_text(
            "hive:\n"
            "monitor:\n"
            "  no_such_key: 1\n"
            "no_such_section:\n"
            "  x: 1\n"
        )
        assert load_config() == NexusConfig()

    def test_hive_url_env_fallback(self, monkeypatch):
        monkeypatch.setenv("HIVE_URL", "http://from-env:1234")
        assert load_config().hive.url == "http://from-env:1234"

    def test_config_file_beats_hive_url_env(self, monkeypatch, tmp_nexus_dir):
        monkeypatch.setenv("HIVE_URL", "http://from-env:1234")
        (tmp_nexus_dir / "config.yaml").write_text(
            yaml.dump({"hive": {"url": "http://from-file:5678"}})
        )
        assert load_config().hive.url == "http://from-file:5678"


class TestExampleConfig:
    """config.example.yaml documents exactly the keys the code reads."""

    def test_example_lists_every_key_with_its_default(self):
        raw = yaml.safe_load(EXAMPLE_CONFIG.read_text())
        defaults = dataclasses.asdict(NexusConfig())
        assert raw == defaults

    def test_example_loads(self, tmp_nexus_dir):
        (tmp_nexus_dir / "config.yaml").write_text(EXAMPLE_CONFIG.read_text())
        assert load_config() == NexusConfig()


class TestEnsureNexusDir:
    """ensure_nexus_dir creates required directories."""

    def test_creates_directories(self, tmp_path):
        from nexus.config import ensure_nexus_dir
        nexus_dir = tmp_path / "state"
        with patch("nexus.config.NEXUS_DIR", nexus_dir), \
             patch("nexus.config.PROJECTS_DIR", nexus_dir / "projects"), \
             patch("nexus.config.JOBS_DIR", nexus_dir / "jobs"):
            ensure_nexus_dir()
        assert nexus_dir.is_dir()
        assert (nexus_dir / "projects").is_dir()
        assert (nexus_dir / "jobs").is_dir()

    def test_idempotent(self, tmp_path):
        from nexus.config import ensure_nexus_dir
        nexus_dir = tmp_path / "state"
        with patch("nexus.config.NEXUS_DIR", nexus_dir), \
             patch("nexus.config.PROJECTS_DIR", nexus_dir / "projects"), \
             patch("nexus.config.JOBS_DIR", nexus_dir / "jobs"):
            ensure_nexus_dir()
            ensure_nexus_dir()  # should not raise
        assert nexus_dir.is_dir()
