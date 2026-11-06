"""Tests for hive.config — config loading, ResourceSpec serialization."""

from pathlib import Path
from unittest.mock import patch

import yaml

import hive.config as hive_config
from hive.config import HiveConfig, ResourceSpec, load_config

EXAMPLE_CONFIG = Path(hive_config.__file__).resolve().parent / "config" / "config.example.yaml"


def _load_from(path, tmp_path):
    with patch("hive.config.PACKAGE_CONFIG_PATH", path), \
         patch("hive.config.USER_CONFIG_PATH", tmp_path / "nonexistent"):
        return load_config()


class TestResourceSpec:
    """ResourceSpec dataclass serialization round-trip."""

    def test_to_dict_from_dict_roundtrip(self, sample_resource):
        sample_resource.exclude_nodes = "node01,node02"
        sample_resource.qos = "normal"
        d = sample_resource.to_dict()
        restored = ResourceSpec.from_dict(d)
        assert restored == sample_resource

    def test_from_dict_defaults(self):
        r = ResourceSpec.from_dict({"name": "minimal"})
        assert r.gpus_per_job == 1
        assert r.max_jobs == 1
        assert r.test is False
        assert r.num_jobs == 0
        assert r.partition is None
        assert r.exclude_nodes is None

    def test_from_dict_ignores_unknown_keys(self):
        r = ResourceSpec.from_dict({"name": "x", "no_such_key": 1})
        assert r.name == "x"
        assert not hasattr(r, "no_such_key")


class TestLoadConfig:
    """Config loading from YAML files."""

    def test_defaults_without_config_file(self):
        config = load_config()
        assert isinstance(config, HiveConfig)
        assert config.slurm.resources == []
        assert config.ray.dashboard_host == "127.0.0.1"
        assert config.env.working_dir == ""
        assert config.env.setup_commands == []

    def test_example_config_parses(self, tmp_path):
        """The shipped example must load with the real loader."""
        assert EXAMPLE_CONFIG.exists()
        config = _load_from(EXAMPLE_CONFIG, tmp_path)
        names = [r.name for r in config.slurm.resources]
        assert names, "example config defines no resource"
        assert len(names) == len(set(names))
        assert config.ray.dashboard_host == "127.0.0.1"

    def test_example_config_has_only_known_keys(self):
        """Every key in the example is one the loader actually reads."""
        raw = yaml.safe_load(EXAMPLE_CONFIG.read_text())
        defaults = HiveConfig()
        assert set(raw) <= {"slurm", "ray", "env"}
        for key in raw.get("slurm", {}):
            assert key == "resources" or hasattr(defaults.slurm, key), key
        for key in raw.get("ray", {}):
            assert hasattr(defaults.ray, key), key
        for key in (raw.get("env") or {}):
            assert hasattr(defaults.env, key), key
        resource_fields = set(ResourceSpec(name="x").to_dict())
        for res in raw["slurm"]["resources"]:
            assert set(res) <= resource_fields, set(res) - resource_fields

    def test_load_from_custom_yaml(self, tmp_path):
        custom = tmp_path / "config.yaml"
        custom.write_text(yaml.dump({
            "slurm": {
                "default_walltime": "6:00:00",
                "resources": [
                    {"name": "custom", "gpus_per_job": 8, "partition": "big",
                     "exclude_nodes": "node07"},
                    {"name": "custom-test", "gpus_per_job": 1, "test": True},
                ],
            },
            "ray": {"dashboard_host": "0.0.0.0", "head_port": 7001},
            "env": {"setup_commands": ["module load cuda"], "working_dir": "/work"},
        }))
        config = _load_from(custom, tmp_path)
        assert config.slurm.default_walltime == "6:00:00"
        assert config.slurm.resources[0].name == "custom"
        assert config.slurm.resources[0].gpus_per_job == 8
        assert config.slurm.resources[0].exclude_nodes == "node07"
        assert config.slurm.resources[1].test is True
        assert config.ray.dashboard_host == "0.0.0.0"
        assert config.ray.head_port == 7001
        assert config.env.setup_commands == ["module load cuda"]
        assert config.env.working_dir == "/work"

    def test_unknown_sections_are_ignored(self, tmp_path):
        custom = tmp_path / "config.yaml"
        custom.write_text(yaml.dump({"no_such_section": {"a": 1}, "ray": {"no_such_key": 2}}))
        config = _load_from(custom, tmp_path)
        assert not hasattr(config, "no_such_section")
        assert not hasattr(config.ray, "no_such_key")
