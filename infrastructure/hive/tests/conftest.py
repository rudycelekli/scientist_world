"""Shared fixtures for hive tests."""

from unittest.mock import patch

import pytest

from hive.config import HiveConfig, ResourceSpec, SlurmConfig, RayConfig, EnvConfig


@pytest.fixture(autouse=True)
def _no_local_config(tmp_path, monkeypatch):
    """Keep tests independent of any config.yaml on the developer's machine."""
    missing = tmp_path / "no-such-config.yaml"
    monkeypatch.setattr("hive.config.PACKAGE_CONFIG_PATH", missing)
    monkeypatch.setattr("hive.config.USER_CONFIG_PATH", missing)


@pytest.fixture
def tmp_hive_dir(tmp_path):
    """Temporary .hive directory for tests that touch the filesystem."""
    hive_dir = tmp_path / ".hive"
    hive_dir.mkdir()
    (hive_dir / "jobs").mkdir()
    (hive_dir / "logs").mkdir()
    with patch("hive.config.HIVE_DIR", hive_dir), \
         patch("hive.config.STATE_PATH", hive_dir / "state.json"), \
         patch("hive.config.JOBS_DIR", hive_dir / "jobs"), \
         patch("hive.config.DRAINING_NODES_DIR", hive_dir / "draining_nodes"), \
         patch("hive.dispatcher.STATE_PATH", hive_dir / "state.json"), \
         patch("hive.dispatcher.JOBS_DIR", hive_dir / "jobs"), \
         patch("hive.dispatcher.DRAINING_NODES_DIR", hive_dir / "draining_nodes"):
        yield hive_dir


@pytest.fixture
def sample_config():
    """Minimal HiveConfig for unit tests."""
    return HiveConfig(
        slurm=SlurmConfig(
            default_walltime="1:00:00",
            resources=[
                ResourceSpec(
                    name="test-gpu",
                    partition="gpu-test",
                    gpus_per_job=2,
                    cpus_per_task=8,
                    mem="64G",
                    walltime="0:59:59",
                    constraint="a100",
                    max_jobs=2,
                    test=True,
                ),
                ResourceSpec(
                    name="prod-gpu",
                    partition="gpu",
                    gpus_per_job=4,
                    cpus_per_task=16,
                    mem="256G",
                    walltime="12:00:00",
                    max_jobs=4,
                ),
            ],
        ),
        ray=RayConfig(head_port=6379),
        env=EnvConfig(venv_path="/tmp/test-venv"),
    )


@pytest.fixture
def sample_resource():
    """Single ResourceSpec for targeted tests."""
    return ResourceSpec(
        name="test-gpu",
        partition="gpu-test",
        gpus_per_job=2,
        cpus_per_task=8,
        mem="64G",
        walltime="0:59:59",
        constraint="a100",
        max_jobs=2,
        test=True,
    )
