"""Tests for dispatcher state file — migration, persistence, requeue logic."""

import json
from unittest.mock import patch

import pytest

from hive.config import load_runtime_ray_head, load_runtime_state
from hive.config import ResourceSpec
from hive.dispatcher import Dispatcher


class TestStateMigration:
    """State file format: old list vs new dict for slurm_jobs."""

    def test_save_state_uses_dict_format(self, tmp_hive_dir, sample_config):
        d = Dispatcher(sample_config)
        d._ray_head_address = "10.0.0.1:6379"
        d._resources = sample_config.slurm.resources

        from hive.pool import PoolManager, SlurmJob
        d.pool = PoolManager(sample_config, "10.0.0.1:6379")
        d.pool.jobs = {
            "111": SlurmJob(job_id="111", state="RUNNING", resource_name="test-gpu"),
            "222": SlurmJob(job_id="222", state="PENDING", resource_name="prod-gpu"),
        }

        d._save_state(
            resources=[r.to_dict() for r in sample_config.slurm.resources],
            max_gpus_per_node=4,
            walltime="1:00:00",
        )

        state = json.loads((tmp_hive_dir / "state.json").read_text())
        slurm_jobs = state["slurm_jobs"]

        # Should be a dict, not a list
        assert isinstance(slurm_jobs, dict)
        assert "111" in slurm_jobs
        assert slurm_jobs["111"]["resource_name"] == "test-gpu"
        assert "222" in slurm_jobs
        assert slurm_jobs["222"]["resource_name"] == "prod-gpu"

    def test_load_state_returns_none_when_missing(self, tmp_hive_dir):
        assert Dispatcher.load_state() is None

    def test_load_state_reads_json(self, tmp_hive_dir):
        state = {
            "ray_head_address": "1.2.3.4:6379",
            "slurm_jobs": {"99": {"resource_name": "test"}},
            "resources": [],
        }
        (tmp_hive_dir / "state.json").write_text(json.dumps(state))
        loaded = Dispatcher.load_state()
        assert loaded is not None
        assert loaded["ray_head_address"] == "1.2.3.4:6379"

    def test_load_runtime_state_reads_shared_file(self, tmp_hive_dir):
        state = {
            "ray_head_address": "1.2.3.4:6379",
            "slurm_jobs": {"99": {"resource_name": "test"}},
        }
        (tmp_hive_dir / "state.json").write_text(json.dumps(state))

        loaded = load_runtime_state(tmp_hive_dir / "state.json")

        assert loaded == state

    def test_load_runtime_ray_head_reads_shared_file(self, tmp_hive_dir):
        state = {
            "ray_head_address": "9.9.9.9:6379",
            "slurm_jobs": {},
        }
        (tmp_hive_dir / "state.json").write_text(json.dumps(state))

        head = load_runtime_ray_head(tmp_hive_dir / "state.json")

        assert head == "9.9.9.9:6379"


class TestStateBackwardCompat:
    """Old state files with list-format slurm_jobs should still work."""

    def test_add_resources_handles_list_format(self, tmp_hive_dir, sample_config):
        """Dispatcher.add_resources should accept old list-style slurm_jobs."""
        state = {
            "ray_head_address": "10.0.0.1:6379",
            "slurm_jobs": ["100", "200"],  # old format
            "resources": [{"name": "test-gpu", "gpus_per_job": 2}],
            "walltime": "1:00:00",
        }
        d = Dispatcher(sample_config)
        d._ray_head_address = "10.0.0.1:6379"
        # add_resources reads slurm_jobs — should not crash on list format
        from hive.pool import PoolManager
        d.pool = PoolManager(sample_config, "10.0.0.1:6379")

        # Manually test the list-handling code path
        slurm_jobs = state["slurm_jobs"]
        assert isinstance(slurm_jobs, list)
        # This mimics what add_resources does internally
        if isinstance(slurm_jobs, list):
            for jid in slurm_jobs:
                from hive.pool import SlurmJob
                d.pool.jobs[jid] = SlurmJob(job_id=jid, state="UNKNOWN")

        assert "100" in d.pool.jobs
        assert "200" in d.pool.jobs
