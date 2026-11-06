"""Tests for hive.pool — sbatch script generation, SLURM job management."""

import re

import pytest

from hive.config import ResourceSpec
from hive.pool import PoolManager, SlurmJob


class TestSbatchScript:
    """Verify generated sbatch scripts have correct directives."""

    def _make_pool(self, sample_config):
        return PoolManager(sample_config, "10.0.0.1:6379")

    def test_gpu_directive(self, sample_config):
        pm = self._make_pool(sample_config)
        r = sample_config.slurm.resources[0]  # test-gpu, 2 GPUs
        script = pm._build_sbatch_script(r, "1:00:00")
        assert "#SBATCH --gres=gpu:2" in script

    def test_partition_directive(self, sample_config):
        pm = self._make_pool(sample_config)
        r = sample_config.slurm.resources[0]
        script = pm._build_sbatch_script(r, "1:00:00")
        assert "#SBATCH --partition=gpu-test" in script

    def test_constraint_directive(self, sample_config):
        pm = self._make_pool(sample_config)
        r = sample_config.slurm.resources[0]
        script = pm._build_sbatch_script(r, "1:00:00")
        assert "#SBATCH --constraint=a100" in script

    def test_mem_directive(self, sample_config):
        pm = self._make_pool(sample_config)
        r = sample_config.slurm.resources[0]
        script = pm._build_sbatch_script(r, "1:00:00")
        assert "#SBATCH --mem=64G" in script

    def test_cpus_per_task_directive(self, sample_config):
        pm = self._make_pool(sample_config)
        r = sample_config.slurm.resources[0]
        script = pm._build_sbatch_script(r, "1:00:00")
        assert "#SBATCH --cpus-per-task=8" in script

    def test_walltime_uses_argument(self, sample_config):
        """Walltime comes from the caller argument, not the resource field."""
        pm = self._make_pool(sample_config)
        r = sample_config.slurm.resources[0]
        script = pm._build_sbatch_script(r, "2:30:00")
        assert "#SBATCH --time=2:30:00" in script

    def test_ray_start_command(self, sample_config):
        pm = self._make_pool(sample_config)
        r = sample_config.slurm.resources[0]
        script = pm._build_sbatch_script(r, "1:00:00")
        assert "_hive_current_head()" in script
        assert '_RAY_HEAD=$(_hive_current_head)' in script
        assert 'ray start --address="$_RAY_HEAD"' in script
        assert "--num-gpus=2" in script
        assert "10.0.0.1:6379" not in script

    def test_partition_resource_label(self, sample_config):
        pm = self._make_pool(sample_config)
        r = sample_config.slurm.resources[0]
        script = pm._build_sbatch_script(r, "1:00:00")
        assert '"partition:test-gpu"' in script

    def test_worker_only_joins_ray(self, sample_config):
        """Between `ray start` and the wait loop the worker runs nothing else,
        and the script leaves thread-count variables to the job."""
        pm = self._make_pool(sample_config)
        r = sample_config.slurm.resources[0]
        script = pm._build_sbatch_script(r, "1:00:00")
        python_calls = [ln for ln in script.splitlines() if "python -m " in ln]
        assert all("hive.node_drain" in ln for ln in python_calls), python_calls
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            assert var not in script

    def test_exports_state_dir(self, sample_config, tmp_hive_dir):
        """Workers must resolve the same HIVE_DIR as the head node."""
        import hive.pool as pool_mod
        from unittest.mock import patch
        with patch.object(pool_mod, "HIVE_DIR", tmp_hive_dir):
            pm = self._make_pool(sample_config)
            script = pm._build_sbatch_script(sample_config.slurm.resources[0], "1:00:00")
        assert f"export HIVE_DIR={tmp_hive_dir}" in script

    def test_setup_commands_emitted_before_ray_start(self, sample_config):
        sample_config.env.setup_commands = ["module load cuda/12.4", "export FOO=bar"]
        pm = self._make_pool(sample_config)
        script = pm._build_sbatch_script(sample_config.slurm.resources[0], "1:00:00")
        assert "module load cuda/12.4" in script
        assert script.index("export FOO=bar") < script.index("ray start --address")

    def test_no_setup_commands_by_default(self, sample_config):
        pm = self._make_pool(sample_config)
        script = pm._build_sbatch_script(sample_config.slurm.resources[0], "1:00:00")
        assert "module load" not in script

    def test_exclude_nodes_directive(self, sample_config):
        r = ResourceSpec(name="ex", gpus_per_job=1, exclude_nodes="node01,node02")
        pm = self._make_pool(sample_config)
        script = pm._build_sbatch_script(r, "1:00:00")
        assert "#SBATCH --exclude=node01,node02" in script

    def test_no_ray_block_flag(self, sample_config):
        """ray start must not use --block: the script's own wait loop keeps
        the job alive and handles reconnects."""
        pm = self._make_pool(sample_config)
        r = sample_config.slurm.resources[0]
        script = pm._build_sbatch_script(r, "1:00:00")
        assert "--block" not in script

    def test_reconnect_loop(self, sample_config):
        """Workers should use a reconnect loop instead of sleep infinity."""
        pm = self._make_pool(sample_config)
        r = sample_config.slurm.resources[0]
        script = pm._build_sbatch_script(r, "1:00:00")
        assert "while true; do" in script
        assert "ray status" in script
        assert "Hive stopped" in script
        assert 'if ! ray start --address="$_RAY_HEAD"' in script
        assert "_RESTART_GRACE_SEC=330" in script
        assert "_RESTART_GRACE_UNTIL=0" in script
        assert "_CONNECT_FAILS=0" in script
        assert "_MAX_CONNECT_FAILS=12" in script
        assert 'ray start failed for stale head $_RAY_HEAD — retrying in 5s...' in script
        assert 'ray start still failing during restart grace for $_RAY_HEAD — retrying in 5s...' in script
        assert 'ray start failed for $_RAY_HEAD after $_CONNECT_FAILS attempts — exiting for SLURM requeue' in script
        assert '_RESTART_TS=$(cat $_RESTART_SIGNAL 2>/dev/null || echo 0)' in script
        assert '_RESTART_GRACE_UNTIL=$(($(date +%s) + $_RESTART_GRACE_SEC))' in script

    def test_pretimeout_drain_signal_and_trap(self, sample_config):
        pm = self._make_pool(sample_config)
        r = sample_config.slurm.resources[0]
        script = pm._build_sbatch_script(r, "1:00:00")
        assert "#SBATCH --signal=B:USR1@300" in script
        assert "trap _hive_drain USR1" in script
        assert "python -m hive.node_drain" in script
        assert '--state-path "$_STATE_PATH"' in script
        assert "--resource-name test-gpu" in script
        assert 'ray stop --grace-period=30' in script
        assert "if ! python -m hive.node_drain" in script
        assert "WARN: node_drain failed; leaving worker running" in script

    def test_venv_activation(self, sample_config):
        pm = self._make_pool(sample_config)
        r = sample_config.slurm.resources[0]
        script = pm._build_sbatch_script(r, "1:00:00")
        assert "source /tmp/test-venv/bin/activate" in script

    def test_resource_without_partition(self, sample_config):
        """Resource without explicit partition should omit #SBATCH --partition."""
        r = ResourceSpec(name="nop", gpus_per_job=1)
        pm = self._make_pool(sample_config)
        script = pm._build_sbatch_script(r, "1:00:00")
        assert "#SBATCH --partition" not in script

    def test_resource_without_constraint(self, sample_config):
        r = ResourceSpec(name="nop", gpus_per_job=1)
        pm = self._make_pool(sample_config)
        script = pm._build_sbatch_script(r, "1:00:00")
        assert "#SBATCH --constraint" not in script


class TestSlurmJob:
    """SlurmJob dataclass basics."""

    def test_default_fields(self):
        j = SlurmJob(job_id="123", state="PENDING")
        assert j.node is None
        assert j.gpus == 0
        assert j.partition == ""
        assert j.resource_name == ""

    def test_fields_set(self):
        j = SlurmJob(
            job_id="456",
            state="RUNNING",
            node="node01",
            gpus=4,
            partition="gpu",
            resource_name="prod-gpu",
        )
        assert j.job_id == "456"
        assert j.state == "RUNNING"
        assert j.node == "node01"
        assert j.gpus == 4
