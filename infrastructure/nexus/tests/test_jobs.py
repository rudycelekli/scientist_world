"""Tests for nexus.jobs: NexusJob persistence, workspace writes."""

import json
import time
from pathlib import Path

from nexus.jobs import NexusJob


class TestJobSerialization:
    """NexusJob dataclass serialization round-trip."""

    def test_to_dict_from_dict_roundtrip(self):
        j = NexusJob(
            run_id="abc12345",
            project_id="proj001",
            command="python train.py",
            num_gpus=2,
            num_cpus=4,
            hive_job_id="hive-xyz",
            status="running",
            submitted_at=1000.0,
            started_at=1001.0,
        )
        d = j.to_dict()
        restored = NexusJob.from_dict(d)
        assert restored.run_id == "abc12345"
        assert restored.project_id == "proj001"
        assert restored.command == "python train.py"
        assert restored.num_gpus == 2
        assert restored.num_cpus == 4
        assert restored.hive_job_id == "hive-xyz"
        assert restored.status == "running"

    def test_from_dict_defaults(self):
        j = NexusJob.from_dict({"run_id": "x", "project_id": "p"})
        assert j.num_gpus == 1
        assert j.num_cpus == 1
        assert j.status == "queued"
        assert j.hive_job_id is None
        assert j.checkpoint_count == 0

    def test_auto_generated_run_id(self):
        j = NexusJob(project_id="p")
        assert len(j.run_id) == 8
        j2 = NexusJob(project_id="p")
        assert j.run_id != j2.run_id  # should be unique


class TestJobPersistence:
    """Save, load, delete operations."""

    def test_save_and_load(self, tmp_nexus_dir):
        j = NexusJob(
            run_id="save01",
            project_id="proj",
            command="echo test",
            num_gpus=1,
            status="queued",
        )
        j.save()

        loaded = NexusJob.load("save01")
        assert loaded is not None
        assert loaded.run_id == "save01"
        assert loaded.command == "echo test"

    def test_load_nonexistent_returns_none(self, tmp_nexus_dir):
        assert NexusJob.load("nonexistent") is None

    def test_load_all_empty(self, tmp_nexus_dir):
        assert NexusJob.load_all() == []

    def test_load_all_multiple(self, tmp_nexus_dir):
        for i in range(3):
            NexusJob(run_id=f"job{i}", project_id="proj", command=f"echo {i}").save()

        jobs = NexusJob.load_all()
        assert len(jobs) == 3
        ids = {j.run_id for j in jobs}
        assert ids == {"job0", "job1", "job2"}

    def test_load_by_project(self, tmp_nexus_dir):
        NexusJob(run_id="j1", project_id="A", command="echo A1").save()
        NexusJob(run_id="j2", project_id="B", command="echo B1").save()
        NexusJob(run_id="j3", project_id="A", command="echo A2").save()

        a_jobs = NexusJob.load_by_project("A")
        assert len(a_jobs) == 2
        assert all(j.project_id == "A" for j in a_jobs)

        b_jobs = NexusJob.load_by_project("B")
        assert len(b_jobs) == 1

    def test_save_overwrites(self, tmp_nexus_dir):
        j = NexusJob(run_id="upd01", project_id="proj", command="echo")
        j.save()

        j.status = "running"
        j.started_at = time.time()
        j.save()

        loaded = NexusJob.load("upd01")
        assert loaded.status == "running"
        assert loaded.started_at is not None

    def test_delete(self, tmp_nexus_dir):
        j = NexusJob(run_id="del01", project_id="proj", command="echo")
        j.save()
        assert NexusJob.load("del01") is not None

        j.delete()
        assert NexusJob.load("del01") is None


class TestJobLifecycle:
    """Simulate a job going through status transitions."""

    def test_full_lifecycle(self, tmp_nexus_dir):
        job = NexusJob(
            run_id="lc01",
            project_id="proj",
            command="python train.py",
            num_gpus=2,
            status="queued",
        )
        job.save()
        assert NexusJob.load("lc01").status == "queued"

        job.status = "running"
        job.started_at = time.time()
        job.save()
        assert NexusJob.load("lc01").status == "running"

        job.status = "completed"
        job.completed_at = time.time()
        job.returncode = 0
        job.save()

        final = NexusJob.load("lc01")
        assert final.status == "completed"
        assert final.returncode == 0
        assert final.completed_at >= final.started_at


class TestWorkspaceWrites:
    """Workspace file output for agent consumption."""

    def test_write_meta(self, tmp_nexus_dir):
        workspace = str(tmp_nexus_dir / "workspace")
        j = NexusJob(
            run_id="ws01",
            project_id="proj",
            command="python train.py",
            num_gpus=2,
            status="running",
        )
        j.write_meta(workspace)

        meta_path = Path(workspace) / "jobs" / "ws01" / "meta.json"
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text())
        assert meta["run_id"] == "ws01"
        assert meta["command"] == "python train.py"
        assert meta["num_gpus"] == 2

    def test_write_logs(self, tmp_nexus_dir):
        workspace = str(tmp_nexus_dir / "workspace")
        j = NexusJob(run_id="ws02", project_id="proj")
        j.write_logs(workspace, "stdout content\n", "stderr content\n")

        jobs_dir = Path(workspace) / "jobs" / "ws02"
        assert (jobs_dir / "stdout.log").read_text() == "stdout content\n"
        assert (jobs_dir / "stderr.log").read_text() == "stderr content\n"

    def test_write_checkpoint(self, tmp_nexus_dir):
        workspace = str(tmp_nexus_dir / "workspace")
        j = NexusJob(run_id="ws03", project_id="proj", checkpoint_count=0)

        j.write_checkpoint(workspace, "epoch 1 output", "")
        assert j.checkpoint_count == 1

        cp_path = Path(workspace) / "jobs" / "ws03" / "checkpoints" / "001.json"
        assert cp_path.exists()
        cp = json.loads(cp_path.read_text())
        assert "timestamp" in cp
        assert cp["checkpoint"] == 1
        assert "epoch 1" in cp["stdout"]

        # Second checkpoint
        j.write_checkpoint(workspace, "epoch 2 output", "")
        assert j.checkpoint_count == 2
        cp2_path = Path(workspace) / "jobs" / "ws03" / "checkpoints" / "002.json"
        assert cp2_path.exists()

    def test_write_result(self, tmp_nexus_dir):
        workspace = str(tmp_nexus_dir / "workspace")
        j = NexusJob(
            run_id="ws04",
            project_id="proj",
            command="python train.py",
            num_gpus=2,
            status="completed",
            submitted_at=1000.0,
            started_at=1001.0,
            completed_at=1100.0,
        )
        j.write_result(workspace, "done\n", "", returncode=0)

        result_path = Path(workspace) / "jobs" / "ws04" / "result.json"
        assert result_path.exists()
        result = json.loads(result_path.read_text())
        assert result["run_id"] == "ws04"
        assert result["status"] == "completed"
        assert result["returncode"] == 0
        assert result["duration"] == 99.0  # 1100 - 1001
