"""Tests for nexus.hive_bridge: Hive API client and apptainer wrapping."""

import urllib.error
from unittest.mock import patch

import pytest

from nexus.hive_bridge import HiveBridge, _shell_quote
from nexus.project import Project

from .conftest import FAKE_HIVE_URL


def _http_error(code):
    return urllib.error.HTTPError(f"{FAKE_HIVE_URL}/api/submit", code, "error", {}, None)


class TestShellQuote:
    """_shell_quote escapes special characters."""

    def test_simple_string_not_quoted(self):
        assert _shell_quote("hello") == "hello"

    def test_path_not_quoted(self):
        assert _shell_quote("/home/user/train.py") == "/home/user/train.py"

    def test_flag_not_quoted(self):
        assert _shell_quote("--nv") == "--nv"

    def test_spaces_quoted(self):
        result = _shell_quote("hello world")
        assert result.startswith("'")
        assert result.endswith("'")

    def test_single_quotes_escaped(self):
        result = _shell_quote("it's")
        assert "\\'" in result


class TestBuildApptainerCommand:
    """Apptainer command generation from project config."""

    def _make_project(self, **kwargs):
        defaults = dict(
            project_id="test",
            max_gpus=4,
            workspace_dir="/workspaces/test",
            sif_path="/containers/test.sif",
            shared_dirs=[],
        )
        defaults.update(kwargs)
        return Project(**defaults)

    def test_basic_gpu_command(self):
        p = self._make_project()
        cmd = HiveBridge.build_apptainer_command(p, "python train.py", num_gpus=2)
        assert "apptainer exec" in cmd
        assert "--nv" in cmd
        assert "--containall" in cmd
        assert "--cleanenv" in cmd
        assert "--bind /workspaces/test:/home/user" in cmd
        assert "/containers/test.sif" in cmd
        assert "python train.py" in cmd

    def test_no_gpu_no_nv(self):
        p = self._make_project()
        cmd = HiveBridge.build_apptainer_command(p, "python eval.py", num_gpus=0)
        assert "--nv" not in cmd

    def test_shared_dirs(self):
        p = self._make_project(shared_dirs=["/shared:/shared:ro", "/data:/data"])
        cmd = HiveBridge.build_apptainer_command(p, "echo hi", num_gpus=0)
        assert "--bind /shared:/shared:ro" in cmd
        assert "--bind /data:/data" in cmd

    def test_custom_workdir(self):
        p = self._make_project()
        cmd = HiveBridge.build_apptainer_command(
            p, "ls", num_gpus=0, working_directory="/workspace"
        )
        assert "cd /workspace" in cmd

    def test_full_command(self):
        """The exact command handed to Hive (documented in the README)."""
        p = self._make_project(
            overlay_path="/images/overlay.img",
            shared_dirs=["/data:/data:ro"],
            hf_cache_path="/cache/hf",
        )
        cmd = HiveBridge.build_apptainer_command(
            p, "python train.py --lr 1e-4", num_gpus=1)
        assert cmd == (
            "apptainer exec --nv --containall --cleanenv "
            "--env HF_HOME=/home/user/shared/hf_cache "
            "--env TRANSFORMERS_CACHE=/home/user/shared/hf_cache "
            "--env HF_DATASETS_CACHE=/home/user/shared/hf_cache/datasets "
            "--env TORCH_HOME=/home/user/shared/hf_cache/torch "
            "--bind /workspaces/test:/home/user "
            "--overlay /images/overlay.img:ro "
            "--bind /data:/data:ro "
            "--bind /cache/hf:/home/user/shared/hf_cache "
            "/containers/test.sif bash -c "
            "'cd /home/user && python train.py --lr 1e-4'"
        )


class TestHiveBridgeSubmit:
    """HiveBridge.submit and submit_for_project."""

    def test_submit_calls_post(self):
        bridge = HiveBridge(FAKE_HIVE_URL)
        with patch.object(bridge, "_post", return_value={"job_id": "abc123"}) as mock:
            job_id = bridge.submit("echo hi", num_gpus=1)
        assert job_id == "abc123"
        mock.assert_called_once()
        call_args = mock.call_args
        assert call_args[0][0] == "/api/submit"
        data = call_args[0][1]
        assert data["command"] == "echo hi"
        assert data["num_gpus"] == 1

    def test_submit_payload(self):
        """The fields Nexus sends to Hive's /api/submit."""
        bridge = HiveBridge(FAKE_HIVE_URL)
        with patch.object(bridge, "_post", return_value={"job_id": "j"}) as post:
            bridge.submit("echo hi", num_gpus=2, num_cpus=4)
            minimal = post.call_args[0][1]
            bridge.submit("echo hi", timeout_sec=60, resource="gpu-a,gpu-b")
            full = post.call_args[0][1]
        assert set(minimal) == {"command", "num_gpus", "num_cpus", "user", "max_retries"}
        assert minimal["num_cpus"] == 4
        assert minimal["max_retries"] == 3
        assert minimal["user"].startswith("nexus/")
        assert full["timeout_sec"] == 60
        assert full["resource"] == "gpu-a,gpu-b"

    def test_submit_raises_on_error(self):
        bridge = HiveBridge(FAKE_HIVE_URL)
        with patch.object(bridge, "_post", return_value={"error": "no GPUs"}):
            with pytest.raises(RuntimeError, match="no GPUs"):
                bridge.submit("echo hi")

    def test_submit_retries_a_transient_failure(self):
        bridge = HiveBridge(FAKE_HIVE_URL)
        responses = [_http_error(503), urllib.error.URLError("refused"),
                     {"job_id": "abc123"}]
        with patch.object(bridge, "_post", side_effect=responses) as post, \
             patch("nexus.hive_bridge.time.sleep") as sleep:
            assert bridge.submit("echo hi") == "abc123"
        assert post.call_count == 3
        assert [call[0][0] for call in sleep.call_args_list] == [5, 10]

    def test_submit_does_not_retry_a_rejection(self):
        bridge = HiveBridge(FAKE_HIVE_URL)
        with patch.object(bridge, "_post", side_effect=_http_error(400)) as post, \
             patch("nexus.hive_bridge.time.sleep") as sleep:
            with pytest.raises(RuntimeError, match="HTTP 400"):
                bridge.submit("echo hi")
        assert post.call_count == 1
        sleep.assert_not_called()

    def test_submit_for_project(self):
        bridge = HiveBridge(FAKE_HIVE_URL)
        project = Project(
            project_id="test", max_gpus=4,
            workspace_dir="/ws", sif_path="/test.sif",
            allowed_resources=["gpu", "cpu"],
        )
        with patch.object(bridge, "submit", return_value="job456") as mock:
            job_id = bridge.submit_for_project(
                project, "python train.py", num_gpus=2, resource="gpu",
            )
        assert job_id == "job456"
        call_args = mock.call_args
        assert "apptainer exec" in call_args[1].get("command", call_args[0][0])
        assert call_args[1]["resource"] == "gpu"


class TestHiveBridgePoll:
    """HiveBridge.poll, cancel and cluster_status."""

    def test_poll_returns_status(self):
        bridge = HiveBridge(FAKE_HIVE_URL)
        with patch.object(bridge, "_get", return_value={
            "status": "running", "stdout": "epoch 1", "stderr": "",
        }) as get:
            result = bridge.poll("job123")
        get.assert_called_once_with("/api/jobs/job123/logs")
        assert result["status"] == "running"
        assert result["stdout"] == "epoch 1"

    def test_poll_on_error_returns_unknown(self):
        bridge = HiveBridge(FAKE_HIVE_URL)
        with patch.object(bridge, "_get", side_effect=urllib.error.URLError("connection refused")):
            result = bridge.poll("job123")
        assert result["status"] == "unknown"

    def test_poll_http_error_is_unknown_too(self):
        """A job that Hive no longer knows (404) looks like a transient error."""
        bridge = HiveBridge(FAKE_HIVE_URL)
        with patch.object(bridge, "_get", side_effect=_http_error(404)):
            assert bridge.poll("job123") == {"status": "unknown"}

    def test_cluster_status(self):
        bridge = HiveBridge(FAKE_HIVE_URL)
        with patch.object(bridge, "_get", return_value={
            "total_gpus": 16, "free_gpus": 8,
        }):
            status = bridge.cluster_status()
        assert status["total_gpus"] == 16

    def test_cluster_status_unreachable(self):
        bridge = HiveBridge(FAKE_HIVE_URL)
        with patch.object(bridge, "_get", side_effect=urllib.error.URLError("refused")):
            status = bridge.cluster_status()
        assert "error" in status

    def test_cancel(self):
        bridge = HiveBridge(FAKE_HIVE_URL)
        with patch.object(bridge, "_post", return_value={"status": "cancelled"}) as post:
            assert bridge.cancel("job123") is True
        post.assert_called_once_with("/api/jobs/job123/cancel", {})

    def test_cancel_failure_returns_false(self):
        bridge = HiveBridge(FAKE_HIVE_URL)
        for failure in (urllib.error.URLError("connection refused"), _http_error(503)):
            with patch.object(bridge, "_post", side_effect=failure):
                assert bridge.cancel("job123") is False
        with patch.object(bridge, "_post", return_value={"error": "no such job"}):
            assert bridge.cancel("job123") is False
