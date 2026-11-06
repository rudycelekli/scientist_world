"""Tests for nexus.client: NexusClient HTTP client.

The client is exercised over real HTTP against a mock daemon that
listens on an ephemeral loopback port.
"""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

import pytest

from nexus.client import NexusClient


class _MockHandler(BaseHTTPRequestHandler):
    """Minimal mock Nexus daemon for client tests."""

    def do_GET(self):
        if self.path == "/api/status":
            self._respond(200, {
                "daemon": {"uptime": 100, "pid": 1234},
                "hive": {"url": "http://hive.invalid:38581", "connected": True},
                "projects": {"total": 0, "active": 0, "budget_exhausted": 0},
                "jobs": {"running": 0, "completed": 0, "failed": 0},
            })
        elif self.path.startswith("/api/projects/") and "/jobs" in self.path:
            self._respond(200, [{"run_id": "abc123", "status": "completed"}])
        elif self.path == "/api/projects/missing":
            self._respond(404, {"error": "Project 'missing' not found"})
        elif self.path.startswith("/api/projects/"):
            project_id = self.path.split("/")[3]
            self._respond(200, {"project_id": project_id, "max_gpus": 4})
        elif self.path.startswith("/api/jobs/"):
            run_id = self.path.split("/")[3]
            self._respond(200, {"run_id": run_id, "status": "completed"})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length else {}

        if self.path == "/api/projects":
            pid = body.get("project_id", "test")
            self._respond(201, {"project_id": pid, "status": "registered"})
        elif self.path.endswith("/jobs"):
            self._respond(201, {"run_id": "job123", "status": "queued"})
        else:
            self._respond(404, {"error": "not found"})

    def do_DELETE(self):
        if self.path.startswith("/api/projects/"):
            project_id = self.path.split("/")[3]
            self._respond(200, {"project_id": project_id, "status": "deleted"})
        else:
            self._respond(404, {"error": "not found"})

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass  # Suppress request logs


@pytest.fixture(scope="module")
def mock_server():
    """Start a mock HTTP server and return NexusClient."""
    server = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield NexusClient(f"http://127.0.0.1:{port}")
    server.shutdown()


class TestNexusClient:
    def test_status(self, mock_server):
        result = mock_server.status()
        assert result["daemon"]["uptime"] == 100
        assert result["hive"]["connected"] is True

    def test_register_project(self, mock_server):
        result = mock_server.register_project(
            project_id="test-01",
            max_gpus=2,
            workspace_dir="/ws",
            sif_path="/sif",
        )
        assert result["project_id"] == "test-01"
        assert result["status"] == "registered"

    def test_register_project_with_cpu_memory(self, mock_server):
        result = mock_server.register_project(
            project_id="test-02",
            max_gpus=4,
            max_cpus=8,
            max_memory_mb=4096,
        )
        assert result["status"] == "registered"

    def test_get_project(self, mock_server):
        result = mock_server.get_project("test-01")
        assert result["project_id"] == "test-01"
        assert result["max_gpus"] == 4

    def test_delete_project(self, mock_server):
        result = mock_server.delete_project("test-01")
        assert result["status"] == "deleted"

    def test_list_jobs(self, mock_server):
        jobs = mock_server.list_jobs("test-01")
        assert isinstance(jobs, list)
        assert len(jobs) == 1
        assert jobs[0]["run_id"] == "abc123"

    def test_get_job(self, mock_server):
        result = mock_server.get_job("abc123")
        assert result["run_id"] == "abc123"
        assert result["status"] == "completed"

    def test_error_body_raises(self, mock_server):
        """A JSON error body from the daemon becomes a RuntimeError."""
        with pytest.raises(RuntimeError, match="not found"):
            mock_server.get_project("missing")

    def test_connection_refused(self):
        # Port 1 on the loopback interface: nothing listens there.
        client = NexusClient("http://127.0.0.1:1")
        with pytest.raises(Exception):
            client.status()
