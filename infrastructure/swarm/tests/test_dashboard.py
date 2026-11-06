"""Tests for the dashboard REST API against a scheduler without threads."""

import threading
from collections import deque

import pytest

from swarm.dashboard import create_app
from swarm.scheduler import PaperJob, Scheduler


@pytest.fixture
def scheduler(sample_config):
    """A Scheduler with real capacity logic but no daemon side effects."""
    sched = Scheduler.__new__(Scheduler)
    sched.config = sample_config
    sched._lock = threading.Lock()
    sched._capacity_cv = threading.Condition(sched._lock)
    sched._capacity_state = {}
    sched._cap_overrides = {}
    sched._shutdown = False
    sched.queue = deque()
    sched.active = {}
    sched.completed = {}
    sched._drain_dispatch = lambda: 0
    # set_cap only reads the executor's size to decide whether to grow it.
    sched._executor = type("Executor", (), {"_max_workers": 64})()
    return sched


@pytest.fixture
def client(sample_config, scheduler):
    return create_app(sample_config, scheduler).test_client()


def test_endpoints_need_a_scheduler(sample_config):
    client = create_app(sample_config, scheduler=None).test_client()
    assert client.get("/").status_code == 200
    for path in ("/api/status", "/api/papers", "/api/workers", "/api/caps", "/api/dashboard"):
        assert client.get(path).status_code == 503, path


def test_status_and_workers_report_capacity(client, scheduler):
    job = PaperJob(arxiv_id="p1", paper=None, workspace_path="", gpu_budget=1.0)
    job.worker_status = "waiting"
    scheduler.active = {"p1": job}

    status = client.get("/api/status").get_json()
    assert status["active_count"] == 1
    assert status["waiting_count"] == 1
    assert status["nexus_url"] == "http://localhost:39000"
    assert "max_waiting_workers" not in status

    workers = client.get("/api/workers")
    assert workers.status_code == 200
    body = workers.get_json()
    assert body["active"] == 1
    assert body["idle"] == body["pool_size"] - 1
    assert "max_waiting_workers" not in body


def test_caps_can_be_set_and_cleared(client):
    caps = client.get("/api/caps").get_json()
    assert set(caps["effective"]) == {
        "total", "working", "overflow", "admitted", "building", "installing",
    }
    assert caps["overrides"] == {}

    caps = client.post("/api/caps", json={"bucket": "working", "value": 1}).get_json()
    assert caps["overrides"] == {"working": 1}
    assert caps["effective"]["working"] == 1

    assert client.post("/api/caps", json={"bucket": "waiting", "value": 1}).status_code == 400
    assert client.post("/api/caps", json={"value": 1}).status_code == 400

    caps = client.delete("/api/caps").get_json()
    assert caps["overrides"] == {}


def test_paper_lookup_and_cancel_of_unknown_paper(client, tmp_swarm_dir):
    assert client.get("/api/papers").get_json() == []
    assert client.get("/api/papers/0000.00000").status_code == 404
    assert client.post("/api/papers/0000.00000/cancel").status_code == 404
    assert client.post("/api/papers", json={}).status_code == 400
