"""Tests for targeted preemption of low-priority jobs."""

import json
from unittest.mock import MagicMock, patch

from hive.dispatcher import JOB_ACTOR_PREFIX, _targeted_preempt


def _write_job(hive_dir, job_id, *, node_id, num_gpus=1, priority="normal", status="running"):
    job_dir = hive_dir / "jobs" / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps({
        "job_id": job_id, "node_id": node_id, "num_gpus": num_gpus,
        "priority": priority, "status": status,
    }))


def _node(node_id, gpus, partition):
    return {"NodeID": node_id, "Alive": True,
            "Resources": {"GPU": gpus, "CPU": 16, f"partition:{partition}": 100.0}}


def _run(nodes, gpus_needed, bundles):
    """Run _targeted_preempt with Ray mocked; return (killed job ids, sleep calls)."""
    actors = {}

    def get_actor(name):
        return actors.setdefault(name, MagicMock(name=name))

    with patch("hive.dispatcher.ray") as mock_ray, \
         patch("hive.dispatcher.time.sleep") as mock_sleep:
        mock_ray.nodes.return_value = nodes
        mock_ray.util.list_named_actors.side_effect = lambda all_namespaces=True: [
            {"name": f"{JOB_ACTOR_PREFIX}{jid}"} for jid in ALL_JOB_IDS
        ]
        mock_ray.get_actor.side_effect = get_actor
        _targeted_preempt(gpus_needed, bundles)
        killed = {c.args[0] for c in mock_ray.kill.call_args_list}
    killed_ids = sorted(name[len(JOB_ACTOR_PREFIX):] for name, a in actors.items() if a in killed)
    return killed_ids, mock_sleep.call_count


ALL_JOB_IDS = []


def setup_function(_fn):
    ALL_JOB_IDS.clear()


def _job(hive_dir, job_id, **kw):
    ALL_JOB_IDS.append(job_id)
    _write_job(hive_dir, job_id, **kw)


def test_no_low_priority_jobs_is_a_noop(tmp_hive_dir):
    _job(tmp_hive_dir, "normal1", node_id="nodeA", num_gpus=4)
    killed, sleeps = _run([_node("nodeA", 4, "gpu-a")], 2, [{"CPU": 1, "GPU": 2}])
    assert killed == []
    assert sleeps == 0


def test_kills_low_priority_job_to_free_gpus(tmp_hive_dir):
    _job(tmp_hive_dir, "low1", node_id="nodeA", num_gpus=2, priority="low")
    _job(tmp_hive_dir, "normal1", node_id="nodeA", num_gpus=2)
    killed, sleeps = _run([_node("nodeA", 4, "gpu-a")], 2, [{"CPU": 1, "GPU": 2}])
    assert killed == ["low1"]
    assert sleeps == 1


def test_idle_gpus_count_towards_the_request(tmp_hive_dir):
    """A 4-GPU request fits on a node with 2 idle GPUs and one 2-GPU low-priority job."""
    _job(tmp_hive_dir, "low1", node_id="nodeA", num_gpus=2, priority="low")
    killed, _ = _run([_node("nodeA", 4, "gpu-a")], 4, [{"CPU": 1, "GPU": 4}])
    assert killed == ["low1"]


def test_does_not_kill_when_node_cannot_satisfy_request(tmp_hive_dir):
    """2 GPUs held by a normal job can not be freed: nothing is killed."""
    _job(tmp_hive_dir, "low1", node_id="nodeA", num_gpus=1, priority="low")
    _job(tmp_hive_dir, "normal1", node_id="nodeA", num_gpus=3)
    killed, sleeps = _run([_node("nodeA", 4, "gpu-a")], 2, [{"CPU": 1, "GPU": 2}])
    assert killed == []
    assert sleeps == 0


def test_kills_only_as_many_jobs_as_needed(tmp_hive_dir):
    for i in range(3):
        _job(tmp_hive_dir, f"low{i}", node_id="nodeA", num_gpus=1, priority="low")
    _job(tmp_hive_dir, "normal1", node_id="nodeA", num_gpus=1)
    killed, _ = _run([_node("nodeA", 4, "gpu-a")], 2, [{"CPU": 1, "GPU": 2}])
    assert len(killed) == 2


def test_prefers_node_with_fewest_low_priority_jobs(tmp_hive_dir):
    _job(tmp_hive_dir, "a1", node_id="nodeA", num_gpus=1, priority="low")
    _job(tmp_hive_dir, "a2", node_id="nodeA", num_gpus=1, priority="low")
    _job(tmp_hive_dir, "b1", node_id="nodeB", num_gpus=2, priority="low")
    nodes = [_node("nodeA", 2, "gpu-a"), _node("nodeB", 2, "gpu-a")]
    killed, _ = _run(nodes, 2, [{"CPU": 1, "GPU": 2}])
    assert killed == ["b1"]


def test_respects_partition_constraint(tmp_hive_dir):
    _job(tmp_hive_dir, "low1", node_id="nodeB", num_gpus=2, priority="low")
    nodes = [_node("nodeA", 2, "gpu-a"), _node("nodeB", 2, "gpu-b")]
    killed, _ = _run(nodes, 2, [{"CPU": 1, "GPU": 2, "partition:gpu-a": 0.01}])
    assert killed == []
    killed, _ = _run(nodes, 2, [{"CPU": 1, "GPU": 2, "partition:gpu-b": 0.01}])
    assert killed == ["low1"]


def test_queued_and_finished_jobs_are_ignored(tmp_hive_dir):
    _job(tmp_hive_dir, "low-queued", node_id="nodeA", num_gpus=2, priority="low", status="queued")
    _job(tmp_hive_dir, "low-done", node_id="nodeA", num_gpus=2, priority="low", status="completed")
    killed, _ = _run([_node("nodeA", 4, "gpu-a")], 2, [{"CPU": 1, "GPU": 2}])
    assert killed == []
