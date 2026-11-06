#!/usr/bin/env python3
"""Exercise the Hive Python client against a live cluster: submit, poll, callback.

Usage:
    python tests/live/remote_client.py [url]

Default URL: http://localhost:38581
Prerequisites: a running Hive cluster with dashboard.
"""

import sys
import threading
import time

from hive.client import Client

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:38581"


def main():
    print("=== Hive Python Client Test ===")
    print(f"  URL: {URL}")
    print()

    client = Client(URL)

    # ── 1. Cluster status ─────────────────────────────────────
    print("[1/5] Checking cluster status...")
    status = client.status()
    print(f"  Total GPUs:   {status['total_gpus']}")
    print(f"  Available:    {status.get('available_gpu', '?')} GPUs")
    print(f"  Nodes:        {status.get('num_gpu_nodes', 0) + status.get('num_cpu_nodes', 0)}")
    print()

    # ── 2. Submit + wait (blocking) ───────────────────────────
    print("[2/5] Submit + wait (blocking)...")
    job_id = client.submit("echo 'CLIENT_TEST: hello from python client'")
    print(f"  Submitted: {job_id}")

    result = client.wait(job_id, poll_interval=2.0)
    print(f"  Status:    {result['status']}")
    print(f"  Exit code: {result['returncode']}")
    print(f"  stdout:    {result['stdout'].strip()}")
    assert result["status"] == "completed", f"Expected completed, got {result['status']}"
    assert result["returncode"] == 0, f"Expected exit 0, got {result['returncode']}"
    assert "CLIENT_TEST" in result["stdout"], "Marker not found in stdout"
    print("  PASS")
    print()

    # ── 3. Submit with on_complete callback ───────────────────
    print("[3/5] Submit with on_complete callback...")
    callback_event = threading.Event()
    callback_result = {}

    def on_done(r):
        callback_result.update(r)
        callback_event.set()

    job_id2 = client.submit(
        "echo 'CALLBACK_CLIENT_TEST: fired'",
        on_complete=on_done,
    )
    print(f"  Submitted: {job_id2}")
    print("  Waiting for callback (up to 30s)...")

    if callback_event.wait(timeout=30):
        print(f"  Callback received: status={callback_result['status']}")
        print(f"  stdout: {callback_result['stdout'].strip()}")
        assert callback_result["status"] == "completed"
        assert "CALLBACK_CLIENT_TEST" in callback_result["stdout"]
        print("  PASS")
    else:
        print("  FAIL: callback did not fire within 30s")
        sys.exit(1)
    print()

    # ── 4. List jobs ──────────────────────────────────────────
    print("[4/5] Listing jobs...")
    all_jobs = client.jobs()
    print(f"  Total jobs: {len(all_jobs)}")

    completed = client.jobs(status="completed")
    print(f"  Completed:  {len(completed)}")

    # Our two jobs should be in there
    our_ids = {job_id, job_id2}
    found = {j["job_id"] for j in all_jobs} & our_ids
    assert found == our_ids, f"Expected {our_ids}, found {found}"
    print("  PASS")
    print()

    # ── 5. Logs retrieval ─────────────────────────────────────
    print("[5/5] Fetching logs...")
    logs = client.logs(job_id)
    print(f"  job_id:    {logs['job_id']}")
    print(f"  status:    {logs['status']}")
    print(f"  user:      {logs.get('user', 'N/A')}")
    print(f"  command:   {logs['command']}")
    assert logs["job_id"] == job_id
    assert "remote" in logs.get("user", "").lower(), \
        f"Expected '(remote)' tag in user, got '{logs.get('user')}'"
    print("  PASS")
    print()

    print("=== All tests passed ===")


if __name__ == "__main__":
    main()
