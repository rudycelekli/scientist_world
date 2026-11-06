# Live-cluster checks

These scripts need a running Hive cluster (`hive start ...`). They are not part
of the unit test suite and are not collected by pytest.

| Script | What it does |
|---|---|
| `submit_workloads.sh` | Submits every script in `tests/workloads/` (GPU workloads need `torch` on the worker nodes) plus a few CPU-only jobs. |
| `output_capture.sh` | Checks that stdout and stderr of a job end up in the job record. |
| `remote_submit.sh [host:port]` | Exercises remote mode: `hive connect`, submit, logs, `hive disconnect`. |
| `remote_client.py [url]` | Exercises the Python client (`hive.client.Client`): submit, wait, callback, logs. |

Inspect the results with `hive jobs`, `hive logs <job_id>` or the dashboard.
