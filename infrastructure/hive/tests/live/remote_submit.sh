#!/bin/bash
# Test remote job submission via hive connect.
# Usage: bash tests/live/remote_submit.sh [host:port]
#
# Prerequisites: a running Hive cluster with dashboard (hive start + hive serve)
# Default target: localhost:38581 (same node, self-connect)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
W="$REPO_DIR/tests/workloads"

TARGET="${1:-localhost:38581}"

echo "=== Remote Submit Tests ==="
echo "  Target: $TARGET"
echo ""

# ── 1. Connect ────────────────────────────────────────────────
echo "[1/8] Connecting to remote cluster..."
hive connect "$TARGET"
echo ""

# ── 2. Cluster status ────────────────────────────────────────
echo "[2/8] Checking cluster status..."
hive status
echo ""

# ── 3. Submit a basic job ─────────────────────────────────────
echo "[3/8] Submitting basic echo job..."
hive submit "echo 'REMOTE_TEST: hello from remote'"
echo ""

# ── 4. Quiet mode — capture job ID ───────────────────────────
echo "[4/8] Quiet mode submit..."
JOB_ID=$(hive submit -q "echo 'REMOTE_QUIET_TEST: success'")
echo "  Captured job ID: $JOB_ID"
echo ""

# ── 5. Multi-GPU job ─────────────────────────────────────────
echo "[5/8] Submitting 2-GPU job..."
hive submit --gpus 2 "python $W/sleep_job.py --seconds 5"
echo ""

# ── 6. On-complete callback ──────────────────────────────────
CALLBACK_FILE="/tmp/hive_remote_test_$$"
echo "[6/8] Submitting job with --on-complete callback..."
hive submit "echo 'CALLBACK_TEST'" \
  --on-complete "echo HIVE_CALLBACK_FIRED > $CALLBACK_FILE"
echo "  Callback will write to: $CALLBACK_FILE"
echo ""

# ── 7. Wait for jobs, check logs ─────────────────────────────
echo "[7/8] Waiting 15s for jobs to complete..."
sleep 15

echo "  Job list:"
hive jobs
echo ""

echo "  Logs for quiet-mode job ($JOB_ID):"
hive logs "$JOB_ID"
echo ""

# Check on-complete callback
if [ -f "$CALLBACK_FILE" ]; then
    echo "  PASS: on-complete callback fired"
    rm -f "$CALLBACK_FILE"
else
    echo "  INFO: on-complete callback not yet fired (may need more time)"
fi
echo ""

# ── 8. Disconnect ─────────────────────────────────────────────
echo "[8/8] Disconnecting..."
hive disconnect
echo ""

echo ""
echo "=== Remote submit tests complete ==="
