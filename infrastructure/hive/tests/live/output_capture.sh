#!/bin/bash
# Test that stdout and stderr are correctly captured in job logs.
# Usage: bash tests/live/output_capture.sh
#
# Prerequisites: hive start --test-only (or any running cluster)
# Submits 3 small jobs, waits, then verifies output files.

set -e

echo "=== Testing stdout/stderr capture ==="
echo ""

# Job 1: stdout only (exit 0)
echo "[1/3] Submitting stdout-only job..."
OUT1=$(hive submit "echo 'CAPTURE_TEST_STDOUT: success'")
JOB1=$(echo "$OUT1" | grep "Job ID:" | awk '{print $NF}')
echo "  Job ID: $JOB1"

# Job 2: stderr only (exit 1)
echo "[2/3] Submitting stderr-only job..."
OUT2=$(hive submit "echo 'CAPTURE_TEST_STDERR: failure' >&2; exit 1")
JOB2=$(echo "$OUT2" | grep "Job ID:" | awk '{print $NF}')
echo "  Job ID: $JOB2"

# Job 3: both stdout and stderr (exit 0)
echo "[3/3] Submitting mixed output job..."
OUT3=$(hive submit "echo 'CAPTURE_TEST_BOTH_OUT: hello'; echo 'CAPTURE_TEST_BOTH_ERR: world' >&2")
JOB3=$(echo "$OUT3" | grep "Job ID:" | awk '{print $NF}')
echo "  Job ID: $JOB3"

echo ""
echo "Waiting for jobs to complete..."
sleep 15

# Verify
PASS=0
FAIL=0

check() {
    local label="$1" job_id="$2" file="$3" marker="$4"
    local HIVE_DIR="${HIVE_DIR:-$(dirname "$(dirname "$(dirname "$0")")")/.hive}"
    local path="$HIVE_DIR/jobs/$job_id/$file"
    if [ ! -f "$path" ]; then
        echo "  FAIL: $label — $path does not exist"
        FAIL=$((FAIL+1))
        return
    fi
    if grep -q "$marker" "$path" 2>/dev/null; then
        echo "  PASS: $label"
        PASS=$((PASS+1))
    else
        echo "  FAIL: $label — marker '$marker' not found in $path"
        echo "        Contents: $(head -3 "$path")"
        FAIL=$((FAIL+1))
    fi
}

echo ""
echo "=== Checking output files ==="

check "Job1 stdout" "$JOB1" "stdout.log" "CAPTURE_TEST_STDOUT: success"
check "Job2 stderr" "$JOB2" "stderr.log" "CAPTURE_TEST_STDERR: failure"
check "Job3 stdout" "$JOB3" "stdout.log" "CAPTURE_TEST_BOTH_OUT: hello"
check "Job3 stderr" "$JOB3" "stderr.log" "CAPTURE_TEST_BOTH_ERR: world"

# Also verify via hive logs
echo ""
echo "=== Checking via 'hive logs' ==="

LOGS1=$(hive logs "$JOB1" 2>&1)
if echo "$LOGS1" | grep -q "CAPTURE_TEST_STDOUT: success"; then
    echo "  PASS: hive logs Job1 stdout"
    PASS=$((PASS+1))
else
    echo "  FAIL: hive logs Job1 — marker not found"
    FAIL=$((FAIL+1))
fi

LOGS2=$(hive logs "$JOB2" 2>&1)
if echo "$LOGS2" | grep -q "CAPTURE_TEST_STDERR: failure"; then
    echo "  PASS: hive logs Job2 stderr"
    PASS=$((PASS+1))
else
    echo "  FAIL: hive logs Job2 — marker not found"
    FAIL=$((FAIL+1))
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
