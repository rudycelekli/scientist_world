#!/usr/bin/env bash
# Manual end-to-end smoke test for a Nexus deployment.
#
# This script is NOT part of the pytest suite: it needs real infrastructure.
# It registers a throwaway project, submits one "echo" job through the agent
# CLI, waits for it, checks its output and removes the project again. It does
# not start or stop the daemon.
#
# Requirements:
#   - a running Hive pool and a running Nexus daemon (`nexus start`)
#   - the `nexus` and `nx` commands on PATH (pip install -e infrastructure/nexus)
#   - Apptainer on the compute nodes
#   - a SIF image that contains bash, for example:
#       apptainer build smoke.sif docker://ubuntu:22.04
#   - a workspace directory on a filesystem that the daemon host and the
#     compute nodes both see
#
# Usage:
#   tests/manual_smoke.sh <workspace-dir> <sif-path> <hive-resource> [num-gpus]
#
# num-gpus defaults to 1; use 0 for a CPU-only job. The agent CLI reads the
# daemon address from NEXUS_URL (default here: http://127.0.0.1:39000); the
# admin CLI talks to the local daemon on the port from the Nexus config.
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <workspace-dir> <sif-path> <hive-resource> [num-gpus]" >&2
    exit 2
fi

WORKSPACE="$1"
SIF="$2"
RESOURCE="$3"
GPUS="${4:-1}"
MAX_GPUS=$(( GPUS > 0 ? GPUS : 1 ))
PROJECT="nexus-smoke-$$"

export NEXUS_URL="${NEXUS_URL:-http://127.0.0.1:39000}"
export NEXUS_PROJECT_ID="$PROJECT"

mkdir -p "$WORKSPACE"

echo "== 1. daemon status =="
nexus status

echo
echo "== 2. register project $PROJECT =="
nexus register --project-id "$PROJECT" --max-gpus "$MAX_GPUS" \
    --workspace "$WORKSPACE" --sif "$SIF" --resource "$RESOURCE" --gpu-budget 1
# Deleting the project also cancels a job that is still active.
trap 'nexus deregister "$PROJECT" >/dev/null 2>&1 || true' EXIT
nexus project "$PROJECT"

echo
echo "== 3. submit =="
SUBMIT_OUT="$(nx submit "echo hello from nexus" --gpus "$GPUS")"
echo "$SUBMIT_OUT"
RUN_ID="$(awk '/^run_id:/ {print $2}' <<<"$SUBMIT_OUT")"
if [[ -z "$RUN_ID" ]]; then
    echo "FAILED: could not read the run_id from the submit output" >&2
    exit 1
fi

echo
echo "== 4. wait =="
# One wait returns after at most 600 s. A checkpoint is answered with
# "continue". Give up after 12 rounds (two hours).
STATUS=""
for _ in $(seq 1 12); do
    WAIT_OUT="$(nx wait --run-id "$RUN_ID" --timeout 600)"
    echo "$WAIT_OUT"
    STATUS="$(awk '/^STATUS:/ {print $2}' <<<"$WAIT_OUT")"
    case "$STATUS" in
        completed|failed|cancelled) break ;;
        awaiting_response) nx continue "$RUN_ID" ;;
        *) ;;  # timeout: wait again
    esac
done

echo
echo "== 5. result =="
nx status "$RUN_ID"
nx info

# The daemon writes the job's files into the workspace right after it marks
# the job as ended; result.json is written last.
JOB_DIR="$WORKSPACE/jobs/$RUN_ID"
for _ in $(seq 1 30); do
    [[ -f "$JOB_DIR/result.json" ]] && break
    sleep 1
done

if [[ "$STATUS" != "completed" ]]; then
    echo "FAILED: job ended with status '${STATUS:-none}'" >&2
    [[ -f "$JOB_DIR/stderr.log" ]] && cat "$JOB_DIR/stderr.log" >&2
    exit 1
fi
if ! grep -q "hello from nexus" "$JOB_DIR/stdout.log"; then
    echo "FAILED: expected output not found in $JOB_DIR/stdout.log" >&2
    exit 1
fi

echo
echo "SMOKE TEST PASSED (job $RUN_ID, output in $JOB_DIR)"
